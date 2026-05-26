#!/usr/bin/env python3
"""FastAPI live-inference server for V5e-0 demo.

Streams AR vs V5e-0 SSD generation token-by-token via WebSocket, plus serves the
static web UI on the same port.

Usage:
  python server.py \
      --model_path llava-hf/llava-1.5-7b-hf \
      --ckpt ./checkpoints/v5e0_llava-1.5-7b.pt \
      --port 8000 --gpu 0

  Then open http://localhost:8000  (or via VS Code Port Forwarding)
"""
import os, sys, json, time, base64, argparse
from pathlib import Path
from io import BytesIO

import torch
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parent.parent))
from infer import load_v5e0, to_tuple_kv, kv_to_cache


# ============================================================
# Globals (set in __main__)
# ============================================================
v5e0 = None
app = FastAPI()


def save_image_to_tempfile(b64_str):
    """Decode base64 image and save to /tmp for prepare_inputs path."""
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    p = Path("/tmp/demo_upload.jpg")
    img.save(p, "JPEG")
    return str(p)


def incremental_decode(tokenizer, prev_ids, new_ids):
    """Return the *incremental text* added by appending new_ids to prev_ids.

    Avoids the single-token-decode space-stripping issue with SentencePiece-style
    tokenizers (LLaMA / LLaVA): decoding `[t1, t2]` joined gives proper spacing,
    while decoding `[t1]` and `[t2]` separately may drop leading spaces.
    """
    if not new_ids: return ""
    prev_text = tokenizer.decode(prev_ids) if prev_ids else ""
    new_text = tokenizer.decode(prev_ids + new_ids)
    return new_text[len(prev_text):]


# ============================================================
# Static serving (web UI lives on the same port)
# ============================================================
WEB_DIR = Path(__file__).parent / "web"


@app.get("/")
async def root():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/index.html")
async def index_html():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/style.css")
async def style_css():
    return FileResponse(WEB_DIR / "style.css")


@app.get("/main.js")
async def main_js():
    return FileResponse(WEB_DIR / "main.js")


@app.get("/samples.json")
async def samples_json():
    p = WEB_DIR / "samples.json"
    if p.exists(): return FileResponse(p)
    return {"error": "no samples.json yet"}


@app.get("/info")
async def info():
    """Return model + drafter metadata for the UI."""
    if v5e0 is None: return {"error": "model not loaded"}
    tc = getattr(v5e0.V.config, "text_config", v5e0.V.config)
    D = getattr(tc, "hidden_size", None)
    vocab = getattr(tc, "vocab_size", None)
    # Drafter params
    cont_p = sum(p.numel() for p in v5e0.cont.parameters())
    cont2_p = sum(p.numel() for p in v5e0.cont2.parameters())
    return {
        "model_path": getattr(v5e0.V.config, "_name_or_path", v5e0.vclass),
        "vclass": v5e0.vclass,
        "hidden_dim": D,
        "vocab_size": vocab,
        "drafter": {
            "cont_params_M": cont_p / 1e6,
            "cont2_params_M": cont2_p / 1e6,
            "total_params_M": (cont_p + cont2_p) / 1e6,
        },
        "tree": {
            "M": v5e0.tree.M,
            "K": v5e0.tree.K,
            "n_input_tokens": v5e0.tree.n_input,
        },
        "alpha": float(v5e0.cont.alpha.item()),
        "beta": float(v5e0.cont2.beta.item()),
    }


# ============================================================
# Live inference WebSocket
# ============================================================
@app.websocket("/run")
async def ws_run(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_json()
        image_b64 = msg.get("image_b64")
        prompt = msg.get("prompt", "Describe.")
        max_tokens = int(msg.get("max_tokens", 64))
        if not image_b64:
            await ws.send_json({"kind": "error", "message": "image_b64 missing"})
            return
        image_path = save_image_to_tempfile(image_b64)
        inputs = v5e0.prepare(image_path, prompt)
        tokenizer = v5e0.proc.tokenizer

        # ------------------------------------------------------------
        # AR side
        # ------------------------------------------------------------
        await ws.send_json({"side": "ar", "kind": "start", "prompt": prompt})
        from transformers.cache_utils import DynamicCache
        with torch.no_grad():
            out = v5e0.V(**inputs, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
        kv = to_tuple_kv(out.past_key_values)
        last = int(out.logits[0, -1, :].argmax().item())
        ar_ids = [last]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        text = tokenizer.decode([last])
        await ws.send_json({"side": "ar", "kind": "token", "text": text,
                            "token": last, "elapsed_ms": 0.0, "burst": 1})

        with torch.no_grad():
            for _ in range(max_tokens - 1):
                x = torch.tensor([[last]], device="cuda")
                # IMPORTANT: do NOT pass manual position_ids for Qwen2-VL multimodal
                # rope; HF auto-computes from cache_position based on past KV
                if v5e0.is_qwen:
                    out = v5e0.V(input_ids=x, past_key_values=kv_to_cache(kv),
                                 use_cache=True, return_dict=True)
                else:
                    cur = kv[0][0].shape[2]
                    pid = torch.tensor([[cur]], device="cuda")
                    cp = torch.tensor([cur], device="cuda")
                    out = v5e0.V(input_ids=x, past_key_values=kv_to_cache(kv),
                                 position_ids=pid, cache_position=cp,
                                 use_cache=True, return_dict=True)
                last = int(out.logits[0, -1, :].argmax().item())
                kv = to_tuple_kv(out.past_key_values)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                inc = incremental_decode(tokenizer, ar_ids, [last])
                ar_ids.append(last)
                await ws.send_json({"side": "ar", "kind": "token", "text": inc,
                                    "token": last, "elapsed_ms": elapsed_ms, "burst": 1})

        ar_total_ms = (time.perf_counter() - t0) * 1000
        await ws.send_json({"side": "ar", "kind": "done", "total_ms": ar_total_ms,
                            "n_tokens": len(ar_ids)})

        # ------------------------------------------------------------
        # SSD side
        # ------------------------------------------------------------
        await ws.send_json({"side": "ssd", "kind": "start"})
        with torch.no_grad():
            kv2, h_t, ll, prompt_len = v5e0.prefill(inputs)
        ssd_ids = []
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = 0
        with torch.no_grad():
            while len(ssd_ids) < max_tokens:
                new_toks, kv2, h_t, ll, prompt_len = v5e0.step(
                    kv2, h_t, ll, prompt_len, is_first=(r == 0))
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                burst_size = len(new_toks)
                # Emit incremental text per burst (preserves spaces)
                inc = incremental_decode(tokenizer, ssd_ids, new_toks)
                # Emit as a single burst event with the cumulative incremental
                await ws.send_json({"side": "ssd", "kind": "burst",
                                    "text": inc, "tokens": new_toks,
                                    "elapsed_ms": elapsed_ms, "burst": burst_size})
                ssd_ids.extend(new_toks)
                if len(ssd_ids) >= max_tokens: break
                r += 1

        ssd_total_ms = (time.perf_counter() - t0) * 1000
        sp = (ar_total_ms / ssd_total_ms) if ssd_total_ms > 0 else 0
        await ws.send_json({"side": "ssd", "kind": "done", "total_ms": ssd_total_ms,
                            "n_tokens": len(ssd_ids), "speedup": sp})

    except WebSocketDisconnect:
        return
    except Exception as e:
        import traceback; traceback.print_exc()
        try:
            await ws.send_json({"kind": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu", default="0")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    v5e0 = load_v5e0(args.model_path, args.ckpt)
    print(f"[live demo server] http://{args.host}:{args.port}/")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
