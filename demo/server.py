#!/usr/bin/env python3
"""FastAPI live demo server for V5e-0 with 5 selectable verifier VLMs.

WebSocket /run protocol (client → server):
    {"model": "<key>", "prompt": "...", "image_b64": "data:...", "max_tokens": 64}

Server streams (server → client):
    {"kind": "loading", "model": "<key>"}                 - sent if model needs to be loaded
    {"kind": "ready"}                                     - model ready, generation starts
    {"kind": "trace", "ar": {...}, "ssd": {...}, "speedup": ...}
                                                           - both AR and SSD traces ready
                                                             (browser replays at the recorded
                                                              wall-clock timings to show race)
    {"kind": "error", "message": "..."}

Usage:
    python demo/server.py --port 8000 --gpu 0 --preload qwen2-2b
"""
import os, sys, json, time, base64, argparse, gc, asyncio
from pathlib import Path
from io import BytesIO

import torch
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
from infer import V5e0, V5e0_Cont, V5e0_Cont2, to_tuple_kv, kv_to_cache, load_v5e0


# ============================================================
# Model registry — 5 representative VLMs spanning 4 families
# ============================================================
# "loader" is either:
#   ("basic", model_path)   → uses infer.load_v5e0 (Qwen-VL / LLaVA / LLaVA-Next)
#   ("new", vlm_key)        → uses run_new_vlms.setup_new_vlm (GLM-4V, Qwen3-VL,
#                              InternVL3.5, ...) with manual V5e0 wiring.
MODEL_REGISTRY = {
    "qwen2-2b": {
        "label": "Qwen2-VL-2B (fastest)",
        "loader": ("basic", "Qwen/Qwen2-VL-2B-Instruct"),
        "ckpt":   "./checkpoints/v5e0_qwen2-2b.pt",
        "paper_sp": 1.91,
    },
    "qwen3-vl-4b": {
        "label": "Qwen3-VL-4B",
        "loader": ("new", "qwen3-vl-4b"),
        "ckpt":   "./checkpoints/v5e0_qwen3vl_4b.pt",
        "paper_sp": 1.58,
    },
    "llava-1.5-7b": {
        "label": "LLaVA-1.5-7B",
        "loader": ("basic", "llava-hf/llava-1.5-7b-hf"),
        "ckpt":   "./checkpoints/v5e0_llava-1.5-7b.pt",
        "paper_sp": 1.87,
    },
    "llava-1.6-mistral-7b": {
        "label": "LLaVA-1.6-Mistral-7B",
        "loader": ("basic", "llava-hf/llava-v1.6-mistral-7b-hf"),
        "ckpt":   "./checkpoints/v5e0_llava-1.6-mistral-7b.pt",
        "paper_sp": 1.25,
    },
    "internvl3.5-8b": {
        "label": "InternVL3.5-8B",
        "loader": ("new", "internvl3.5-8b"),
        "ckpt":   "./checkpoints/v5e0_internvl3_5_8b.pt",
        "paper_sp": 1.70,
    },
}

MAX_LOADED = 1
MODEL_CACHE = {}
LOAD_ORDER = []
app = FastAPI()


def _load_drafter(ckpt_path, hidden_dim):
    """Load V5e-0 Cont/Cont2 heads from a drafter checkpoint."""
    state = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    ckpt_dim = state.get("hidden_dim", state["W_Q1"].shape[0])
    if ckpt_dim != hidden_dim:
        raise ValueError(f"drafter ckpt dim {ckpt_dim} != verifier hidden_dim {hidden_dim}")
    alpha = float(state.get("alpha", 30.0))
    beta  = float(state.get("beta",  30.0))
    cont  = V5e0_Cont( dim=hidden_dim, alpha=alpha).to("cuda")
    cont2 = V5e0_Cont2(dim=hidden_dim, alpha=alpha, beta=beta).to("cuda")
    cont.W_Q.weight.data.copy_(state["W_Q1"].float())
    cont.W_Q.bias.data.copy_(state["W_Q1_bias"].float())
    cont2.W_Q.weight.data.copy_(state["W_Q2"].float())
    cont2.W_Q.bias.data.copy_(state["W_Q2_bias"].float())
    cont.eval(); cont2.eval()
    for p in cont.parameters():  p.requires_grad_(False)
    for p in cont2.parameters(): p.requires_grad_(False)
    print(f"[loaded drafter: {ckpt_path} (dim={hidden_dim}, alpha={alpha}, beta={beta})]",
          flush=True)
    return cont, cont2


def load_v5e0_unified(model_key: str) -> V5e0:
    """Load a V5e0 instance for any model in MODEL_REGISTRY (basic or new)."""
    cfg = MODEL_REGISTRY[model_key]
    loader_kind, loader_arg = cfg["loader"]
    if loader_kind == "basic":
        return load_v5e0(loader_arg, cfg["ckpt"])
    # "new" path: setup_new_vlm + manual V5e0 wiring
    from run_new_vlms import setup_new_vlm
    model, processor, _prep, is_qwen, vclass = setup_new_vlm(loader_arg)
    # Determine hidden_size across heterogeneous model configs:
    # - HF standard: model.config.text_config.hidden_size or model.config.hidden_size
    # - InternVL:    model.config.llm_config.hidden_size (Qwen2-style inner LM)
    # - GLM-4V:      model.config.hidden_size
    cfg_obj = model.config
    D = None
    for path in [
        ("text_config", "hidden_size"),
        ("llm_config",  "hidden_size"),
        ("hidden_size",),
    ]:
        obj = cfg_obj
        try:
            for attr in path: obj = getattr(obj, attr)
            if isinstance(obj, int):
                D = obj; break
        except Exception:
            continue
    if D is None and hasattr(model, "language_model"):
        D = getattr(model.language_model.config, "hidden_size", None)
    if D is None:
        raise RuntimeError(f"could not determine hidden_size for {model_key}")
    print(f"[unified loader: {model_key} hidden_size={D}]", flush=True)
    cont, cont2 = _load_drafter(cfg["ckpt"], D)
    # The V5e0 class's `prepare` method dispatches by vclass; for new VLMs we
    # override prepare with the per-VLM prep function from setup_new_vlm.
    v = V5e0(model, processor, cont, cont2, vclass=vclass, is_qwen=is_qwen)

    def _new_prepare(image_path, prompt_text):
        return _prep({"image": image_path, "prompt": prompt_text})
    v.prepare = _new_prepare

    # Some custom-modeling VLMs (GLM-4V, InternVL3.5) return None from
    # get_output_embeddings(); locate the actual lm_head manually.
    if v.lm_head is None:
        for attr_path in [
            ("language_model", "lm_head"),
            ("transformer", "output_layer"),
            ("output_layer",),
            ("lm_head",),
        ]:
            obj = model
            try:
                for a in attr_path: obj = getattr(obj, a)
                if callable(obj):
                    v.lm_head = obj
                    print(f"[unified loader: patched lm_head via {'.'.join(attr_path)}]", flush=True)
                    break
            except AttributeError:
                continue
    if v.lm_head is None:
        raise RuntimeError(f"could not locate lm_head for {model_key}")
    # Same for input embeddings (for v.E embedding table)
    if v.E is None or (hasattr(v.E, 'numel') and v.E.numel() == 0):
        emb = model.get_input_embeddings()
        if emb is None:
            for attr_path in [("language_model","embed_tokens"), ("transformer","embedding","word_embeddings"),]:
                obj = model
                try:
                    for a in attr_path: obj = getattr(obj, a)
                    emb = obj; break
                except AttributeError:
                    continue
        if emb is not None and hasattr(emb, 'weight'):
            v.E = emb.weight.detach().to(torch.float32)
    return v


def get_model(model_key: str):
    """Return loaded v5e0 instance for `model_key`, lazily loading + evicting as needed."""
    if model_key in MODEL_CACHE:
        LOAD_ORDER.remove(model_key); LOAD_ORDER.append(model_key)
        return MODEL_CACHE[model_key], False
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model key: {model_key}")
    while len(MODEL_CACHE) >= MAX_LOADED and LOAD_ORDER:
        old = LOAD_ORDER.pop(0)
        print(f"[evict {old}]", flush=True)
        try:
            inst = MODEL_CACHE.pop(old); del inst
        finally:
            gc.collect(); torch.cuda.empty_cache()
    print(f"[load {model_key}]", flush=True)
    inst = load_v5e0_unified(model_key)
    MODEL_CACHE[model_key] = inst
    LOAD_ORDER.append(model_key)
    return inst, True


def save_image_to_tempfile(b64_str):
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    img = Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")
    p = Path("/tmp/demo_upload.jpg")
    img.save(p, "JPEG")
    return str(p)


def incremental_decode(tokenizer, prev_ids, new_ids):
    """Decode incrementally, robust to special / out-of-vocab token ids.

    Some VLMs (notably GLM-4V) emit special-token ids that the tokenizer cannot
    decode in `decode(ids)` mode unless skip_special_tokens=True is set. We also
    fall back to per-token decode on error.
    """
    if not new_ids: return ""
    def _safe(ids):
        try:
            return tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            parts = []
            for t in ids:
                try:    parts.append(tokenizer.decode([t], skip_special_tokens=True))
                except Exception: parts.append("")
            return "".join(parts)
    prev_text = _safe(prev_ids) if prev_ids else ""
    new_text  = _safe(prev_ids + new_ids)
    return new_text[len(prev_text):]


# ============================================================
# Static UI serving (no-cache so updates show without hard refresh)
# ============================================================
WEB_DIR = Path(__file__).parent / "web"
NO_CACHE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}


@app.get("/")
async def root():        return FileResponse(WEB_DIR / "index.html", headers=NO_CACHE)
@app.get("/style.css")
async def style_css():   return FileResponse(WEB_DIR / "style.css",  headers=NO_CACHE)
@app.get("/main.js")
async def main_js():     return FileResponse(WEB_DIR / "main.js",    headers=NO_CACHE)


@app.get("/models")
async def models():
    """Return the model dropdown: {key: label}."""
    return JSONResponse({k: cfg["label"] for k, cfg in MODEL_REGISTRY.items()})


# ============================================================
# Live inference WebSocket — runs AR and SSD sequentially with
# precise wall-clock measurement, then sends the full trace.
# ============================================================
@app.websocket("/run")
async def ws_run(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_json()
        model_key = msg.get("model", "qwen2-2b")
        prompt = msg.get("prompt", "Describe.")
        image_b64 = msg.get("image_b64")
        max_tokens = int(msg.get("max_tokens", 64))
        if not image_b64:
            await ws.send_json({"kind": "error", "message": "image_b64 missing"})
            return

        if model_key not in MODEL_CACHE:
            await ws.send_json({"kind": "loading", "model": model_key})

        try:
            # Run blocking model load in a thread so asyncio can keep sending
            # WebSocket pings (otherwise long loads trigger keepalive timeout).
            loop = asyncio.get_event_loop()
            v5e0, _ = await loop.run_in_executor(None, get_model, model_key)
        except Exception as e:
            import traceback; traceback.print_exc()
            await ws.send_json({"kind": "error", "message": f"failed to load {model_key}: {e}"})
            return

        await ws.send_json({"kind": "ready"})

        image_path = save_image_to_tempfile(image_b64)
        inputs = v5e0.prepare(image_path, prompt)
        tokenizer = v5e0.proc.tokenizer if hasattr(v5e0.proc, "tokenizer") else v5e0.proc

        # Heavy blocking work — run AR then SSD inside an executor, then return
        # the complete trace. The executor frees the asyncio loop so it can keep
        # sending WS pings; without this, long inference can trigger keepalive
        # timeouts on slower model loads.
        def _run_blocking():
            from transformers.cache_utils import DynamicCache as _DC
            # ----- AR -----
            ar_events = []
            with torch.no_grad():
                out = v5e0.V(**inputs, past_key_values=_DC(), use_cache=True, return_dict=True)
            kv = to_tuple_kv(out.past_key_values)
            # Qwen2-VL / Qwen3-VL return `rope_deltas` from prefill: the offset
            # to apply to cache_position so that MRoPE positions land on the
            # correct (post-image-expansion) text positions. Without this,
            # decode-mode position_ids point past any seen position and the
            # model produces degenerate "TheTheThe..." output.
            rope_delta_val = 0
            rd = getattr(out, "rope_deltas", None)
            if v5e0.is_qwen and rd is not None:
                rope_delta_val = int(rd.flatten()[0].item())
            last = int(out.logits[0, -1, :].argmax().item())
            ar_ids = [last]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ar_events.append({"text": tokenizer.decode([last]), "elapsed_ms": 0.0})
            with torch.no_grad():
                for _ in range(max_tokens - 1):
                    x = torch.tensor([[last]], device="cuda")
                    cur = kv[0][0].shape[2]
                    pid_val = cur + rope_delta_val
                    pid = torch.tensor([[pid_val]], device="cuda")
                    if v5e0.is_qwen:
                        pid = pid.unsqueeze(0).expand(3, 1, -1)
                    cp = torch.tensor([cur], device="cuda")
                    out = v5e0.V(input_ids=x, past_key_values=kv_to_cache(kv),
                                 position_ids=pid, cache_position=cp,
                                 use_cache=True, return_dict=True)
                    last = int(out.logits[0, -1, :].argmax().item())
                    kv = to_tuple_kv(out.past_key_values)
                    torch.cuda.synchronize()
                    inc = incremental_decode(tokenizer, ar_ids, [last])
                    ar_ids.append(last)
                    ar_events.append({"text": inc, "elapsed_ms": (time.perf_counter() - t0) * 1000})
            ar_total_ms = (time.perf_counter() - t0) * 1000

            # ----- SSD -----
            ssd_events = []
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
                    inc = incremental_decode(tokenizer, ssd_ids, new_toks)
                    ssd_events.append({
                        "text": inc, "burst": len(new_toks),
                        "elapsed_ms": (time.perf_counter() - t0) * 1000,
                    })
                    ssd_ids.extend(new_toks)
                    if len(ssd_ids) >= max_tokens: break
                    r += 1
            ssd_total_ms = (time.perf_counter() - t0) * 1000
            return ar_events, ar_total_ms, len(ar_ids), ssd_events, ssd_total_ms, len(ssd_ids)

        loop = asyncio.get_event_loop()
        ar_events, ar_total_ms, n_ar, ssd_events, ssd_total_ms, n_ssd = \
            await loop.run_in_executor(None, _run_blocking)
        sp = (ar_total_ms / ssd_total_ms) if ssd_total_ms > 0 else 0.0
        paper_sp = MODEL_REGISTRY[model_key].get("paper_sp")
        await ws.send_json({
            "kind": "trace",
            "ar":  {"events": ar_events,  "total_ms": ar_total_ms,  "n_tokens": n_ar},
            "ssd": {"events": ssd_events, "total_ms": ssd_total_ms, "n_tokens": n_ssd},
            "speedup": sp,
            "paper_sp": paper_sp,
        })

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
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu", default="0")
    p.add_argument("--preload", default="qwen2-2b",
                   help="Comma-separated model keys to preload at startup")
    p.add_argument("--max_loaded", type=int, default=1,
                   help="Max models resident in GPU memory at once")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    MAX_LOADED = max(1, args.max_loaded)
    for key in (args.preload.split(",") if args.preload else []):
        key = key.strip()
        if key in MODEL_REGISTRY:
            print(f"[preload {key}]", flush=True)
            try:
                get_model(key)
            except Exception as e:
                print(f"[preload FAILED for {key}: {e}]", flush=True)
    print(f"[demo server] http://{args.host}:{args.port}/  models={list(MODEL_REGISTRY)}",
          flush=True)
    import uvicorn
    # Large ping interval so slow first-model loads (~60 s on GPU) don't trip the keepalive.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                ws_ping_interval=120.0, ws_ping_timeout=600.0)
