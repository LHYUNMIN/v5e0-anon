#!/usr/bin/env python3
"""Record AR baseline vs V5e-0 SSD inference with per-token timestamps.

Output: JSON file consumable by demo/web/main.js for side-by-side playback.

Usage:
  python record_inference.py \
      --model_path Qwen/Qwen2-VL-2B-Instruct \
      --ckpt ./checkpoints/v5e0_qwen2-2b.pt \
      --image image.jpg --prompt "Describe..." \
      --output demo/web/sample_01.json
"""
import os, sys, json, time, gc, argparse, base64
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from infer import load_v5e0, SSDTree, to_tuple_kv, kv_to_cache


@torch.no_grad()
def record_ar(v5e0, inputs, max_tokens):
    """Run greedy AR, record per-token (text, elapsed_ms) tuples."""
    from transformers.cache_utils import DynamicCache
    out = v5e0.V(**inputs, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
    kv = to_tuple_kv(out.past_key_values)
    last = int(out.logits[0, -1, :].argmax().item())

    tokens = []
    tok_text = v5e0.proc.tokenizer.decode([last])
    torch.cuda.synchronize(); t_start = time.perf_counter()
    tokens.append({"token": last, "text": tok_text, "elapsed_ms": 0.0, "burst": 1})

    for _ in range(max_tokens - 1):
        x = torch.tensor([[last]], device="cuda")
        cur = kv[0][0].shape[2]
        pid = torch.tensor([[cur]], device="cuda")
        if v5e0.is_qwen: pid = pid.unsqueeze(0).expand(3, 1, -1)
        cp = torch.tensor([cur], device="cuda")
        out = v5e0.V(input_ids=x, past_key_values=kv_to_cache(kv),
                     position_ids=pid, cache_position=cp,
                     use_cache=True, return_dict=True)
        last = int(out.logits[0, -1, :].argmax().item())
        kv = to_tuple_kv(out.past_key_values)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        tokens.append({"token": last, "text": v5e0.proc.tokenizer.decode([last]),
                       "elapsed_ms": elapsed_ms, "burst": 1})
    return tokens


@torch.no_grad()
def record_ssd(v5e0, inputs, max_tokens):
    """Run V5e-0 SSD, record per-burst (tokens, elapsed_ms) — burst = tokens accepted in one round."""
    kv, h_t, ll, prompt_len = v5e0.prefill(inputs)
    tokens = []
    torch.cuda.synchronize(); t_start = time.perf_counter()
    r = 0
    while len(tokens) < max_tokens:
        new_toks, kv, h_t, ll, prompt_len = v5e0.step(kv, h_t, ll, prompt_len, is_first=(r == 0))
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        burst_size = len(new_toks)
        for i, t in enumerate(new_toks):
            tokens.append({
                "token": t,
                "text": v5e0.proc.tokenizer.decode([t]),
                "elapsed_ms": elapsed_ms,        # all tokens in burst appear "together"
                "burst": burst_size,
                "burst_idx": i,
            })
        r += 1
    return tokens[:max_tokens]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="Describe this image.")
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--output", required=True, help="path to write JSON")
    p.add_argument("--gpu", default="0")
    p.add_argument("--include_image_base64", action="store_true",
                   help="embed image as base64 in output JSON (for static deployment)")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"[loading {args.model_path} + ckpt {args.ckpt}]", flush=True)
    v5e0 = load_v5e0(args.model_path, args.ckpt)
    inputs = v5e0.prepare(args.image, args.prompt)

    # Warmup
    print("[warmup]", flush=True)
    _ = record_ar(v5e0, inputs, 8)
    _ = record_ssd(v5e0, inputs, 8)

    print("[record AR]", flush=True)
    ar_tokens = record_ar(v5e0, inputs, args.max_tokens)
    ar_total_ms = ar_tokens[-1]["elapsed_ms"]
    ar_tps = (len(ar_tokens) - 1) / (ar_total_ms / 1000) if ar_total_ms > 0 else 0

    print("[record V5e-0 SSD]", flush=True)
    ssd_tokens = record_ssd(v5e0, inputs, args.max_tokens)
    ssd_total_ms = ssd_tokens[-1]["elapsed_ms"]
    ssd_tps = len(ssd_tokens) / (ssd_total_ms / 1000) if ssd_total_ms > 0 else 0

    sp = ssd_tps / ar_tps if ar_tps > 0 else 0
    print(f"\n  AR:    {ar_tps:.2f} tok/s ({ar_total_ms:.0f} ms total)")
    print(f"  V5e-0: {ssd_tps:.2f} tok/s ({ssd_total_ms:.0f} ms total)")
    print(f"  sp:    {sp:.3f}×")

    # Optionally embed image as base64 for fully-static GitHub Pages demo
    image_data = None
    if args.include_image_base64:
        with open(args.image, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("ascii")

    out = {
        "model": args.model_path,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "image_path": args.image if not args.include_image_base64 else None,
        "image_base64": image_data,
        "image_format": os.path.splitext(args.image)[1].lstrip(".") if image_data else None,
        "ar": {
            "tokens": ar_tokens,
            "total_ms": ar_total_ms,
            "tps": ar_tps,
            "n_tokens": len(ar_tokens),
        },
        "ssd": {
            "tokens": ssd_tokens,
            "total_ms": ssd_total_ms,
            "tps": ssd_tps,
            "n_tokens": len(ssd_tokens),
            "n_rounds": len(set(t["elapsed_ms"] for t in ssd_tokens)),   # unique timestamps = rounds
        },
        "speedup": sp,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f: json.dump(out, f, indent=2)
    print(f"\n[saved {args.output}]")


if __name__ == "__main__":
    main()
