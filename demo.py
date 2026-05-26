#!/usr/bin/env python3
"""V5e-0 local CLI demo — accurate AR vs V5e-0 wall-clock comparison.

Designed for reviewers who want to reproduce the paper's headline sp number on
their own GPU (no network overhead, no server contention). Includes warm-up,
N-run averaging, and per-VLM expected-sp comparison.

Usage:
    # Default: Qwen2-VL-2B, one COCO image, 64 tokens, 5 runs
    python demo.py --image path/to/image.jpg --prompt "Describe this image."

    # Specific model + more runs for tighter estimate
    python demo.py --model llava-1.5-7b --runs 10 \\
                   --image path/to/image.jpg --prompt "Describe."

    # All 5 models on the same prompt
    python demo.py --all --image path/to/image.jpg --prompt "Describe."

Available models (use --model <key>):
    qwen2-2b              Qwen2-VL-2B          (paper sp 1.91×)
    qwen3-vl-4b           Qwen3-VL-4B          (paper sp 1.58×)
    llava-1.5-7b          LLaVA-1.5-7B         (paper sp 1.87×)
    llava-1.6-mistral-7b  LLaVA-1.6-Mistral-7B (paper sp 1.25×)
    internvl3.5-8b        InternVL3.5-8B       (paper sp 1.70×)
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "vendor"))
from infer import (
    V5e0, V5e0_Cont, V5e0_Cont2,
    to_tuple_kv, kv_to_cache, load_v5e0,
    generate_ar, generate_ssd,
)


MODELS = {
    "qwen2-2b": {
        "label":      "Qwen2-VL-2B",
        "loader":     ("basic", "Qwen/Qwen2-VL-2B-Instruct"),
        "ckpt":       "./checkpoints/v5e0_qwen2-2b.pt",
        "paper_sp":   1.91,
    },
    "qwen3-vl-4b": {
        "label":      "Qwen3-VL-4B",
        "loader":     ("new", "qwen3-vl-4b"),
        "ckpt":       "./checkpoints/v5e0_qwen3vl_4b.pt",
        "paper_sp":   1.58,
    },
    "llava-1.5-7b": {
        "label":      "LLaVA-1.5-7B",
        "loader":     ("basic", "llava-hf/llava-1.5-7b-hf"),
        "ckpt":       "./checkpoints/v5e0_llava-1.5-7b.pt",
        "paper_sp":   1.87,
    },
    "llava-1.6-mistral-7b": {
        "label":      "LLaVA-1.6-Mistral-7B",
        "loader":     ("basic", "llava-hf/llava-v1.6-mistral-7b-hf"),
        "ckpt":       "./checkpoints/v5e0_llava-1.6-mistral-7b.pt",
        "paper_sp":   1.25,
    },
    "internvl3.5-8b": {
        "label":      "InternVL3.5-8B",
        "loader":     ("new", "internvl3.5-8b"),
        "ckpt":       "./checkpoints/v5e0_internvl3_5_8b.pt",
        "paper_sp":   1.70,
    },
}


def _load_drafter(ckpt_path: str, hidden_dim: int):
    state = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    ckpt_dim = state.get("hidden_dim", state["W_Q1"].shape[0])
    if ckpt_dim != hidden_dim:
        raise ValueError(f"drafter ckpt dim {ckpt_dim} != verifier dim {hidden_dim}")
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
    return cont, cont2


def load_model(model_key: str) -> V5e0:
    """Load a V5e0 instance for any model in MODELS (basic or new loader)."""
    cfg = MODELS[model_key]
    loader_kind, loader_arg = cfg["loader"]
    if loader_kind == "basic":
        return load_v5e0(loader_arg, cfg["ckpt"])
    # New-VLM path: setup_new_vlm + manual V5e0 wiring
    from run_new_vlms import setup_new_vlm
    model, processor, _prep, is_qwen, vclass = setup_new_vlm(loader_arg)
    # Robust hidden_size discovery for heterogeneous configs.
    cfg_obj = model.config
    D = None
    for path in [("text_config","hidden_size"), ("llm_config","hidden_size"), ("hidden_size",)]:
        obj = cfg_obj
        try:
            for a in path: obj = getattr(obj, a)
            if isinstance(obj, int): D = obj; break
        except Exception: continue
    if D is None and hasattr(model, "language_model"):
        D = getattr(model.language_model.config, "hidden_size", None)
    if D is None:
        raise RuntimeError(f"could not determine hidden_size for {model_key}")
    cont, cont2 = _load_drafter(cfg["ckpt"], D)
    v = V5e0(model, processor, cont, cont2, vclass=vclass, is_qwen=is_qwen)
    def _new_prepare(image_path, prompt_text):
        return _prep({"image": image_path, "prompt": prompt_text})
    v.prepare = _new_prepare
    # Patch lm_head fallback for custom-modeling VLMs
    if v.lm_head is None:
        for attr_path in [("language_model","lm_head"), ("transformer","output_layer"), ("output_layer",), ("lm_head",)]:
            obj = model
            try:
                for a in attr_path: obj = getattr(obj, a)
                if callable(obj): v.lm_head = obj; break
            except AttributeError: continue
    return v


def measure(v: V5e0, image: str, prompt: str, max_tokens: int, runs: int):
    """Warm up once, then time AR and SSD `runs` times each. Returns dict."""
    inputs = v.prepare(image, prompt)

    # ---- Warm-up (compile/cache, not included in timings) ----
    print(f"  warming up...", flush=True)
    _ = generate_ar(v.V, v.proc, inputs, v.is_qwen, max_tokens=min(8, max_tokens))
    _ = generate_ssd(v, inputs, max_tokens=min(8, max_tokens))
    torch.cuda.empty_cache(); gc.collect()

    # ---- AR runs ----
    ar_tps_list, ar_text = [], None
    for r in range(runs):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        tokens, elapsed = generate_ar(v.V, v.proc, inputs, v.is_qwen, max_tokens)
        ar_tps_list.append(max_tokens / elapsed)
        if r == 0: ar_text = v.proc.tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"    AR  run {r+1}/{runs}: {1000*elapsed:.0f} ms ({max_tokens/elapsed:.1f} tok/s)", flush=True)

    # ---- SSD runs ----
    ssd_tps_list, ssd_text, ssd_round_stats = [], None, []
    for r in range(runs):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        tokens, elapsed, n_per_round = generate_ssd(v, inputs, max_tokens)
        ssd_tps_list.append(len(tokens) / elapsed)
        ssd_round_stats.append(n_per_round)
        if r == 0: ssd_text = v.proc.tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"    V5e0 run {r+1}/{runs}: {1000*elapsed:.0f} ms ({len(tokens)/elapsed:.1f} tok/s, "
              f"{len(n_per_round)} rounds, TPI {len(tokens)/max(1,len(n_per_round)):.2f})", flush=True)

    # ---- Stats ----
    sp_list = [s/a for a, s in zip(ar_tps_list, ssd_tps_list)]
    rounds_avg = mean(len(r) for r in ssd_round_stats)
    tpi_avg = max_tokens / rounds_avg
    return {
        "ar":  {"tps_mean": mean(ar_tps_list),  "tps_median": median(ar_tps_list),
                 "tps_std": (stdev(ar_tps_list) if len(ar_tps_list)>1 else 0),
                 "text": ar_text},
        "ssd": {"tps_mean": mean(ssd_tps_list), "tps_median": median(ssd_tps_list),
                 "tps_std": (stdev(ssd_tps_list) if len(ssd_tps_list)>1 else 0),
                 "text": ssd_text,
                 "tpi":   tpi_avg, "rounds_avg": rounds_avg},
        "sp":  {"mean": mean(sp_list), "median": median(sp_list),
                 "min":  min(sp_list), "max":    max(sp_list),
                 "std":  (stdev(sp_list) if len(sp_list)>1 else 0)},
        "text_match": ar_text == ssd_text,
    }


def report(model_key: str, m: dict, max_tokens: int):
    cfg = MODELS[model_key]
    print(f"\n========== {cfg['label']}  (paper {cfg['paper_sp']}×) ==========")
    print(f"  AR  : {m['ar']['tps_mean']:>6.2f} ± {m['ar']['tps_std']:.2f} tok/s "
          f"(median {m['ar']['tps_median']:.2f})")
    print(f"  V5e0: {m['ssd']['tps_mean']:>6.2f} ± {m['ssd']['tps_std']:.2f} tok/s "
          f"(median {m['ssd']['tps_median']:.2f}, TPI {m['ssd']['tpi']:.2f}, "
          f"~{m['ssd']['rounds_avg']:.1f} rounds/{max_tokens} tok)")
    print(f"  sp  : {m['sp']['mean']:>6.3f}× ± {m['sp']['std']:.3f}  "
          f"[range {m['sp']['min']:.3f}–{m['sp']['max']:.3f}, median {m['sp']['median']:.3f}]")
    pct = m['sp']['mean'] / cfg['paper_sp'] * 100
    flag = "  ≈ paper" if pct >= 90 else ("  ~paper" if pct >= 75 else "  ⚠ below paper")
    print(f"  vs paper: {pct:.0f}% of {cfg['paper_sp']}×{flag}")
    print(f"  text match: {m['text_match']}")
    print(f"  AR  text : {m['ar']['text'][:120]!r}")
    print(f"  V5e0 text: {m['ssd']['text'][:120]!r}")


def race_visual(v: V5e0, image: str, prompt: str, max_tokens: int):
    """Single-shot visualization: stream AR then V5e-0 with colored output, then
    print the wall-clock comparison. Designed for terminal recording (asciinema).
    """
    from transformers.cache_utils import DynamicCache as _DC
    inputs = v.prepare(image, prompt)
    tokenizer = v.proc.tokenizer if hasattr(v.proc, "tokenizer") else v.proc

    BOLD, DIM, GREEN, BLUE, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[34m", "\033[0m"

    print(f"\n{BOLD}=== AR vs V5e-0 on {v.vclass} ==={RESET}")
    print(f"{DIM}prompt: {prompt}{RESET}\n")

    # --- AR side ---
    print(f"{BOLD}AR (greedy autoregressive):{RESET}")
    with torch.no_grad():
        out = v.V(**inputs, past_key_values=_DC(), use_cache=True,
                  output_hidden_states=True, return_dict=True)
    rd = getattr(out, "rope_deltas", None)
    rope_delta_val = int(rd.flatten()[0].item()) if v.is_qwen and rd is not None else 0
    kv = to_tuple_kv(out.past_key_values)
    last = int(out.logits[0, -1, :].argmax().item())
    ar_ids = [last]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    print(tokenizer.decode([last], skip_special_tokens=True), end="", flush=True)
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            x = torch.tensor([[last]], device="cuda")
            cur = kv[0][0].shape[2]
            pid_val = cur + rope_delta_val
            pid = torch.tensor([[pid_val]], device="cuda")
            if v.is_qwen: pid = pid.unsqueeze(0).expand(3, 1, -1)
            cp = torch.tensor([cur], device="cuda")
            out = v.V(input_ids=x, past_key_values=kv_to_cache(kv),
                      position_ids=pid, cache_position=cp,
                      use_cache=True, return_dict=True)
            last = int(out.logits[0, -1, :].argmax().item())
            kv = to_tuple_kv(out.past_key_values)
            prev = tokenizer.decode(ar_ids, skip_special_tokens=True)
            ar_ids.append(last)
            cur_full = tokenizer.decode(ar_ids, skip_special_tokens=True)
            print(cur_full[len(prev):], end="", flush=True)
    torch.cuda.synchronize()
    ar_ms = (time.perf_counter() - t0) * 1000
    ar_tps = max_tokens * 1000 / ar_ms
    print(f"\n{DIM}{ar_ms:.0f} ms / {max_tokens} tokens = {ar_tps:.1f} tok/s{RESET}\n")

    # --- V5e-0 side ---
    print(f"{BOLD}V5e-0 (ours):{RESET}  {GREEN}green spans = burst accepted in single verifier round{RESET}")
    kv2, h_t, ll, prompt_len = v.prefill(inputs)
    ssd_ids = []
    rounds = 0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        while len(ssd_ids) < max_tokens:
            new_toks, kv2, h_t, ll, prompt_len = v.step(
                kv2, h_t, ll, prompt_len, is_first=(rounds == 0))
            torch.cuda.synchronize()
            prev = tokenizer.decode(ssd_ids, skip_special_tokens=True)
            ssd_ids.extend(new_toks)
            cur_full = tokenizer.decode(ssd_ids, skip_special_tokens=True)
            inc = cur_full[len(prev):]
            # Highlight bursts of 2+ tokens (green); single-token rounds in normal color
            if len(new_toks) >= 2:
                print(f"{GREEN}{inc}{RESET}", end="", flush=True)
            else:
                print(inc, end="", flush=True)
            if len(ssd_ids) >= max_tokens: break
            rounds += 1
    torch.cuda.synchronize()
    ssd_ms = (time.perf_counter() - t0) * 1000
    ssd_tps = len(ssd_ids) * 1000 / ssd_ms
    tpi = len(ssd_ids) / max(rounds, 1)
    print(f"\n{DIM}{ssd_ms:.0f} ms / {len(ssd_ids)} tokens = {ssd_tps:.1f} tok/s  "
          f"({rounds} rounds, TPI {tpi:.2f}){RESET}\n")

    # --- Comparison ---
    sp = ar_ms / ssd_ms
    print(f"{BOLD}{BLUE}wall-clock speedup: {sp:.2f}× faster{RESET}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--model", default="qwen2-2b", choices=list(MODELS),
                   help="Which verifier VLM to load.")
    p.add_argument("--all", action="store_true",
                   help="Run all 5 models (warning: ~30 GB of model downloads first time).")
    p.add_argument("--image", default=None,
                   help="Path to an input image. If omitted, uses a built-in COCO sample.")
    p.add_argument("--prompt", default="Describe this image in detail.",
                   help="Text prompt to pair with the image.")
    p.add_argument("--max_tokens", type=int, default=64,
                   help="Number of tokens to generate per run.")
    p.add_argument("--runs", type=int, default=5,
                   help="Number of timing runs per side (AR and V5e-0).")
    p.add_argument("--race", action="store_true",
                   help="Single-shot streamed AR-then-V5e-0 visualization "
                        "(designed for terminal recording). Skips --runs averaging.")
    p.add_argument("--gpu", default="0", help="CUDA device index.")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    image = args.image or "demo/sample.jpg"
    if not Path(image).exists():
        sys.exit(f"image not found: {image}  (pass --image PATH)")

    keys = list(MODELS) if args.all else [args.model]
    print(f"[demo.py] Running {len(keys)} model(s) on {image} with prompt {args.prompt!r}, "
          f"max_tokens={args.max_tokens}, runs={args.runs}\n")

    all_results = {}
    for key in keys:
        print(f"\n>>> Loading {key} (this may take 10-60s on first run)...", flush=True)
        try:
            v = load_model(key)
            if args.race:
                # Warm up first so timing reflects steady-state behavior.
                _ = generate_ar(v.V, v.proc, v.prepare(image, args.prompt),
                                v.is_qwen, max_tokens=min(8, args.max_tokens))
                _ = generate_ssd(v, v.prepare(image, args.prompt),
                                 max_tokens=min(8, args.max_tokens))
                torch.cuda.empty_cache(); gc.collect()
                race_visual(v, image, args.prompt, args.max_tokens)
            else:
                m = measure(v, image, args.prompt, args.max_tokens, args.runs)
                report(key, m, args.max_tokens)
                all_results[key] = m
            del v
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {type(e).__name__}: {e}")
        gc.collect(); torch.cuda.empty_cache()

    if len(all_results) > 1:
        print("\n\n========== SUMMARY ==========")
        print(f"{'Model':<22} {'AR t/s':>10} {'V5e0 t/s':>10} {'sp':>10} {'paper':>8} {'% paper':>8}")
        for k, m in all_results.items():
            paper = MODELS[k]['paper_sp']
            pct = m['sp']['mean'] / paper * 100
            print(f"{k:<22} {m['ar']['tps_mean']:>9.1f}  {m['ssd']['tps_mean']:>9.1f}  "
                  f"{m['sp']['mean']:>8.3f}×  {paper:>6.2f}×  {pct:>6.0f}%")


if __name__ == "__main__":
    main()
