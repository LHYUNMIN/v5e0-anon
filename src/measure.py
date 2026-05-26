#!/usr/bin/env python3
"""Measurement-only script: load drafter ckpt + measure sp at arbitrary (M, K, max_tokens).

For T2.1 (longer context) and T2.2 ((M, K) sweep on remaining VLMs).

Usage:
  python measure.py --vlm qwen2-2b --ckpt ./checkpoints/v5e0_qwen2-2b.pt \
      --m 5 --k 3 --max_tokens 128 --n_test 50 --gpu 0
"""
import os, sys, json, gc, time, random, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (
    VLM_CONFIGS, V5e0_Cont, V5e0_Cont2,
    to_tuple_kv, kv_to_cache, load_llava_prompts,
    prepare_qwen_inputs, prepare_llava_inputs,
    build_tree_d3, build_attn_mask_and_pos, prune_kv, measure_d3,
)


def main(vlm_name, ckpt_path, M, K, max_tokens, n_test):
    cfg = VLM_CONFIGS[vlm_name]
    print(f"[MEASURE: {vlm_name}, M={M}, K={K}, max_tokens={max_tokens}, n_test={n_test}]\n",
          flush=True)

    sys.path.insert(0, "./vendor")
    from runtime_env import strip_user_site_packages; strip_user_site_packages()

    if cfg["model_class"] == "qwen":
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa").to("cuda").eval()
        processor = AutoProcessor.from_pretrained(cfg["model_id"], trust_remote_code=True)
        prepare_inputs = lambda s: prepare_qwen_inputs(processor, s["image"], s["prompt"], "cuda")
        is_qwen = True
    elif cfg["model_class"] == "llava":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        processor = AutoProcessor.from_pretrained(cfg["model_id"])
        prepare_inputs = lambda s: prepare_llava_inputs(processor, s["image"], s["prompt"], "cuda")
        is_qwen = False
    else:
        from transformers import LlavaNextForConditionalGeneration, AutoProcessor
        model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        processor = AutoProcessor.from_pretrained(cfg["model_id"])
        prepare_inputs = lambda s: prepare_llava_inputs(processor, s["image"], s["prompt"], "cuda")
        is_qwen = False

    tc = getattr(model.config, "text_config", model.config)
    vocab = tc.vocab_size; hidden_dim = tc.hidden_size
    emb = model.get_input_embeddings()
    if emb.weight.shape[0] > vocab:
        new = nn.Embedding(vocab, emb.weight.shape[1], device="cuda", dtype=emb.weight.dtype)
        new.weight.data.copy_(emb.weight.data[:vocab]); model.set_input_embeddings(new); emb = new
    lm_head = model.get_output_embeddings()
    for p in model.parameters(): p.requires_grad_(False)
    root_emb_table = emb.weight.detach().to(torch.float32)
    print(f"  hidden_dim={hidden_dim}\n", flush=True)

    # Load drafter ckpt (Cont + Cont2)
    state = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    n1_cont = V5e0_Cont(dim=hidden_dim, alpha_init=state.get("alpha", 30.0)).to("cuda")
    n1_cont.Q_proj.weight.data.copy_(state["W_Q1"].float())
    n1_cont.Q_proj.bias.data.copy_(state["W_Q1_bias"].float())
    n1_cont.eval()
    for p in n1_cont.parameters(): p.requires_grad_(False)

    n1_cont2 = V5e0_Cont2(dim=hidden_dim, alpha_init=state.get("alpha", 30.0),
                           beta_init=state.get("beta", 30.0)).to("cuda")
    n1_cont2.Q_proj.weight.data.copy_(state["W_Q2"].float())
    n1_cont2.Q_proj.bias.data.copy_(state["W_Q2_bias"].float())
    n1_cont2.eval()
    for p in n1_cont2.parameters(): p.requires_grad_(False)

    test_samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        n_test, seed=999)
    print(f"[walltime test M={M}, K={K}, tree={1+M+M*K} tokens, {n_test} prompts × {max_tokens}]")
    ar_list, ssd_list, n_d2, n_d3, n_rounds, n_tokens = measure_d3(
        model, n1_cont, n1_cont2, lm_head, root_emb_table, prepare_inputs,
        is_qwen, test_samples, M, K, max_tokens)
    sp = [s/a for a, s in zip(ar_list, ssd_list)]
    ar_a = np.array(ar_list); ssd_a = np.array(ssd_list); sp_a = np.array(sp)
    d2_rate = n_d2 / n_rounds if n_rounds > 0 else 0
    d3_rate = n_d3 / n_rounds if n_rounds > 0 else 0
    tpi = n_tokens / n_rounds if n_rounds > 0 else 0

    print(f"\n{'='*70}")
    print(f"{vlm_name} measure M={M}, K={K}, max_tokens={max_tokens}")
    print(f"{'='*70}")
    print(f"  AR:    {ar_a.mean():.2f} ± {ar_a.std():.2f} t/s")
    print(f"  SSD:   {ssd_a.mean():.2f} ± {ssd_a.std():.2f} t/s")
    print(f"  sp:    {sp_a.mean():.3f} ± {sp_a.std():.3f}")
    print(f"        median {np.median(sp_a):.3f}, range [{sp_a.min():.2f}, {sp_a.max():.2f}]")
    print(f"  TPI {tpi:.2f}, d2 {d2_rate:.3f}, d3 {d3_rate:.3f}")

    out = {
        "vlm": vlm_name, "M": M, "K": K, "max_tokens": max_tokens, "n_test": n_test,
        "ar_tps":  {"mean": float(ar_a.mean()),  "std": float(ar_a.std())},
        "ssd_tps": {"mean": float(ssd_a.mean()), "std": float(ssd_a.std())},
        "sp": {"mean": float(sp_a.mean()), "std": float(sp_a.std()),
               "median": float(np.median(sp_a)),
               "min": float(sp_a.min()), "max": float(sp_a.max())},
        "tpi": tpi, "d2_rate": d2_rate, "d3_rate": d3_rate,
        "n_prompts": len(ar_list),
        "raw": {"ar": ar_list, "ssd": ssd_list, "sp": sp},
    }
    save_path = f"./results/measure_{vlm_name}_M{M}_K{K}_t{max_tokens}.json"
    with open(save_path, "w") as f: json.dump(out, f, indent=2)
    print(f"\n[saved {save_path}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", required=True, choices=list(VLM_CONFIGS.keys()))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--n_test", type=int, default=50)
    p.add_argument("--gpu", default="0")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args.vlm, args.ckpt, args.m, args.k, args.max_tokens, args.n_test)
