#!/usr/bin/env python3
"""Anchor-skip impact ablation (Section 6.1 methodology bug evidence).

Run V5e-0 SSD with anchor_skip ON vs OFF on identical prompts.
ON  (correct): rounds k>=1 skip the anchor token (matches AR)
OFF (buggy):   every round outputs anchor first → 50% token inflation

Compares both sp and lossless token match.

Usage:
  python anchor_skip_test.py --vlm qwen2-2b --ckpt ./checkpoints/v5e0_qwen2-2b.pt --gpu 0
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
    build_tree_d3, build_attn_mask_and_pos, prune_kv,
)


@torch.no_grad()
def measure_with_flag(model, n1_cont, n1_cont2, lm_head, root_emb_table, prepare_inputs, is_qwen,
                     test_samples, M, K, max_tokens, anchor_skip):
    """Same as measure_d3 in run.py but anchor_skip flag controls bug fix."""
    from transformers.cache_utils import DynamicCache
    parents, depths = build_tree_d3(M, K)
    n_input = len(parents) - 1

    ar_tps_list = []; ssd_tps_list = []
    ar_outputs_list = []; ssd_outputs_list = []
    n_rounds_total = 0; n_tokens_total = 0

    for s_i, s in enumerate(test_samples):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        try:
            inputs = prepare_inputs(s)
            out = model(**inputs, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
        except Exception: continue
        kv = to_tuple_kv(out.past_key_values)
        last = int(out.logits[0, -1, :].argmax().item())
        ar_tokens = [last]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            for _ in range(max_tokens - 1):
                x = torch.tensor([[last]], device="cuda")
                cur = kv[0][0].shape[2]
                pid = torch.tensor([[cur]], device="cuda")
                if is_qwen: pid = pid.unsqueeze(0).expand(3, 1, -1)
                cp = torch.tensor([cur], device="cuda")
                out = model(input_ids=x, past_key_values=kv_to_cache(kv),
                            position_ids=pid, cache_position=cp, use_cache=True, return_dict=True)
                last = int(out.logits[0, -1, :].argmax().item())
                kv = to_tuple_kv(out.past_key_values)
                ar_tokens.append(last)
        except Exception: continue
        torch.cuda.synchronize()
        ar_tps_list.append(max_tokens / (time.perf_counter() - t0))
        ar_outputs_list.append(ar_tokens)

        # SSD with anchor_skip flag
        try:
            inputs = prepare_inputs(s)
            out = model(**inputs, past_key_values=DynamicCache(),
                        use_cache=True, output_hidden_states=True, return_dict=True)
        except Exception: continue
        kv = to_tuple_kv(out.past_key_values)
        prompt_len = kv[0][0].shape[2]
        h_t = out.hidden_states[-1][0, -1, :].float()
        ll = out.logits[0, -1, :].float()

        gen = []
        n_rounds = 0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            while len(gen) < max_tokens:
                r_star = int(ll.argmax().item())
                root_emb = root_emb_table[r_star].unsqueeze(0)
                z = n1_cont(h_t.unsqueeze(0), root_emb)
                cont_logits = lm_head(z.to(torch.bfloat16)).float()
                cont_topm = cont_logits.topk(M, dim=-1).indices[0]

                h_b = h_t.unsqueeze(0).expand(M, -1)
                r_b = root_emb.expand(M, -1)
                c_embs = root_emb_table[cont_topm]
                z2 = n1_cont2(h_b, r_b, c_embs)
                cont2_logits = lm_head(z2.to(torch.bfloat16)).float()
                cont2_topk = cont2_logits.topk(K, dim=-1).indices

                tree_ids = [r_star] + cont_topm.tolist()
                for i in range(M):
                    tree_ids.extend(cont2_topk[i].tolist())
                xt = torch.tensor([tree_ids], device="cuda")
                mask, pos_ids = build_attn_mask_and_pos(parents, depths, prompt_len,
                                                          "cuda", model.dtype, qwen=is_qwen)
                cp = torch.arange(prompt_len, prompt_len + n_input, device="cuda")
                out = model(input_ids=xt, past_key_values=kv_to_cache(kv),
                            attention_mask=mask, position_ids=pos_ids, cache_position=cp,
                            use_cache=True, output_hidden_states=True, return_dict=True)
                tree_logits = out.logits[0]
                tree_hidden = out.hidden_states[-1][0]
                kv_after = to_tuple_kv(out.past_key_values)

                cont_arg = int(tree_logits[0].argmax().item())
                acc_cont = None; cont_pos = -1
                for j in range(M):
                    pos = 1 + j
                    if tree_ids[pos] == cont_arg:
                        acc_cont = pos; cont_pos = j; break

                # ANCHOR SKIP LOGIC
                is_first = (n_rounds == 0)
                if anchor_skip:
                    base = [r_star] if is_first else []
                else:
                    base = [r_star]   # BUG: always emit anchor

                if acc_cont is not None:
                    cont2_start = 1 + M + cont_pos * K
                    cont2_arg = int(tree_logits[acc_cont].argmax().item())
                    acc_cont2 = None
                    for c2 in range(K):
                        pos = cont2_start + c2
                        if tree_ids[pos] == cont2_arg:
                            acc_cont2 = pos; break
                    if acc_cont2 is not None:
                        bonus = int(tree_logits[acc_cont2].argmax().item())
                        new_toks = base + [cont_arg, cont2_arg, bonus]
                        acc_idx = [0, acc_cont, acc_cont2]
                        new_h = tree_hidden[acc_cont2].float()
                        new_ll = tree_logits[acc_cont2].float()
                    else:
                        new_toks = base + [cont_arg, cont2_arg]
                        acc_idx = [0, acc_cont]
                        new_h = tree_hidden[acc_cont].float()
                        new_ll = tree_logits[acc_cont].float()
                else:
                    bonus = cont_arg
                    new_toks = base + [bonus]
                    acc_idx = [0]
                    new_h = tree_hidden[0].float()
                    new_ll = tree_logits[0].float()

                kv = prune_kv(kv_after, prompt_len, acc_idx)
                prompt_len += len(acc_idx)
                h_t = new_h; ll = new_ll
                gen.extend(new_toks)
                n_rounds += 1
        except Exception as e:
            print(f"  prompt {s_i} SSD failed: {e}", flush=True)
            continue
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ssd_tps_list.append(len(gen) / elapsed)
        ssd_outputs_list.append(gen[:max_tokens])
        n_rounds_total += n_rounds; n_tokens_total += len(gen)

    # Token match
    n_pairs = min(len(ar_outputs_list), len(ssd_outputs_list))
    total_match = 0; total_pos = 0; n_exact = 0
    for ar, ssd in zip(ar_outputs_list[:n_pairs], ssd_outputs_list[:n_pairs]):
        L = min(len(ar), len(ssd))
        m = sum(1 for i in range(L) if ar[i] == ssd[i])
        total_match += m; total_pos += L
        if ar[:L] == ssd[:L]: n_exact += 1

    return {
        "ar_tps": ar_tps_list, "ssd_tps": ssd_tps_list,
        "n_rounds": n_rounds_total, "n_tokens": n_tokens_total,
        "token_match_rate": total_match / total_pos if total_pos else 0,
        "n_exact": n_exact, "n_pairs": n_pairs,
    }


def main(vlm_name, ckpt_path, n_test, max_tokens):
    cfg = VLM_CONFIGS[vlm_name]
    print(f"[ANCHOR-SKIP ABLATION: {vlm_name}, n={n_test}, max_tokens={max_tokens}]\n", flush=True)

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
    vocab = tc.vocab_size; D = tc.hidden_size
    emb = model.get_input_embeddings()
    if emb.weight.shape[0] > vocab:
        new = nn.Embedding(vocab, emb.weight.shape[1], device="cuda", dtype=emb.weight.dtype)
        new.weight.data.copy_(emb.weight.data[:vocab]); model.set_input_embeddings(new); emb = new
    lm_head = model.get_output_embeddings()
    for p in model.parameters(): p.requires_grad_(False)
    root_emb_table = emb.weight.detach().to(torch.float32)

    # Load ckpt
    state = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    n1_cont = V5e0_Cont(dim=D, alpha_init=state.get("alpha", 30.0)).to("cuda")
    n1_cont.Q_proj.weight.data.copy_(state["W_Q1"].float())
    n1_cont.Q_proj.bias.data.copy_(state["W_Q1_bias"].float())
    n1_cont2 = V5e0_Cont2(dim=D, alpha_init=state.get("alpha", 30.0),
                           beta_init=state.get("beta", 30.0)).to("cuda")
    n1_cont2.Q_proj.weight.data.copy_(state["W_Q2"].float())
    n1_cont2.Q_proj.bias.data.copy_(state["W_Q2_bias"].float())
    for h in [n1_cont, n1_cont2]:
        h.eval()
        for p in h.parameters(): p.requires_grad_(False)

    test_samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        n_test, seed=999)

    M, K = 5, 3
    print(f"[anchor_skip=ON (correct)]")
    r_on = measure_with_flag(model, n1_cont, n1_cont2, lm_head, root_emb_table,
                              prepare_inputs, is_qwen, test_samples, M, K, max_tokens,
                              anchor_skip=True)
    print(f"\n[anchor_skip=OFF (buggy)]")
    r_off = measure_with_flag(model, n1_cont, n1_cont2, lm_head, root_emb_table,
                               prepare_inputs, is_qwen, test_samples, M, K, max_tokens,
                               anchor_skip=False)

    sp_on  = [s/a for a, s in zip(r_on["ar_tps"],  r_on["ssd_tps"])]
    sp_off = [s/a for a, s in zip(r_off["ar_tps"], r_off["ssd_tps"])]

    print(f"\n{'='*70}\nANCHOR-SKIP IMPACT — {vlm_name}\n{'='*70}")
    print(f"{'mode':12s} {'sp (mean)':>12s} {'tokens':>8s} {'token match':>12s} {'exact':>8s}")
    print(f"{'ON (fixed)':12s} {np.mean(sp_on):>12.3f} {r_on['n_tokens']:>8d} "
          f"{r_on['token_match_rate']*100:>11.2f}% {r_on['n_exact']:>3d}/{r_on['n_pairs']:<3d}")
    print(f"{'OFF (buggy)':12s} {np.mean(sp_off):>12.3f} {r_off['n_tokens']:>8d} "
          f"{r_off['token_match_rate']*100:>11.2f}% {r_off['n_exact']:>3d}/{r_off['n_pairs']:<3d}")
    print(f"\nBug inflation factor: {np.mean(sp_off)/np.mean(sp_on):.3f}× (sp_off / sp_on)")
    print(f"Lossless drop: {(r_on['token_match_rate']-r_off['token_match_rate'])*100:.2f}pp")

    out = {
        "vlm": vlm_name, "n_test": n_test, "max_tokens": max_tokens, "M": M, "K": K,
        "anchor_skip_on":  {"sp_mean": float(np.mean(sp_on)),  "sp_std": float(np.std(sp_on)),
                             "token_match_rate": r_on["token_match_rate"],
                             "n_exact": r_on["n_exact"], "n_pairs": r_on["n_pairs"]},
        "anchor_skip_off": {"sp_mean": float(np.mean(sp_off)), "sp_std": float(np.std(sp_off)),
                             "token_match_rate": r_off["token_match_rate"],
                             "n_exact": r_off["n_exact"], "n_pairs": r_off["n_pairs"]},
        "bug_inflation_factor": float(np.mean(sp_off) / np.mean(sp_on)) if np.mean(sp_on) > 0 else None,
    }
    save_path = f"./results/B_anchor_skip_{vlm_name}.json"
    with open(save_path, "w") as f: json.dump(out, f, indent=2)
    print(f"\n[saved {save_path}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", required=True, choices=list(VLM_CONFIGS.keys()))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--n_test", type=int, default=30)
    p.add_argument("--max_tokens", type=int, default=64)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args.vlm, args.ckpt, args.n_test, args.max_tokens)
