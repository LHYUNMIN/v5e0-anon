#!/usr/bin/env python3
"""Baseline measurement: static (16, 3) tree depth-2 — for paper Section 4.2 progression.

Reuses V5e-0-Cont trained head (no Cont2 needed for depth-2 only). Loads from a saved
drafter checkpoint produced by `run.py --save_ckpt`.

Usage:
  python baseline.py --vlm qwen2-2b --ckpt ./checkpoints/v5e0_qwen2-2b.pt --gpu 0 --n_test 100
"""
import os, sys, json, gc, time, random, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (
    VLM_CONFIGS, V5e0_Cont, MAX_TOKENS,
    to_tuple_kv, kv_to_cache, load_llava_prompts,
    prepare_qwen_inputs, prepare_llava_inputs,
)


def build_tree_static_16_3():
    """Canonical multi-root (16, 3) tree: 1 anchor + 16 roots + 16*3 conts = 64 input."""
    parents = [-1] + [0] * 16
    depths  = [0]  + [1] * 16
    for j in range(16):
        for _ in range(3):
            parents.append(1 + j); depths.append(2)
    return parents, depths


def build_attn_mask_and_pos(parents, depths, prompt_len, device, dtype, qwen=False):
    seq = len(parents) - 1
    mask = torch.full((seq, prompt_len + seq), float('-inf'), device=device, dtype=dtype)
    mask[:, :prompt_len] = 0.0
    for i in range(1, len(parents)):
        in_idx = i - 1
        mask[in_idx, prompt_len + in_idx] = 0.0
        cur = parents[i]
        while cur != -1 and cur != 0:
            mask[in_idx, prompt_len + cur - 1] = 0.0
            cur = parents[cur]
    pos = torch.tensor(
        [prompt_len + depths[i] - 1 for i in range(1, len(parents))],
        device=device, dtype=torch.long,
    )
    if qwen:
        pos = pos.unsqueeze(0).unsqueeze(0).expand(3, 1, -1).contiguous()
    else:
        pos = pos.unsqueeze(0)
    return mask.unsqueeze(0).unsqueeze(0), pos


def prune_kv(kv_tuple, prompt_len, accepted_indices):
    if not accepted_indices:
        return tuple((k[..., :prompt_len, :].contiguous(),
                      v[..., :prompt_len, :].contiguous()) for k, v in kv_tuple)
    idx = torch.tensor([prompt_len + i for i in accepted_indices],
                        device=kv_tuple[0][0].device, dtype=torch.long)
    new = []
    for k, v in kv_tuple:
        pk, pv = k[..., :prompt_len, :], v[..., :prompt_len, :]
        sk, sv = k.index_select(-2, idx), v.index_select(-2, idx)
        new.append((torch.cat([pk, sk], dim=-2).contiguous(),
                    torch.cat([pv, sv], dim=-2).contiguous()))
    return tuple(new)


@torch.no_grad()
def measure_static(model, n1_cont, lm_head, root_emb_table, prepare_inputs, is_qwen,
                    test_samples, max_tokens):
    """Measure static (16, 3) depth-2 sp paired with AR."""
    from transformers.cache_utils import DynamicCache

    parents, depths = build_tree_static_16_3()
    n_input = len(parents) - 1   # 64

    ar_tps_list = []; ssd_tps_list = []
    n_d2 = 0; n_rounds_total = 0; n_tokens_total = 0

    for s_i, s in enumerate(test_samples):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

        # AR
        try:
            inputs = prepare_inputs(s)
            out = model(**inputs, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
        except Exception: continue
        kv = to_tuple_kv(out.past_key_values)
        last = int(out.logits[0, -1, :].argmax().item())
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            for _ in range(max_tokens):
                x = torch.tensor([[last]], device="cuda")
                cur = kv[0][0].shape[2]
                pid = torch.tensor([[cur]], device="cuda")
                if is_qwen: pid = pid.unsqueeze(0).expand(3, 1, -1)
                cp = torch.tensor([cur], device="cuda")
                out = model(input_ids=x, past_key_values=kv_to_cache(kv),
                            position_ids=pid, cache_position=cp, use_cache=True, return_dict=True)
                last = int(out.logits[0, -1, :].argmax().item())
                kv = to_tuple_kv(out.past_key_values)
        except Exception: continue
        torch.cuda.synchronize()
        ar_tps_list.append(max_tokens / (time.perf_counter() - t0))

        # SSD static (16, 3)
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
                # Free-Root variant: take verifier's top-16 as roots; r_1 = argmax.
                top16 = ll.topk(16).indices.tolist()
                anchor_argmax = top16[0]

                # Cont head: top-3 cont per root (batched over 16)
                root_embs = root_emb_table[torch.tensor(top16, device=h_t.device)]
                h_b = h_t.unsqueeze(0).expand(16, -1)
                z = n1_cont(h_b, root_embs)
                cont_logits = lm_head(z.to(torch.bfloat16)).float()
                cont_top3 = cont_logits.topk(3, dim=-1).indices   # [16, 3]

                # Build static (16, 3) tree
                tree_ids = list(top16)
                for r in range(16):
                    tree_ids.extend(cont_top3[r].tolist())
                xt = torch.tensor([tree_ids], device=h_t.device)
                mask, pos_ids = build_attn_mask_and_pos(parents, depths, prompt_len,
                                                          "cuda", model.dtype, qwen=is_qwen)
                cp = torch.arange(prompt_len, prompt_len + n_input, device="cuda")
                out = model(input_ids=xt, past_key_values=kv_to_cache(kv),
                            attention_mask=mask, position_ids=pos_ids, cache_position=cp,
                            use_cache=True, output_hidden_states=True, return_dict=True)
                tree_logits = out.logits[0]
                tree_hidden = out.hidden_states[-1][0]
                kv_after = to_tuple_kv(out.past_key_values)

                # Verify: under Free-Root + greedy, only root_1 (top16[0]) and its conts can accept
                cont_argmax = int(tree_logits[0].argmax().item())
                accepted_cont_idx = None
                # Conts under root_1 are at indices 16, 17, 18 in tree_ids
                for c_j in range(3):
                    pos = 16 + 0 * 3 + c_j
                    if tree_ids[pos] == cont_argmax:
                        accepted_cont_idx = pos; break

                is_first = (n_rounds == 0)
                base = [anchor_argmax] if is_first else []

                if accepted_cont_idx is not None:
                    bonus = int(tree_logits[accepted_cont_idx].argmax().item())
                    new_toks = base + [cont_argmax, bonus]
                    acc_idx = [0, accepted_cont_idx]
                    new_h = tree_hidden[accepted_cont_idx].float()
                    new_ll = tree_logits[accepted_cont_idx].float()
                    n_d2 += 1
                else:
                    new_toks = base + [cont_argmax]
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
        ssd_tps_list.append(len(gen) / (time.perf_counter() - t0))
        n_rounds_total += n_rounds; n_tokens_total += len(gen)

    return ar_tps_list, ssd_tps_list, n_d2, n_rounds_total, n_tokens_total


def main(vlm_name, ckpt_path, n_test):
    cfg = VLM_CONFIGS[vlm_name]
    print(f"[STATIC (16, 3) BASELINE: {vlm_name}, n_test={n_test}]\n", flush=True)

    sys.path.insert(0, "./vendor")
    from runtime_env import strip_user_site_packages
    strip_user_site_packages()
    from transformers.cache_utils import DynamicCache

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
    elif cfg["model_class"] == "llava-next":
        from transformers import LlavaNextForConditionalGeneration, AutoProcessor
        model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        processor = AutoProcessor.from_pretrained(cfg["model_id"])
        prepare_inputs = lambda s: prepare_llava_inputs(processor, s["image"], s["prompt"], "cuda")
        is_qwen = False

    text_config = getattr(model.config, "text_config", model.config)
    vocab = text_config.vocab_size
    hidden_dim = text_config.hidden_size
    emb = model.get_input_embeddings()
    if emb.weight.shape[0] > vocab:
        new = nn.Embedding(vocab, emb.weight.shape[1], device="cuda", dtype=emb.weight.dtype)
        new.weight.data.copy_(emb.weight.data[:vocab])
        model.set_input_embeddings(new); emb = new
    lm_head = model.get_output_embeddings()
    for p in model.parameters(): p.requires_grad_(False)
    root_emb_table = emb.weight.detach().to(torch.float32)

    # Load V5e-0-Cont from ckpt
    print(f"[loading drafter ckpt {ckpt_path}]")
    state = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    alpha = state.get("alpha", 30.0)
    n1_cont = V5e0_Cont(dim=hidden_dim, alpha_init=alpha).to("cuda")
    n1_cont.Q_proj.weight.data.copy_(state["W_Q1"].float())
    n1_cont.Q_proj.bias.data.copy_(state["W_Q1_bias"].float())
    n1_cont.eval()
    for p in n1_cont.parameters(): p.requires_grad_(False)

    # Test samples (same seed as run.py)
    test_samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        n_test, seed=999)

    print(f"[walltime test STATIC (16, 3), {n_test} prompts × {MAX_TOKENS}]")
    ar_list, ssd_list, n_d2, n_rounds, n_tokens = measure_static(
        model, n1_cont, lm_head, root_emb_table, prepare_inputs, is_qwen,
        test_samples, MAX_TOKENS)
    sp = [s/a for a, s in zip(ar_list, ssd_list)]
    ar_a = np.array(ar_list); ssd_a = np.array(ssd_list); sp_a = np.array(sp)
    d2_rate = n_d2 / n_rounds if n_rounds > 0 else 0
    tpi = n_tokens / n_rounds if n_rounds > 0 else 0

    print(f"\n{'='*70}")
    print(f"{vlm_name} STATIC (16, 3) depth-2 baseline (n_test={n_test})")
    print(f"{'='*70}")
    print(f"  AR:    {ar_a.mean():.2f} ± {ar_a.std():.2f} t/s")
    print(f"  SSD:   {ssd_a.mean():.2f} ± {ssd_a.std():.2f} t/s")
    print(f"  sp:    {sp_a.mean():.3f} ± {sp_a.std():.3f}")
    print(f"        median {np.median(sp_a):.3f}, range [{sp_a.min():.2f}, {sp_a.max():.2f}]")
    print(f"  TPI {tpi:.2f}, d2 {d2_rate:.3f}")

    out = {
        "vlm": vlm_name, "tree": "static_16_3",
        "n_input_tokens": 64,
        "ar_tps":  {"mean": float(ar_a.mean()),  "std": float(ar_a.std())},
        "ssd_tps": {"mean": float(ssd_a.mean()), "std": float(ssd_a.std())},
        "sp": {"mean": float(sp_a.mean()), "std": float(sp_a.std()),
               "median": float(np.median(sp_a)),
               "min": float(sp_a.min()), "max": float(sp_a.max())},
        "tpi": tpi, "d2_rate": d2_rate,
        "n_prompts": len(ar_list),
        "raw": {"ar": ar_list, "ssd": ssd_list, "sp": sp},
    }
    save_path = f"./results/T1_4_static_{vlm_name}.json"
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved {save_path}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", required=True, choices=list(VLM_CONFIGS.keys()))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--n_test", type=int, default=100)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args.vlm, args.ckpt, args.n_test)
