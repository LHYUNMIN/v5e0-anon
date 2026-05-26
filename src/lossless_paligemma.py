"""Lossless verification for PaliGemma using the saved drafter checkpoint.
Reuses ./checkpoints/v5e0_paligemma.pt (trained Cont/Cont2 weights).
"""
import os, sys, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (
    V5e0_Cont, V5e0_Cont2, M, K,
    to_tuple_kv, kv_to_cache,
    build_tree_d3, build_attn_mask_and_pos, prune_kv, load_llava_prompts,
)
from run_new_vlms import prepare_paligemma_inputs


def main(n_test, ckpt_path):
    from transformers.cache_utils import DynamicCache
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

    MODEL_ID = "google/paligemma-3b-mix-448"
    MAX_TOKENS = 64

    print(f"[lossless PaliGemma | n_test={n_test} | ckpt={ckpt_path}]")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    prep = lambda s: prepare_paligemma_inputs(proc, s["image"], s["prompt"], "cuda")

    tc = getattr(model.config, "text_config", model.config)
    vocab = getattr(tc, "vocab_size", None) or model.config.vocab_size
    hidden_dim = getattr(tc, "hidden_size", None) or model.config.hidden_size
    emb = model.get_input_embeddings()
    if emb.weight.shape[0] > vocab:
        new = nn.Embedding(vocab, emb.weight.shape[1], device="cuda", dtype=emb.weight.dtype)
        new.weight.data.copy_(emb.weight.data[:vocab])
        model.set_input_embeddings(new); emb = new
    lm_head = model.get_output_embeddings()
    for p in model.parameters(): p.requires_grad_(False)
    root_emb_table = emb.weight.detach().to(torch.float32)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n1_cont = V5e0_Cont(dim=hidden_dim, alpha_init=ckpt["alpha"]).to("cuda")
    n1_cont.Q_proj.weight.data.copy_(ckpt["W_Q1"].to("cuda"))
    n1_cont.Q_proj.bias.data.copy_(ckpt["W_Q1_bias"].to("cuda"))
    n1_cont.eval()
    n1_cont2 = V5e0_Cont2(dim=hidden_dim, alpha_init=ckpt["alpha"], beta_init=ckpt["beta"]).to("cuda")
    n1_cont2.Q_proj.weight.data.copy_(ckpt["W_Q2"].to("cuda"))
    n1_cont2.Q_proj.bias.data.copy_(ckpt["W_Q2_bias"].to("cuda"))
    n1_cont2.eval()
    print(f"  loaded ckpt (hidden_dim={hidden_dim}, alpha={ckpt['alpha']}, beta={ckpt['beta']})")

    samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        n_test, seed=999)

    parents, depths = build_tree_d3(M, K)
    n_input = len(parents) - 1

    per_prompt = []
    total_match = 0; total_pos = 0; n_exact = 0
    for s_i, s in enumerate(samples):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        try:
            inputs = prep(s)
        except Exception as e:
            print(f"  prompt {s_i+1}: skip (prep err: {e})"); continue

        with torch.no_grad():
            out = model(**inputs, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
        cur_kv = to_tuple_kv(out.past_key_values)
        last_token = int(out.logits[0, -1, :].argmax().item())
        ar_tokens = [last_token]
        with torch.no_grad():
            for _ in range(MAX_TOKENS - 1):
                tok = torch.tensor([[last_token]], device="cuda")
                cur_len = cur_kv[0][0].shape[2]
                pid = torch.tensor([[cur_len]], device="cuda")
                cp = torch.tensor([cur_len], device="cuda")
                out_step = model(input_ids=tok, past_key_values=kv_to_cache(cur_kv),
                                 position_ids=pid, cache_position=cp,
                                 use_cache=True, return_dict=True)
                last_token = int(out_step.logits[0, -1, :].argmax().item())
                cur_kv = to_tuple_kv(out_step.past_key_values)
                ar_tokens.append(last_token)

        with torch.no_grad():
            out = model(**inputs, past_key_values=DynamicCache(),
                        use_cache=True, output_hidden_states=True, return_dict=True)
        cur_kv = to_tuple_kv(out.past_key_values)
        prompt_len = cur_kv[0][0].shape[2]
        last_h_t = out.hidden_states[-1][0, -1, :].float()
        last_logits = out.logits[0, -1, :].float()
        ssd_tokens = []
        n_rounds = 0
        with torch.no_grad():
            while len(ssd_tokens) < MAX_TOKENS:
                anchor_argmax = int(last_logits.argmax().item())
                root_emb = root_emb_table[anchor_argmax].unsqueeze(0)
                z = n1_cont(last_h_t.unsqueeze(0), root_emb)
                cont_logits = lm_head(z.to(torch.bfloat16)).float()
                cont_topm = cont_logits.topk(M, dim=-1).indices[0]

                h_t_b = last_h_t.unsqueeze(0).expand(M, -1)
                root_emb_b = root_emb.expand(M, -1)
                cont_embs = root_emb_table[cont_topm]
                z2 = n1_cont2(h_t_b, root_emb_b, cont_embs)
                cont2_logits = lm_head(z2.to(torch.bfloat16)).float()
                cont2_topk = cont2_logits.topk(K, dim=-1).indices

                tree_input_ids = [anchor_argmax] + cont_topm.tolist()
                for cont_i in range(M):
                    tree_input_ids.extend(cont2_topk[cont_i].tolist())
                tree_input_t = torch.tensor([tree_input_ids], device="cuda")
                mask, pos_ids = build_attn_mask_and_pos(parents, depths, prompt_len,
                                                       "cuda", model.dtype, qwen=False)
                cache_pos = torch.arange(prompt_len, prompt_len + n_input, device="cuda")
                out_tree = model(input_ids=tree_input_t,
                                 past_key_values=kv_to_cache(cur_kv),
                                 attention_mask=mask, position_ids=pos_ids,
                                 cache_position=cache_pos,
                                 use_cache=True, output_hidden_states=True, return_dict=True)
                tree_logits = out_tree.logits[0]
                tree_hidden = out_tree.hidden_states[-1][0]
                kv_after = to_tuple_kv(out_tree.past_key_values)

                cont_argmax = int(tree_logits[0].argmax().item())
                accepted_cont_idx = None
                for c_j in range(M):
                    pos = 1 + c_j
                    if tree_input_ids[pos] == cont_argmax:
                        accepted_cont_idx = pos; break

                is_first = (n_rounds == 0)
                base = [anchor_argmax] if is_first else []

                if accepted_cont_idx is not None:
                    cont_position_in_array = accepted_cont_idx - 1
                    cont2_start_idx = 1 + M + cont_position_in_array * K
                    cont2_argmax = int(tree_logits[accepted_cont_idx].argmax().item())
                    accepted_cont2_idx = None
                    for c2_j in range(K):
                        pos = cont2_start_idx + c2_j
                        if tree_input_ids[pos] == cont2_argmax:
                            accepted_cont2_idx = pos; break
                    if accepted_cont2_idx is not None:
                        bonus = int(tree_logits[accepted_cont2_idx].argmax().item())
                        new_tokens = base + [cont_argmax, cont2_argmax, bonus]
                        accepted_indices = [0, accepted_cont_idx, accepted_cont2_idx]
                        new_last_logits = tree_logits[accepted_cont2_idx].float()
                        new_last_h_t = tree_hidden[accepted_cont2_idx].float()
                    else:
                        new_tokens = base + [cont_argmax, cont2_argmax]
                        accepted_indices = [0, accepted_cont_idx]
                        new_last_logits = tree_logits[accepted_cont_idx].float()
                        new_last_h_t = tree_hidden[accepted_cont_idx].float()
                else:
                    new_tokens = base + [cont_argmax]
                    accepted_indices = [0]
                    new_last_logits = tree_logits[0].float()
                    new_last_h_t = tree_hidden[0].float()

                cur_kv = prune_kv(kv_after, prompt_len, accepted_indices)
                prompt_len += len(accepted_indices)
                last_logits = new_last_logits
                last_h_t = new_last_h_t
                ssd_tokens.extend(new_tokens)
                n_rounds += 1
        ssd_tokens = ssd_tokens[:MAX_TOKENS]

        L = min(len(ar_tokens), len(ssd_tokens))
        match = sum(1 for i in range(L) if ar_tokens[i] == ssd_tokens[i])
        first_diff = next((i for i in range(L) if ar_tokens[i] != ssd_tokens[i]), L)
        is_exact = (ar_tokens[:L] == ssd_tokens[:L])
        per_prompt.append({"prompt_idx": s_i, "match": match, "L": L,
                           "first_diff": first_diff, "exact": is_exact})
        total_match += match; total_pos += L
        if is_exact: n_exact += 1
        print(f"  prompt {s_i+1}: {match}/{L} ({100*match/L:.1f}%), "
              f"first_diff={first_diff}, exact={is_exact}", flush=True)

    rate = total_match / total_pos if total_pos else 0.0
    print(f"\n{'='*70}\nLOSSLESS depth-3 (M={M}, K={K}) on PaliGemma-3B\n{'='*70}")
    print(f"  Token match: {100*rate:.2f}%")
    print(f"  Exact: {n_exact}/{len(per_prompt)}")

    save_path = "./results/path1_v5e0_d3_lossless_paligemma.json"
    with open(save_path, "w") as f:
        json.dump({"vlm": "paligemma", "M": M, "K": K,
                   "n_test": len(per_prompt), "max_tokens": MAX_TOKENS,
                   "total_match_rate": rate, "n_exact": n_exact,
                   "per_prompt": per_prompt}, f, indent=2)
    print(f"[saved {save_path}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="0")
    p.add_argument("--n_test", type=int, default=30)
    p.add_argument("--ckpt", default="./checkpoints/v5e0_paligemma.pt")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(n_test=args.n_test, ckpt_path=args.ckpt)
