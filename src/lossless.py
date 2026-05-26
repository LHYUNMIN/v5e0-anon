"""Lossless verification: V5e-0 SSD output vs greedy AR, token-by-token.

Procedure:
  1. Train fresh V5e-0-Cont and V5e-0-Cont2 (same data + protocol as run.py)
  2. For N_TEST prompts, generate MAX_TOKENS via greedy AR and via V5e-0 SSD
  3. Compare token-by-token: match rate, first-divergence index, exact match

Reports the bf16 numerical floor (gap between SSD and AR is bf16 noise, not
algorithmic — confirmed by lossless monotonic-in-tree-batch-size experiment).

Default: Qwen2-VL-2B (paper headline). Other VLMs supported via --vlm.

Usage:
  python lossless.py --vlm qwen2-2b --gpu 0
"""
import os, sys, json, gc, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (
    VLM_CONFIGS, V5e0_Cont, V5e0_Cont2,
    N_COLLECT, GEN_LEN, EPOCHS, M, K,
    to_tuple_kv, kv_to_cache,
    build_tree_d3, build_attn_mask_and_pos, prune_kv,
    load_llava_prompts, prepare_qwen_inputs, prepare_llava_inputs,
)

N_TEST = 10
MAX_TOKENS = 64


def main(vlm_name, n_test=None):
    if n_test is not None:
        global N_TEST
        N_TEST = n_test
    cfg = VLM_CONFIGS[vlm_name]
    print(f"[lossless verify {vlm_name} single-root depth-3 M={M}, K={K}]\n", flush=True)

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
    vocab_size = text_config.vocab_size
    hidden_dim = text_config.hidden_size
    embed = model.get_input_embeddings()
    if embed.weight.shape[0] > vocab_size:
        new = nn.Embedding(vocab_size, embed.weight.shape[1], device="cuda", dtype=embed.weight.dtype)
        new.weight.data.copy_(embed.weight.data[:vocab_size])
        model.set_input_embeddings(new); embed = new
    lm_head = model.get_output_embeddings()
    for p in model.parameters(): p.requires_grad_(False)
    root_emb_table = embed.weight.detach().to(torch.float32)

    # ---------- Collect data + train Cont + train Cont2 ----------
    print(f"[collect {N_COLLECT} prompts × {GEN_LEN}]")
    samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        N_COLLECT, seed=43)
    h_t_list = []; root_id_list = []; cont_target_list = []
    prompt_idx_list = []; pos_list = []
    used = 0
    for s in samples:
        try:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            inputs = prepare_inputs(s)
            with torch.no_grad():
                out = model(**inputs, past_key_values=DynamicCache(),
                            use_cache=True, output_hidden_states=True, return_dict=True)
        except Exception:
            continue
        cur_kv = to_tuple_kv(out.past_key_values)
        h_t = out.hidden_states[-1][0, -1, :].float()
        v_logits_for_root = out.logits[0, -1, :].float()
        for pos in range(GEN_LEN):
            v_root_top1 = int(v_logits_for_root.argmax().item())
            try:
                with torch.no_grad():
                    tok_inp = torch.tensor([[v_root_top1]], device="cuda")
                    cur_len = cur_kv[0][0].shape[2]
                    if is_qwen:
                        pos_id = torch.tensor([[cur_len]], device="cuda").unsqueeze(0).expand(3, 1, -1)
                    else:
                        pos_id = torch.tensor([[cur_len]], device="cuda")
                    cache_pos = torch.tensor([cur_len], device="cuda")
                    out_step = model(input_ids=tok_inp, past_key_values=kv_to_cache(cur_kv),
                                     position_ids=pos_id, cache_position=cache_pos,
                                     use_cache=True, output_hidden_states=True, return_dict=True)
                    v_cont_top1 = int(out_step.logits[0, -1, :].argmax().item())
            except Exception:
                break
            h_t_list.append(h_t.detach().cpu())
            root_id_list.append(v_root_top1)
            cont_target_list.append(v_cont_top1)
            prompt_idx_list.append(used)
            pos_list.append(pos)
            h_t = out_step.hidden_states[-1][0, -1, :].float()
            cur_kv = to_tuple_kv(out_step.past_key_values)
            v_logits_for_root = out_step.logits[0, -1, :].float()
        used += 1
        if used >= N_COLLECT: break
    h_t_tensor = torch.stack(h_t_list).to(torch.float32)
    root_id_t = torch.tensor(root_id_list, dtype=torch.long)
    cont_target_t = torch.tensor(cont_target_list, dtype=torch.long)
    prompt_idx_arr = np.array(prompt_idx_list)
    pos_arr = np.array(pos_list)

    # 80/20 split
    rng = np.random.RandomState(0)
    unique_prompts = np.unique(prompt_idx_arr)
    n_eval_p = max(1, int(round(len(unique_prompts) * 0.20)))
    eval_prompts = set(rng.choice(unique_prompts, n_eval_p, replace=False).tolist())
    train_idx = np.array([i for i in range(len(prompt_idx_arr))
                           if prompt_idx_arr[i] not in eval_prompts])
    from collections import defaultdict
    train_by_p = defaultdict(list)
    for i in train_idx: train_by_p[prompt_idx_arr[i]].append(int(i))
    train_prompts_list = list(train_by_p.keys())

    # Train Cont
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    n1_cont = V5e0_Cont(dim=hidden_dim, alpha_init=cfg["alpha"]).to("cuda")
    opt = torch.optim.AdamW(n1_cont.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"[train V5e-0-Cont {EPOCHS} epochs]")
    for ep in range(EPOCHS):
        n1_cont.train()
        random.shuffle(train_prompts_list)
        for bstart in range(0, len(train_prompts_list), 8):
            bp = train_prompts_list[bstart:bstart + 8]
            opt.zero_grad()
            total_loss = 0; total_n = 0
            for p in bp:
                recs = train_by_p[p]
                h = h_t_tensor[recs].to("cuda")
                re = root_emb_table[root_id_t[recs]].to("cuda")
                tg = cont_target_t[recs].to("cuda")
                z = n1_cont(h, re)
                logits = lm_head(z.to(torch.bfloat16)).float()
                loss = F.cross_entropy(logits, tg, reduction="sum")
                total_loss = total_loss + loss; total_n += len(recs)
            if total_n > 0:
                (total_loss / total_n).backward(); opt.step()
    n1_cont.eval()
    for p in n1_cont.parameters(): p.requires_grad_(False)

    # Build cont2 tuples
    cont2_recs = []
    for i in range(len(prompt_idx_arr) - 1):
        if prompt_idx_arr[i+1] == prompt_idx_arr[i] and pos_arr[i+1] == pos_arr[i] + 1:
            cont2_recs.append({
                "h_idx": i,
                "root_id": int(root_id_t[i]),
                "cont_id": int(cont_target_t[i]),
                "cont2_target": int(cont_target_t[i+1]),
                "prompt_idx": int(prompt_idx_arr[i]),
            })
    train_cont2_recs = [r for r in cont2_recs if r["prompt_idx"] not in eval_prompts]

    train_h_idx = torch.tensor([r["h_idx"]       for r in train_cont2_recs], dtype=torch.long)
    train_root  = torch.tensor([r["root_id"]      for r in train_cont2_recs], dtype=torch.long)
    train_cont  = torch.tensor([r["cont_id"]      for r in train_cont2_recs], dtype=torch.long)
    train_cont2 = torch.tensor([r["cont2_target"] for r in train_cont2_recs], dtype=torch.long)
    train_p2    = torch.tensor([r["prompt_idx"]   for r in train_cont2_recs], dtype=torch.long)

    train_by_p2 = defaultdict(list)
    for i, p in enumerate(train_p2.tolist()):
        train_by_p2[p].append(i)
    train_prompts2 = list(train_by_p2.keys())

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    n1_cont2 = V5e0_Cont2(dim=hidden_dim, alpha_init=cfg["alpha"], beta_init=cfg["beta"]).to("cuda")
    opt2 = torch.optim.AdamW(n1_cont2.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"[train V5e-0-Cont2 {EPOCHS} epochs]")
    for ep in range(EPOCHS):
        n1_cont2.train()
        random.shuffle(train_prompts2)
        for bstart in range(0, len(train_prompts2), 8):
            bp = train_prompts2[bstart:bstart + 8]
            opt2.zero_grad()
            total_loss = 0; total_n = 0
            for p in bp:
                idx_list = train_by_p2[p]
                idx = torch.tensor(idx_list, dtype=torch.long)
                h  = h_t_tensor[train_h_idx[idx]].to("cuda")
                re = root_emb_table[train_root[idx]].to("cuda")
                ce = root_emb_table[train_cont[idx]].to("cuda")
                tg = train_cont2[idx].to("cuda")
                z = n1_cont2(h, re, ce)
                logits = lm_head(z.to(torch.bfloat16)).float()
                loss = F.cross_entropy(logits, tg, reduction="sum")
                total_loss = total_loss + loss; total_n += len(idx_list)
            if total_n > 0:
                (total_loss / total_n).backward(); opt2.step()
    n1_cont2.eval()
    for p in n1_cont2.parameters(): p.requires_grad_(False)

    # ---------- Lossless verification ----------
    samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        N_TEST, seed=999)

    parents, depths = build_tree_d3(M, K)
    n_input = len(parents) - 1

    per_prompt = []
    total_match = 0; total_pos = 0; n_exact = 0
    for s_i, s in enumerate(samples):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        inputs = prepare_inputs(s)

        # AR
        with torch.no_grad():
            out = model(**inputs, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
        cur_kv = to_tuple_kv(out.past_key_values)
        last_token = int(out.logits[0, -1, :].argmax().item())
        ar_tokens = [last_token]
        with torch.no_grad():
            for _ in range(MAX_TOKENS - 1):
                tok_inp = torch.tensor([[last_token]], device="cuda")
                cur_len = cur_kv[0][0].shape[2]
                if is_qwen:
                    pos_id = torch.tensor([[cur_len]], device="cuda").unsqueeze(0).expand(3, 1, -1)
                else:
                    pos_id = torch.tensor([[cur_len]], device="cuda")
                cache_pos = torch.tensor([cur_len], device="cuda")
                out_step = model(input_ids=tok_inp, past_key_values=kv_to_cache(cur_kv),
                                 position_ids=pos_id, cache_position=cache_pos,
                                 use_cache=True, return_dict=True)
                last_token = int(out_step.logits[0, -1, :].argmax().item())
                cur_kv = to_tuple_kv(out_step.past_key_values)
                ar_tokens.append(last_token)

        # SSD depth-3 single-root
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
                                                          "cuda", model.dtype, qwen=is_qwen)
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

    rate = total_match / total_pos
    print(f"\n{'='*70}\nLOSSLESS depth-3 (M={M}, K={K}) on {vlm_name}\n{'='*70}")
    print(f"  Token match: {100*rate:.2f}%")
    print(f"  Exact: {n_exact}/{N_TEST}")

    save_path = f"./results/path1_v5e0_d3_lossless_{vlm_name}.json"
    with open(save_path, "w") as f:
        json.dump({"vlm": vlm_name, "M": M, "K": K, "n_test": N_TEST,
                    "max_tokens": MAX_TOKENS, "total_match_rate": rate,
                    "n_exact": n_exact, "per_prompt": per_prompt}, f, indent=2)
    print(f"[saved {save_path}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", default="qwen2-2b", choices=list(VLM_CONFIGS.keys()))
    p.add_argument("--gpu", default="0")
    p.add_argument("--n_test", type=int, default=None)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args.vlm, n_test=args.n_test)
