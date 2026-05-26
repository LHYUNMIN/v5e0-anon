#!/usr/bin/env python3
"""V5e-0 on text-only LLMs (cross-modality generalization test).

Identical method as run.py (VLM version): Free-Root proposal + V5e-0-Cont (Linear,
identity init) + V5e-0-Cont2 (Linear) trained with CE, single-root depth-3 tree
(M=5, K=3). The only difference is the verifier: an LLM with no vision encoder.

If V5e-0 sp ≈ VLM sp on the same hardware, the paper's claim that the drafter is
modality-agnostic is validated: vision tokens really weren't needed.

Usage:
  python llm_v5e0.py --llm qwen3-4b      --gpu 0
  python llm_v5e0.py --llm llama3.2-3b  --gpu 0
"""
import os, sys, json, gc, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (
    V5e0_Cont, V5e0_Cont2,
    to_tuple_kv, kv_to_cache,
    build_tree_d3, build_attn_mask_and_pos, prune_kv,
)


# ============================================================
# Hyperparams (match VLM run.py exactly for clean comparison)
# ============================================================
N_COLLECT  = 200
GEN_LEN    = 32
N_TEST     = 30
MAX_TOKENS = 64
EPOCHS     = 100
M = 5
K = 3


LLM_CONFIGS = {
    "qwen3-4b":     {"model_id": "Qwen/Qwen3-4B",                       "alpha": 30.0, "beta": 30.0},
    "llama3.2-3b":  {"model_id": "meta-llama/Llama-3.2-3B-Instruct",   "alpha": 30.0, "beta": 30.0},
    "llama3.2-1b":  {"model_id": "meta-llama/Llama-3.2-1B-Instruct",   "alpha": 30.0, "beta": 30.0},
    "qwen2-0.5b":   {"model_id": "Qwen/Qwen2-0.5B-Instruct",            "alpha": 30.0, "beta": 30.0},
}


# ============================================================
# Text prompts: reuse llava_messages text portion (no image)
# ============================================================
def load_text_prompts(path, n, seed):
    rng = random.Random(seed)
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            msgs = d['messages']
            txt = next((c['text'] for c in msgs[0]['content'] if c['type'] == 'text'), None)
            if txt:
                rows.append(txt)
                if len(rows) >= n * 3:
                    break
    rng.shuffle(rows)
    return rows[:n]


def prepare_llm_inputs(tokenizer, prompt_text, device):
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").to(device)


# ============================================================
# Measurement (mirrors run.py measure_d3 but no is_qwen branching since
# all LLMs here use standard 1D RoPE — no multimodal mrope)
# ============================================================
def measure_d3_llm(model, n1_cont, n1_cont2, lm_head, root_emb_table,
                    tokenizer, test_prompts, M, K, max_tokens):
    from transformers.cache_utils import DynamicCache
    parents, depths = build_tree_d3(M, K)
    n_input = len(parents) - 1

    ar_tps_list = []; ssd_tps_list = []
    n_d2 = 0; n_d3 = 0; n_rounds_total = 0; n_tokens_total = 0

    for s_i, prompt in enumerate(test_prompts):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

        # ---------- AR baseline ----------
        try:
            inputs = prepare_llm_inputs(tokenizer, prompt, "cuda")
            with torch.no_grad():
                out = model(**inputs, past_key_values=DynamicCache(),
                            use_cache=True, return_dict=True)
        except Exception as e:
            print(f"  prompt {s_i} AR prefill failed: {e}")
            continue
        cur_kv = to_tuple_kv(out.past_key_values)
        last_token = int(out.logits[0, -1, :].argmax().item())
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            with torch.no_grad():
                for _ in range(max_tokens):
                    tok = torch.tensor([[last_token]], device="cuda")
                    cur_len = cur_kv[0][0].shape[2]
                    pid = torch.tensor([[cur_len]], device="cuda")
                    cp = torch.tensor([cur_len], device="cuda")
                    out_step = model(input_ids=tok, past_key_values=kv_to_cache(cur_kv),
                                     position_ids=pid, cache_position=cp,
                                     use_cache=True, return_dict=True)
                    last_token = int(out_step.logits[0, -1, :].argmax().item())
                    cur_kv = to_tuple_kv(out_step.past_key_values)
        except Exception as e:
            print(f"  prompt {s_i} AR decode failed: {e}")
            continue
        torch.cuda.synchronize()
        ar_tps_list.append(max_tokens / (time.perf_counter() - t0))

        # ---------- V5e-0 SSD (single-root depth-3) ----------
        try:
            inputs = prepare_llm_inputs(tokenizer, prompt, "cuda")
            with torch.no_grad():
                out = model(**inputs, past_key_values=DynamicCache(),
                            use_cache=True, output_hidden_states=True, return_dict=True)
        except Exception:
            continue
        cur_kv = to_tuple_kv(out.past_key_values)
        prompt_len = cur_kv[0][0].shape[2]
        last_h_t = out.hidden_states[-1][0, -1, :].float()
        last_logits = out.logits[0, -1, :].float()

        generated = []; n_rounds = 0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            with torch.no_grad():
                while len(generated) < max_tokens:
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
                    for ci in range(M):
                        tree_input_ids.extend(cont2_topk[ci].tolist())
                    tree_input_t = torch.tensor([tree_input_ids], device="cuda")

                    mask, pos_ids = build_attn_mask_and_pos(
                        parents, depths, prompt_len, "cuda", model.dtype, qwen=False)
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
                        cont_pos_in_arr = accepted_cont_idx - 1
                        c2_start = 1 + M + cont_pos_in_arr * K
                        cont2_argmax = int(tree_logits[accepted_cont_idx].argmax().item())
                        accepted_c2_idx = None
                        for c2_j in range(K):
                            pos = c2_start + c2_j
                            if tree_input_ids[pos] == cont2_argmax:
                                accepted_c2_idx = pos; break
                        if accepted_c2_idx is not None:
                            bonus = int(tree_logits[accepted_c2_idx].argmax().item())
                            new_tokens = base + [cont_argmax, cont2_argmax, bonus]
                            accepted_indices = [0, accepted_cont_idx, accepted_c2_idx]
                            new_last_logits = tree_logits[accepted_c2_idx].float()
                            new_last_h_t = tree_hidden[accepted_c2_idx].float()
                            n_d3 += 1
                        else:
                            new_tokens = base + [cont_argmax, cont2_argmax]
                            accepted_indices = [0, accepted_cont_idx]
                            new_last_logits = tree_logits[accepted_cont_idx].float()
                            new_last_h_t = tree_hidden[accepted_cont_idx].float()
                            n_d2 += 1
                    else:
                        new_tokens = base + [cont_argmax]
                        accepted_indices = [0]
                        new_last_logits = tree_logits[0].float()
                        new_last_h_t = tree_hidden[0].float()

                    cur_kv = prune_kv(kv_after, prompt_len, accepted_indices)
                    prompt_len += len(accepted_indices)
                    last_logits = new_last_logits
                    last_h_t = new_last_h_t
                    generated.extend(new_tokens)
                    n_rounds += 1
        except Exception as e:
            print(f"  prompt {s_i} SSD failed: {e}", flush=True)
            continue
        torch.cuda.synchronize()
        ssd_tps_list.append(len(generated) / (time.perf_counter() - t0))
        n_rounds_total += n_rounds; n_tokens_total += len(generated)

    return ar_tps_list, ssd_tps_list, n_d2, n_d3, n_rounds_total, n_tokens_total


# ============================================================
# Main pipeline
# ============================================================
def main(llm_name, n_test=None, max_tokens=None):
    if n_test is not None:
        global N_TEST; N_TEST = n_test
    if max_tokens is not None:
        global MAX_TOKENS; MAX_TOKENS = max_tokens
    cfg = LLM_CONFIGS[llm_name]
    print(f"[LLM: {llm_name}, single-root depth-3 M={M}, K={K}]\n", flush=True)

    sys.path.insert(0, "./vendor")
    from runtime_env import strip_user_site_packages; strip_user_site_packages()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    print(f"  hidden_dim={hidden_dim}, vocab={vocab_size}\n", flush=True)

    # ---------- Phase 1: collect data ----------
    print(f"[collect {N_COLLECT} prompts × {GEN_LEN}]", flush=True)
    prompts = load_text_prompts(
        "./data/llava_messages_100k.jsonl",
        N_COLLECT, seed=43)
    h_t_list, root_id_list, cont_target_list = [], [], []
    prompt_idx_list, pos_list = [], []
    used = 0; t0 = time.time()
    for prompt in prompts:
        try:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            inputs = prepare_llm_inputs(tokenizer, prompt, "cuda")
            with torch.no_grad():
                out = model(**inputs, past_key_values=DynamicCache(),
                            use_cache=True, output_hidden_states=True, return_dict=True)
        except Exception as e:
            print(f"  prompt {used} prefill failed: {e}"); continue
        cur_kv = to_tuple_kv(out.past_key_values)
        h_t = out.hidden_states[-1][0, -1, :].float()
        v_logits = out.logits[0, -1, :].float()
        for pos in range(GEN_LEN):
            v_root = int(v_logits.argmax().item())
            try:
                with torch.no_grad():
                    tok = torch.tensor([[v_root]], device="cuda")
                    cur_len = cur_kv[0][0].shape[2]
                    pid = torch.tensor([[cur_len]], device="cuda")
                    cp = torch.tensor([cur_len], device="cuda")
                    out_step = model(input_ids=tok, past_key_values=kv_to_cache(cur_kv),
                                     position_ids=pid, cache_position=cp,
                                     use_cache=True, output_hidden_states=True, return_dict=True)
                    v_cont = int(out_step.logits[0, -1, :].argmax().item())
            except Exception: break
            h_t_list.append(h_t.detach().cpu())
            root_id_list.append(v_root); cont_target_list.append(v_cont)
            prompt_idx_list.append(used); pos_list.append(pos)
            h_t = out_step.hidden_states[-1][0, -1, :].float()
            cur_kv = to_tuple_kv(out_step.past_key_values)
            v_logits = out_step.logits[0, -1, :].float()
        used += 1
        if used % 50 == 0:
            print(f"  {used}/{N_COLLECT}, n={len(h_t_list)}, elapsed {(time.time()-t0)/60:.1f}min", flush=True)
        if used >= N_COLLECT: break

    h_t_tensor = torch.stack(h_t_list).to(torch.float32)
    root_id_t = torch.tensor(root_id_list, dtype=torch.long)
    cont_target_t = torch.tensor(cont_target_list, dtype=torch.long)
    prompt_idx_arr = np.array(prompt_idx_list); pos_arr = np.array(pos_list)
    print(f"  collected n={h_t_tensor.shape[0]}\n", flush=True)

    # ---------- 80/20 prompt-level split ----------
    rng = np.random.RandomState(0)
    unique_p = np.unique(prompt_idx_arr)
    n_eval = max(1, int(round(len(unique_p) * 0.20)))
    eval_p = set(rng.choice(unique_p, n_eval, replace=False).tolist())
    train_idx = np.array([i for i in range(len(prompt_idx_arr))
                          if prompt_idx_arr[i] not in eval_p])
    from collections import defaultdict
    train_by_p = defaultdict(list)
    for i in train_idx:
        train_by_p[prompt_idx_arr[i]].append(int(i))
    train_p_list = list(train_by_p.keys())

    # ---------- Phase 2: train V5e-0-Cont ----------
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    n1_cont = V5e0_Cont(dim=hidden_dim, alpha_init=cfg["alpha"]).to("cuda")
    opt = torch.optim.AdamW(n1_cont.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"[train V5e-0-Cont {EPOCHS} epochs]", flush=True)
    t0 = time.time()
    for ep in range(EPOCHS):
        n1_cont.train(); random.shuffle(train_p_list)
        for bstart in range(0, len(train_p_list), 8):
            bp = train_p_list[bstart:bstart + 8]
            opt.zero_grad()
            idxs = []
            for p in bp: idxs.extend(train_by_p[p])
            if not idxs: continue
            idxs_t = torch.tensor(idxs)
            h_b = h_t_tensor[idxs_t].cuda()
            r_b = root_id_t[idxs_t].cuda()
            c_b = cont_target_t[idxs_t].cuda()
            root_emb_b = root_emb_table[r_b]
            z = n1_cont(h_b, root_emb_b)
            logits = lm_head(z.to(torch.bfloat16)).float()
            loss = F.cross_entropy(logits, c_b)
            loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1}/{EPOCHS} loss={loss.item():.4f} t={(time.time()-t0)/60:.1f}min", flush=True)
    print(f"  Cont train done {(time.time()-t0)/60:.1f}min\n", flush=True)
    n1_cont.eval()

    # ---------- Phase 3: collect cont2 targets (need cont's true argmax) ----------
    # For each (prompt, pos) record we also need the verifier's cont2 = argmax after
    # the verifier-true cont. Reuse the same data: cont_target is verifier's argmax,
    # so cont2 is the next AR token. We already collected enough records: for each
    # prompt, pos i has cont_target = pos (i+1)'s root_id. Hence cont2_target_t[i]
    # for record i (prompt p, pos=k) equals root_id of record (p, k+1) if exists,
    # else fall back to cont_target.
    #
    # Build cont2 lookup
    rec_by_p_pos = {}
    for i in range(len(prompt_idx_arr)):
        rec_by_p_pos[(int(prompt_idx_arr[i]), int(pos_arr[i]))] = i
    cont2_target_list = []
    valid_idx_for_c2 = []
    for i in range(len(prompt_idx_arr)):
        key = (int(prompt_idx_arr[i]), int(pos_arr[i]) + 1)
        if key in rec_by_p_pos:
            cont2_target_list.append(root_id_t[rec_by_p_pos[key]].item())
            valid_idx_for_c2.append(i)
    cont2_target_t = torch.tensor(cont2_target_list, dtype=torch.long)
    valid_idx_arr = np.array(valid_idx_for_c2)
    print(f"[cont2 valid records: {len(valid_idx_for_c2)} / {len(prompt_idx_arr)}]", flush=True)

    train_idx_c2 = np.array([i for i, ridx in enumerate(valid_idx_for_c2)
                              if prompt_idx_arr[ridx] not in eval_p])
    train_by_p_c2 = defaultdict(list)
    for i in train_idx_c2:
        train_by_p_c2[prompt_idx_arr[valid_idx_for_c2[i]]].append(int(i))
    train_p_c2_list = list(train_by_p_c2.keys())

    # ---------- Phase 4: train V5e-0-Cont2 ----------
    random.seed(43); np.random.seed(43); torch.manual_seed(43)
    n1_cont2 = V5e0_Cont2(dim=hidden_dim,
                          alpha_init=cfg["alpha"], beta_init=cfg["beta"]).to("cuda")
    opt2 = torch.optim.AdamW(n1_cont2.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"[train V5e-0-Cont2 {EPOCHS} epochs]", flush=True)
    t0 = time.time()
    for ep in range(EPOCHS):
        n1_cont2.train(); random.shuffle(train_p_c2_list)
        for bstart in range(0, len(train_p_c2_list), 8):
            bp = train_p_c2_list[bstart:bstart + 8]
            opt2.zero_grad()
            local_idxs = []
            for p in bp: local_idxs.extend(train_by_p_c2[p])
            if not local_idxs: continue
            rec_idxs = np.array([valid_idx_for_c2[i] for i in local_idxs])
            h_b = h_t_tensor[rec_idxs].cuda()
            r_b = root_id_t[rec_idxs].cuda()
            c_b = cont_target_t[rec_idxs].cuda()
            c2_b = cont2_target_t[torch.tensor(local_idxs)].cuda()
            root_emb_b = root_emb_table[r_b]
            cont_emb_b = root_emb_table[c_b]
            z = n1_cont2(h_b, root_emb_b, cont_emb_b)
            logits = lm_head(z.to(torch.bfloat16)).float()
            loss = F.cross_entropy(logits, c2_b)
            loss.backward(); opt2.step()
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1}/{EPOCHS} loss={loss.item():.4f} t={(time.time()-t0)/60:.1f}min", flush=True)
    print(f"  Cont2 train done {(time.time()-t0)/60:.1f}min\n", flush=True)
    n1_cont2.eval()

    # ---------- Phase 5: walltime measurement ----------
    print(f"[walltime: {N_TEST} held-out prompts, {MAX_TOKENS} tokens]", flush=True)
    # Use prompts the eval split saw (held-out)
    eval_p_list = [p for p in eval_p]
    eval_prompts_text = [prompts[p] for p in eval_p_list if p < len(prompts)][:N_TEST]
    # If not enough eval prompts, supplement with fresh ones from a different seed
    if len(eval_prompts_text) < N_TEST:
        extra = load_text_prompts(
            "./data/llava_messages_100k.jsonl",
            N_TEST * 3, seed=777)
        for p in extra:
            if p not in eval_prompts_text:
                eval_prompts_text.append(p)
                if len(eval_prompts_text) >= N_TEST: break

    ar_tps, ssd_tps, n_d2, n_d3, n_rounds, n_tokens = measure_d3_llm(
        model, n1_cont, n1_cont2, lm_head, root_emb_table,
        tokenizer, eval_prompts_text[:N_TEST], M, K, MAX_TOKENS)

    ar_mean = float(np.mean(ar_tps)) if ar_tps else 0.0
    ssd_mean = float(np.mean(ssd_tps)) if ssd_tps else 0.0
    sp = ssd_mean / ar_mean if ar_mean > 0 else 0.0
    tpi = n_tokens / max(1, n_rounds)
    d2_rate = n_d2 / max(1, n_rounds)
    d3_rate = n_d3 / max(1, n_rounds)

    print(f"\n[RESULTS for {llm_name}]")
    print(f"  AR mean TPS:   {ar_mean:.2f} ± {np.std(ar_tps):.2f}  (n={len(ar_tps)})")
    print(f"  V5e-0 mean TPS: {ssd_mean:.2f} ± {np.std(ssd_tps):.2f}  (n={len(ssd_tps)})")
    print(f"  SPEEDUP:       {sp:.3f}x")
    print(f"  TPI:           {tpi:.3f}")
    print(f"  depth-2 rate:  {d2_rate:.3f}  ({n_d2} rounds)")
    print(f"  depth-3 rate:  {d3_rate:.3f}  ({n_d3} rounds)")
    print(f"  total rounds:  {n_rounds},  total tokens: {n_tokens}")
    print(f"  drafter params: Cont={sum(p.numel() for p in n1_cont.parameters())/1e6:.2f}M, "
          f"Cont2={sum(p.numel() for p in n1_cont2.parameters())/1e6:.2f}M")

    return {
        "llm": llm_name, "ar_tps": ar_mean, "ssd_tps": ssd_mean,
        "speedup": sp, "tpi": tpi, "d2_rate": d2_rate, "d3_rate": d3_rate,
        "n_rounds": n_rounds, "n_tokens": n_tokens,
        "hidden_dim": hidden_dim,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--llm", required=True, choices=list(LLM_CONFIGS.keys()))
    p.add_argument("--gpu", default="0")
    p.add_argument("--n_test", type=int, default=30)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--out", default=None, help="optional JSON path to write results")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    result = main(args.llm, n_test=args.n_test, max_tokens=args.max_tokens)
    if args.out:
        with open(args.out, "w") as f: json.dump(result, f, indent=2)
        print(f"\n[saved → {args.out}]")
