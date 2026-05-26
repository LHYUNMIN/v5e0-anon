#!/usr/bin/env python3
"""Depth-4 tree ablation: adds V5e-0-Cont3 (third Linear head) for 3-step lookahead.

Tree: 1 anchor (KV) + 1 root + M conts + M*K cont2's + M*K*L cont3's
Verify chain: root → cont → cont2 → cont3 → bonus
Output per round (best case): 4 tokens (depth-4 fully accepted) + bonus = 5 in round 0

Reuses existing V5e-0-Cont, V5e-0-Cont2; trains additional V5e-0-Cont3 head.

Usage:
  python depth4_ablation.py --vlm qwen2-2b --gpu 0 --m 3 --k 2 --l 2
"""
import os, sys, json, gc, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (
    VLM_CONFIGS, V5e0_Cont, V5e0_Cont2,
    to_tuple_kv, kv_to_cache, load_llava_prompts,
    prepare_qwen_inputs, prepare_llava_inputs,
)


N_COLLECT = 200
GEN_LEN = 32
N_TEST = 30
MAX_TOKENS = 64
EPOCHS = 100


# ============================================================
# V5e-0-Cont3: 3-step lookahead head (4 inputs)
# ============================================================
class V5e0_Cont3(nn.Module):
    """z3 = W_Q3(h_t + α·E[r] + β·E[c] + γ·E[c2])  → cont3 logits via lm_head."""
    def __init__(self, dim, alpha=30.0, beta=30.0, gamma=30.0):
        super().__init__()
        self.Q_proj = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.Q_proj.weight); nn.init.zeros_(self.Q_proj.bias)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("beta",  torch.tensor(beta,  dtype=torch.float32))
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))
    def forward(self, h_t, root_emb, cont_emb, cont2_emb):
        q = h_t + self.alpha * root_emb + self.beta * cont_emb + self.gamma * cont2_emb
        return self.Q_proj(q)


# ============================================================
# Depth-4 tree builder
# ============================================================
def build_tree_d4(M, K, L):
    """1 anchor + 1 root + M conts + M*K cont2's + M*K*L cont3's.
    Returns: parents, depths, cont_idx_array, cont2_idx_array.
    """
    parents = [-1, 0]; depths = [0, 1]
    # M conts at depth 2 (parent = root at idx 1)
    cont_indices = []
    for _ in range(M):
        cont_indices.append(len(parents))
        parents.append(1); depths.append(2)
    # M*K cont2's at depth 3 (parent = cont_i)
    cont2_indices = []   # cont2_indices[j] = list of K cont2 array indices under cont j
    for j in range(M):
        cont_idx = cont_indices[j]
        sublist = []
        for _ in range(K):
            sublist.append(len(parents))
            parents.append(cont_idx); depths.append(3)
        cont2_indices.append(sublist)
    # M*K*L cont3's at depth 4 (parent = cont2_jk)
    cont3_indices = []   # cont3_indices[j][k] = list of L cont3 array indices
    for j in range(M):
        cont3_indices.append([])
        for k in range(K):
            cont2_idx = cont2_indices[j][k]
            sublist = []
            for _ in range(L):
                sublist.append(len(parents))
                parents.append(cont2_idx); depths.append(4)
            cont3_indices[j].append(sublist)
    return parents, depths, cont_indices, cont2_indices, cont3_indices


def build_attn_mask_and_pos(parents, depths, prompt_len, device, dtype, qwen=False):
    n_input = len(parents) - 1
    mask = torch.full((n_input, prompt_len + n_input), float('-inf'), device=device, dtype=dtype)
    mask[:, :prompt_len] = 0.0
    for i in range(1, len(parents)):
        in_idx = i - 1
        mask[in_idx, prompt_len + in_idx] = 0.0
        cur = parents[i]
        while cur != -1 and cur != 0:
            mask[in_idx, prompt_len + cur - 1] = 0.0
            cur = parents[cur]
    pos_ids = torch.tensor(
        [prompt_len + depths[i] - 1 for i in range(1, len(parents))],
        device=device, dtype=torch.long)
    if qwen:
        pos_ids = pos_ids.unsqueeze(0).unsqueeze(0).expand(3, 1, -1).contiguous()
    else:
        pos_ids = pos_ids.unsqueeze(0)
    return mask.unsqueeze(0).unsqueeze(0), pos_ids


def prune_kv(kv_tuple, prompt_len, accepted_input_indices):
    if not accepted_input_indices:
        return tuple((k[..., :prompt_len, :].contiguous(),
                      v[..., :prompt_len, :].contiguous()) for k, v in kv_tuple)
    idx = torch.tensor([prompt_len + i for i in accepted_input_indices],
                        device=kv_tuple[0][0].device, dtype=torch.long)
    new = []
    for k, v in kv_tuple:
        pk, pv = k[..., :prompt_len, :], v[..., :prompt_len, :]
        sk, sv = k.index_select(-2, idx), v.index_select(-2, idx)
        new.append((torch.cat([pk, sk], dim=-2).contiguous(),
                    torch.cat([pv, sv], dim=-2).contiguous()))
    return tuple(new)


# ============================================================
# Depth-4 inference loop
# ============================================================
@torch.no_grad()
def measure_depth4(model, n1_cont, n1_cont2, n1_cont3, lm_head, root_emb_table,
                    prepare_inputs, is_qwen, test_samples, M, K, L, max_tokens):
    from transformers.cache_utils import DynamicCache
    parents, depths, cont_idx_arr, cont2_idx_arr, cont3_idx_arr = build_tree_d4(M, K, L)
    n_input = len(parents) - 1

    ar_tps_list = []; ssd_tps_list = []
    n_d2 = 0; n_d3 = 0; n_d4 = 0; n_rounds_total = 0; n_tokens_total = 0

    for s_i, s in enumerate(test_samples):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

        # AR baseline
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
                if is_qwen:
                    pid = torch.tensor([[cur]], device="cuda").unsqueeze(0).expand(3, 1, -1)
                else:
                    pid = torch.tensor([[cur]], device="cuda")
                cp = torch.tensor([cur], device="cuda")
                out = model(input_ids=x, past_key_values=kv_to_cache(kv),
                            position_ids=pid, cache_position=cp, use_cache=True, return_dict=True)
                last = int(out.logits[0, -1, :].argmax().item())
                kv = to_tuple_kv(out.past_key_values)
        except Exception: continue
        torch.cuda.synchronize()
        ar_tps_list.append(max_tokens / (time.perf_counter() - t0))

        # SSD depth-4
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
                anchor = int(ll.argmax().item())
                root_emb = root_emb_table[anchor].unsqueeze(0)

                # Cont (top-M)
                z = n1_cont(h_t.unsqueeze(0), root_emb)
                conts = lm_head(z.to(torch.bfloat16)).float().topk(M, dim=-1).indices[0]

                # Cont2 (top-K per cont, batched over M)
                h_b = h_t.unsqueeze(0).expand(M, -1)
                root_b = root_emb.expand(M, -1)
                cont_embs = root_emb_table[conts]
                z2 = n1_cont2(h_b, root_b, cont_embs)
                cont2s = lm_head(z2.to(torch.bfloat16)).float().topk(K, dim=-1).indices  # [M, K]

                # Cont3 (top-L per cont2, batched over M*K)
                MK = M * K
                h_mk = h_t.unsqueeze(0).expand(MK, -1)
                root_mk = root_emb.expand(MK, -1)
                cont_mk = root_emb_table[conts.unsqueeze(1).expand(-1, K).reshape(-1)]  # [MK, D]
                cont2_mk = root_emb_table[cont2s.reshape(-1)]                            # [MK, D]
                z3 = n1_cont3(h_mk, root_mk, cont_mk, cont2_mk)
                cont3s = lm_head(z3.to(torch.bfloat16)).float().topk(L, dim=-1).indices  # [MK, L]
                cont3s = cont3s.reshape(M, K, L)

                # Build tree input ids (ORDER MUST MATCH parents array)
                tree_ids = [anchor] + conts.tolist()
                for j in range(M): tree_ids.extend(cont2s[j].tolist())
                for j in range(M):
                    for k in range(K):
                        tree_ids.extend(cont3s[j][k].tolist())
                x = torch.tensor([tree_ids], device="cuda")
                mask, pos_ids = build_attn_mask_and_pos(parents, depths, prompt_len,
                                                          "cuda", model.dtype, qwen=is_qwen)
                cp = torch.arange(prompt_len, prompt_len + n_input, device="cuda")
                out = model(input_ids=x, past_key_values=kv_to_cache(kv),
                            attention_mask=mask, position_ids=pos_ids, cache_position=cp,
                            use_cache=True, output_hidden_states=True, return_dict=True)
                tree_logits = out.logits[0]
                tree_hidden = out.hidden_states[-1][0]
                kv_after = to_tuple_kv(out.past_key_values)

                # Verify chain: cont → cont2 → cont3
                cont_argmax = int(tree_logits[0].argmax().item())
                acc_cont = None; cont_j = -1
                for j in range(M):
                    pos_in = cont_idx_arr[j] - 1   # input position (exclude anchor)
                    if tree_ids[pos_in] == cont_argmax:
                        acc_cont = pos_in; cont_j = j; break

                is_first = (n_rounds == 0)
                base = [anchor] if is_first else []

                if acc_cont is None:
                    # depth-1: cont not matched, take verifier's argmax
                    new_toks = base + [cont_argmax]
                    acc_idx = [0]
                    new_h = tree_hidden[0].float()
                    new_ll = tree_logits[0].float()
                else:
                    n_d2 += 1
                    cont2_argmax = int(tree_logits[acc_cont].argmax().item())
                    acc_cont2 = None; cont2_k = -1
                    for k in range(K):
                        pos_in = cont2_idx_arr[cont_j][k] - 1
                        if tree_ids[pos_in] == cont2_argmax:
                            acc_cont2 = pos_in; cont2_k = k; break
                    if acc_cont2 is None:
                        # depth-2: stop here, cont2_argmax is bonus
                        new_toks = base + [cont_argmax, cont2_argmax]
                        acc_idx = [0, acc_cont]
                        new_h = tree_hidden[acc_cont].float()
                        new_ll = tree_logits[acc_cont].float()
                    else:
                        n_d3 += 1
                        cont3_argmax = int(tree_logits[acc_cont2].argmax().item())
                        acc_cont3 = None
                        for ll_l in range(L):
                            pos_in = cont3_idx_arr[cont_j][cont2_k][ll_l] - 1
                            if tree_ids[pos_in] == cont3_argmax:
                                acc_cont3 = pos_in; break
                        if acc_cont3 is None:
                            new_toks = base + [cont_argmax, cont2_argmax, cont3_argmax]
                            acc_idx = [0, acc_cont, acc_cont2]
                            new_h = tree_hidden[acc_cont2].float()
                            new_ll = tree_logits[acc_cont2].float()
                        else:
                            n_d4 += 1
                            bonus = int(tree_logits[acc_cont3].argmax().item())
                            new_toks = base + [cont_argmax, cont2_argmax, cont3_argmax, bonus]
                            acc_idx = [0, acc_cont, acc_cont2, acc_cont3]
                            new_h = tree_hidden[acc_cont3].float()
                            new_ll = tree_logits[acc_cont3].float()

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

    return ar_tps_list, ssd_tps_list, n_d2, n_d3, n_d4, n_rounds_total, n_tokens_total


# ============================================================
# Main: data collect + train Cont/Cont2/Cont3 + measure
# ============================================================
def main(vlm_name, M, K, L, n_test):
    cfg = VLM_CONFIGS[vlm_name]
    print(f"[depth-4 ablation: {vlm_name}, M={M}, K={K}, L={L}]\n", flush=True)

    sys.path.insert(0, "./vendor")
    from runtime_env import strip_user_site_packages; strip_user_site_packages()
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
    print(f"  hidden_dim={D}\n", flush=True)

    # ---- Data collection ----
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
        except Exception: continue
        cur_kv = to_tuple_kv(out.past_key_values)
        h_t = out.hidden_states[-1][0, -1, :].float()
        v_logits = out.logits[0, -1, :].float()
        for pos in range(GEN_LEN):
            v_root = int(v_logits.argmax().item())
            try:
                with torch.no_grad():
                    tok = torch.tensor([[v_root]], device="cuda")
                    cur_len = cur_kv[0][0].shape[2]
                    if is_qwen:
                        pid = torch.tensor([[cur_len]], device="cuda").unsqueeze(0).expand(3, 1, -1)
                    else:
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
        if used >= N_COLLECT: break

    h_t_tensor = torch.stack(h_t_list).to(torch.float32)
    root_id_t = torch.tensor(root_id_list, dtype=torch.long)
    cont_target_t = torch.tensor(cont_target_list, dtype=torch.long)
    prompt_idx_arr = np.array(prompt_idx_list)
    pos_arr = np.array(pos_list)
    print(f"  collected n={h_t_tensor.shape[0]}\n")

    # ---- Build cont3 training tuples: (h_idx, root, cont, cont2, cont3_target) ----
    # cont3_target = next-next-next token (at position pos+3 within prompt)
    print("[construct training tuples]")
    cont1_recs = []   # for Cont head (just (h, root, cont))
    cont2_recs = []   # for Cont2 head (h, root, cont, cont2)
    cont3_recs = []   # for Cont3 head (h, root, cont, cont2, cont3)
    for i in range(len(prompt_idx_arr)):
        cont1_recs.append({"h_idx": i, "root_id": int(root_id_t[i]),
                            "cont_target": int(cont_target_t[i]),
                            "prompt_idx": int(prompt_idx_arr[i])})
        if i + 1 < len(prompt_idx_arr) and prompt_idx_arr[i+1] == prompt_idx_arr[i] \
                and pos_arr[i+1] == pos_arr[i] + 1:
            cont2_recs.append({"h_idx": i, "root_id": int(root_id_t[i]),
                                "cont_id": int(cont_target_t[i]),
                                "cont2_target": int(cont_target_t[i+1]),
                                "prompt_idx": int(prompt_idx_arr[i])})
            if i + 2 < len(prompt_idx_arr) and prompt_idx_arr[i+2] == prompt_idx_arr[i] \
                    and pos_arr[i+2] == pos_arr[i] + 2:
                cont3_recs.append({"h_idx": i, "root_id": int(root_id_t[i]),
                                    "cont_id": int(cont_target_t[i]),
                                    "cont2_id": int(cont_target_t[i+1]),
                                    "cont3_target": int(cont_target_t[i+2]),
                                    "prompt_idx": int(prompt_idx_arr[i])})
    rng = np.random.RandomState(0)
    unique = np.unique(prompt_idx_arr)
    n_eval_p = max(1, int(round(len(unique) * 0.20)))
    eval_prompts = set(rng.choice(unique, n_eval_p, replace=False).tolist())
    print(f"  Cont: {len(cont1_recs)}, Cont2: {len(cont2_recs)}, Cont3: {len(cont3_recs)}\n")

    # ---- Train Cont ----
    from collections import defaultdict
    train_by_p = defaultdict(list)
    for r in cont1_recs:
        if r["prompt_idx"] not in eval_prompts: train_by_p[r["prompt_idx"]].append(r)
    tp = list(train_by_p.keys())
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    n1_cont = V5e0_Cont(dim=D, alpha_init=cfg["alpha"]).to("cuda")
    opt = torch.optim.AdamW(n1_cont.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"[train Cont {EPOCHS} epochs]")
    t0 = time.time()
    for ep in range(EPOCHS):
        n1_cont.train(); random.shuffle(tp)
        for bs in range(0, len(tp), 8):
            opt.zero_grad(); total_loss = 0; total_n = 0
            for p in tp[bs:bs+8]:
                recs = train_by_p[p]
                h = torch.stack([h_t_tensor[r["h_idx"]] for r in recs]).to("cuda")
                re = root_emb_table[torch.tensor([r["root_id"] for r in recs])].to("cuda")
                tg = torch.tensor([r["cont_target"] for r in recs], device="cuda")
                z = n1_cont(h, re)
                logits = lm_head(z.to(torch.bfloat16)).float()
                total_loss = total_loss + F.cross_entropy(logits, tg, reduction="sum"); total_n += len(recs)
            if total_n > 0: (total_loss / total_n).backward(); opt.step()
    n1_cont.eval()
    for p in n1_cont.parameters(): p.requires_grad_(False)
    print(f"  trained in {(time.time()-t0)/60:.1f}min\n")

    # ---- Train Cont2 ----
    train_by_p2 = defaultdict(list)
    for r in cont2_recs:
        if r["prompt_idx"] not in eval_prompts: train_by_p2[r["prompt_idx"]].append(r)
    tp2 = list(train_by_p2.keys())
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    n1_cont2 = V5e0_Cont2(dim=D, alpha_init=cfg["alpha"], beta_init=cfg["beta"]).to("cuda")
    opt2 = torch.optim.AdamW(n1_cont2.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"[train Cont2 {EPOCHS} epochs]")
    t0 = time.time()
    for ep in range(EPOCHS):
        n1_cont2.train(); random.shuffle(tp2)
        for bs in range(0, len(tp2), 8):
            opt2.zero_grad(); total_loss = 0; total_n = 0
            for p in tp2[bs:bs+8]:
                recs = train_by_p2[p]
                h = torch.stack([h_t_tensor[r["h_idx"]] for r in recs]).to("cuda")
                re = root_emb_table[torch.tensor([r["root_id"] for r in recs])].to("cuda")
                ce = root_emb_table[torch.tensor([r["cont_id"] for r in recs])].to("cuda")
                tg = torch.tensor([r["cont2_target"] for r in recs], device="cuda")
                z = n1_cont2(h, re, ce)
                logits = lm_head(z.to(torch.bfloat16)).float()
                total_loss = total_loss + F.cross_entropy(logits, tg, reduction="sum"); total_n += len(recs)
            if total_n > 0: (total_loss / total_n).backward(); opt2.step()
    n1_cont2.eval()
    for p in n1_cont2.parameters(): p.requires_grad_(False)
    print(f"  trained in {(time.time()-t0)/60:.1f}min\n")

    # ---- Train Cont3 ----
    train_by_p3 = defaultdict(list)
    for r in cont3_recs:
        if r["prompt_idx"] not in eval_prompts: train_by_p3[r["prompt_idx"]].append(r)
    tp3 = list(train_by_p3.keys())
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    n1_cont3 = V5e0_Cont3(dim=D, alpha=cfg["alpha"], beta=cfg["beta"], gamma=30.0).to("cuda")
    opt3 = torch.optim.AdamW(n1_cont3.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"[train Cont3 {EPOCHS} epochs]")
    t0 = time.time()
    for ep in range(EPOCHS):
        n1_cont3.train(); random.shuffle(tp3)
        for bs in range(0, len(tp3), 8):
            opt3.zero_grad(); total_loss = 0; total_n = 0
            for p in tp3[bs:bs+8]:
                recs = train_by_p3[p]
                h = torch.stack([h_t_tensor[r["h_idx"]] for r in recs]).to("cuda")
                re = root_emb_table[torch.tensor([r["root_id"] for r in recs])].to("cuda")
                ce = root_emb_table[torch.tensor([r["cont_id"] for r in recs])].to("cuda")
                c2e = root_emb_table[torch.tensor([r["cont2_id"] for r in recs])].to("cuda")
                tg = torch.tensor([r["cont3_target"] for r in recs], device="cuda")
                z = n1_cont3(h, re, ce, c2e)
                logits = lm_head(z.to(torch.bfloat16)).float()
                total_loss = total_loss + F.cross_entropy(logits, tg, reduction="sum"); total_n += len(recs)
            if total_n > 0: (total_loss / total_n).backward(); opt3.step()
    n1_cont3.eval()
    for p in n1_cont3.parameters(): p.requires_grad_(False)
    print(f"  trained in {(time.time()-t0)/60:.1f}min\n")

    # Eval Cont3 top-K
    eval_recs = [r for r in cont3_recs if r["prompt_idx"] in eval_prompts]
    if len(eval_recs) > 0:
        with torch.no_grad():
            h = torch.stack([h_t_tensor[r["h_idx"]] for r in eval_recs]).to("cuda")
            re = root_emb_table[torch.tensor([r["root_id"] for r in eval_recs])].to("cuda")
            ce = root_emb_table[torch.tensor([r["cont_id"] for r in eval_recs])].to("cuda")
            c2e = root_emb_table[torch.tensor([r["cont2_id"] for r in eval_recs])].to("cuda")
            tg = torch.tensor([r["cont3_target"] for r in eval_recs], device="cuda")
            z = n1_cont3(h, re, ce, c2e)
            logits = lm_head(z.to(torch.bfloat16)).float()
            top10 = logits.topk(10, -1).indices
            cont3_topk = {k: float((top10[:, :k] == tg.unsqueeze(-1)).any(-1).float().mean())
                          for k in [1, 3, 5, 10]}
        print(f"Cont3 eval top-K: {cont3_topk}\n")
    else:
        cont3_topk = {}

    # ---- Measure depth-4 sp ----
    test_samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        n_test, seed=999)
    print(f"[walltime test depth-4 (M={M}, K={K}, L={L}, tree={1+M+M*K+M*K*L} tokens)]")
    ar_list, ssd_list, n_d2, n_d3, n_d4, n_rounds, n_tokens = measure_depth4(
        model, n1_cont, n1_cont2, n1_cont3, lm_head, root_emb_table, prepare_inputs,
        is_qwen, test_samples, M, K, L, MAX_TOKENS)
    sp = [s/a for a, s in zip(ar_list, ssd_list)]
    ar_a = np.array(ar_list); ssd_a = np.array(ssd_list); sp_a = np.array(sp)
    tpi = n_tokens / n_rounds if n_rounds > 0 else 0
    d2_rate = n_d2/n_rounds; d3_rate = n_d3/n_rounds; d4_rate = n_d4/n_rounds

    print(f"\n{'='*70}\n{vlm_name} depth-4 (M={M}, K={K}, L={L}, {1+M+M*K+M*K*L} input tok)\n{'='*70}")
    print(f"  AR:    {ar_a.mean():.2f} ± {ar_a.std():.2f} t/s")
    print(f"  SSD:   {ssd_a.mean():.2f} ± {ssd_a.std():.2f} t/s")
    print(f"  sp:    {sp_a.mean():.3f} ± {sp_a.std():.3f}")
    print(f"  TPI:   {tpi:.2f}, d2: {d2_rate:.3f}, d3: {d3_rate:.3f}, d4: {d4_rate:.3f}")

    out = {
        "vlm": vlm_name, "depth": 4, "M": M, "K": K, "L": L,
        "n_input_tokens": 1 + M + M*K + M*K*L,
        "cont3_eval_topk": cont3_topk,
        "ar_tps": {"mean": float(ar_a.mean()), "std": float(ar_a.std())},
        "ssd_tps": {"mean": float(ssd_a.mean()), "std": float(ssd_a.std())},
        "sp": {"mean": float(sp_a.mean()), "std": float(sp_a.std()),
               "median": float(np.median(sp_a))},
        "tpi": tpi, "d2_rate": d2_rate, "d3_rate": d3_rate, "d4_rate": d4_rate,
        "n_prompts": len(ar_list),
        "raw": {"ar": ar_list, "ssd": ssd_list, "sp": sp},
    }
    save_path = f"./results/depth4_{vlm_name}_M{M}_K{K}_L{L}.json"
    with open(save_path, "w") as f: json.dump(out, f, indent=2)
    print(f"\n[saved {save_path}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", required=True, choices=list(VLM_CONFIGS.keys()))
    p.add_argument("--gpu", default="0")
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--l", type=int, default=2)
    p.add_argument("--n_test", type=int, default=30)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args.vlm, args.m, args.k, args.l, args.n_test)
