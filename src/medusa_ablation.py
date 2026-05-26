#!/usr/bin/env python3
"""Medusa-style (parallel) vs V5e-0 (sequential) Cont2 controlled ablation.

The ONLY difference: Cont2 input.
  V5e-0 (sequential):  Linear(h_t + α·E[root] + β·E[cont])    ← knows what +1 was
  Medusa-style (parallel): Linear(h_t + α·E[root])            ← independent of +1

Same target (+2 token), same data, same 100 epochs, same tree shape (M=5, K=3).

At inference both versions occupy M*K=15 cont2 slots in the tree; the difference
is that V5e-0's Cont2 produces different distributions per cont parent while the
Medusa-style version produces the SAME distribution regardless of parent
(only the root_emb varies, which is constant for all M parents).

Usage:
  python medusa_ablation.py --vlm qwen2-2b --gpu 0
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
    build_tree_d3, build_attn_mask_and_pos, prune_kv,
)


N_COLLECT = 200
GEN_LEN = 32
N_TEST = 30
MAX_TOKENS = 64
EPOCHS = 100
M = 5
K = 3


# ============================================================
# Medusa-style Cont2: NO conditioning on cont (parallel head)
# ============================================================
class V5e0_Cont2_Parallel(nn.Module):
    """Linear(h_t + α·E[root])  →  predicts +2 token marginally."""
    def __init__(self, dim, alpha_init=30.0):
        super().__init__()
        self.Q_proj = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.Q_proj.weight); nn.init.zeros_(self.Q_proj.bias)
        self.register_buffer("alpha", torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, h_t, root_emb, cont_emb=None):   # cont_emb ignored
        return self.Q_proj(h_t + self.alpha * root_emb)


# ============================================================
# Tree measurement with arbitrary Cont2 (sequential or parallel)
# ============================================================
def measure(model, n1_cont, n1_cont2, lm_head, root_emb_table,
            prepare_inputs, is_qwen, test_samples, M, K, max_tokens, label):
    from transformers.cache_utils import DynamicCache
    parents, depths = build_tree_d3(M, K)
    n_input = len(parents) - 1

    ar_tps_list = []; ssd_tps_list = []
    n_d2 = 0; n_d3 = 0; n_rounds_total = 0; n_tokens_total = 0

    for s_i, s in enumerate(test_samples):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

        # ---------- AR baseline ----------
        try:
            inputs = prepare_inputs(s)
            with torch.no_grad():
                out = model(**inputs, past_key_values=DynamicCache(),
                            use_cache=True, return_dict=True)
        except Exception:
            continue
        cur_kv = to_tuple_kv(out.past_key_values)
        last = int(out.logits[0, -1, :].argmax().item())
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            with torch.no_grad():
                for _ in range(max_tokens):
                    x = torch.tensor([[last]], device="cuda")
                    cur_len = cur_kv[0][0].shape[2]
                    if is_qwen:
                        pid = torch.tensor([[cur_len]], device="cuda").unsqueeze(0).expand(3, 1, -1)
                    else:
                        pid = torch.tensor([[cur_len]], device="cuda")
                    cp = torch.tensor([cur_len], device="cuda")
                    out_step = model(input_ids=x, past_key_values=kv_to_cache(cur_kv),
                                     position_ids=pid, cache_position=cp,
                                     use_cache=True, return_dict=True)
                    last = int(out_step.logits[0, -1, :].argmax().item())
                    cur_kv = to_tuple_kv(out_step.past_key_values)
        except Exception: continue
        torch.cuda.synchronize()
        ar_tps_list.append(max_tokens / (time.perf_counter() - t0))

        # ---------- SSD ----------
        try:
            inputs = prepare_inputs(s)
            with torch.no_grad():
                out = model(**inputs, past_key_values=DynamicCache(),
                            use_cache=True, output_hidden_states=True, return_dict=True)
        except Exception: continue
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

                    # Cont2: pass cont_embs but Parallel head ignores it
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
                        parents, depths, prompt_len, "cuda", model.dtype, qwen=is_qwen)
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
                    accepted_cont = None
                    for c_j in range(M):
                        pos = 1 + c_j
                        if tree_input_ids[pos] == cont_argmax:
                            accepted_cont = pos; break

                    is_first = (n_rounds == 0)
                    base = [anchor_argmax] if is_first else []

                    if accepted_cont is not None:
                        cont_pos = accepted_cont - 1
                        c2_start = 1 + M + cont_pos * K
                        cont2_argmax = int(tree_logits[accepted_cont].argmax().item())
                        accepted_c2 = None
                        for c2_j in range(K):
                            pos = c2_start + c2_j
                            if tree_input_ids[pos] == cont2_argmax:
                                accepted_c2 = pos; break
                        if accepted_c2 is not None:
                            bonus = int(tree_logits[accepted_c2].argmax().item())
                            new_toks = base + [cont_argmax, cont2_argmax, bonus]
                            acc_idx = [0, accepted_cont, accepted_c2]
                            new_last_logits = tree_logits[accepted_c2].float()
                            new_last_h_t = tree_hidden[accepted_c2].float()
                            n_d3 += 1
                        else:
                            new_toks = base + [cont_argmax, cont2_argmax]
                            acc_idx = [0, accepted_cont]
                            new_last_logits = tree_logits[accepted_cont].float()
                            new_last_h_t = tree_hidden[accepted_cont].float()
                            n_d2 += 1
                    else:
                        new_toks = base + [cont_argmax]
                        acc_idx = [0]
                        new_last_logits = tree_logits[0].float()
                        new_last_h_t = tree_hidden[0].float()

                    cur_kv = prune_kv(kv_after, prompt_len, acc_idx)
                    prompt_len += len(acc_idx)
                    last_logits = new_last_logits
                    last_h_t = new_last_h_t
                    generated.extend(new_toks)
                    n_rounds += 1
        except Exception as e:
            print(f"  [{label}] prompt {s_i} SSD failed: {e}", flush=True)
            continue
        torch.cuda.synchronize()
        ssd_tps_list.append(len(generated) / (time.perf_counter() - t0))
        n_rounds_total += n_rounds; n_tokens_total += len(generated)

    return ar_tps_list, ssd_tps_list, n_d2, n_d3, n_rounds_total, n_tokens_total


# ============================================================
# Eval cont2 accuracy on held-out data (independent of tree)
# ============================================================
def eval_cont2_top_k(cont2_model, lm_head, root_emb_table,
                      h_t_tensor, root_id_t, cont_target_t, cont2_target_t,
                      valid_idx_arr, eval_mask, ks=(1, 3, 5, 10)):
    cont2_model.eval()
    eval_local_idx = [i for i, ridx in enumerate(valid_idx_arr) if eval_mask[ridx]]
    if not eval_local_idx: return {}
    rec_idx = np.array([valid_idx_arr[i] for i in eval_local_idx])
    h_b = h_t_tensor[rec_idx].cuda()
    r_b = root_id_t[rec_idx].cuda()
    c_b = cont_target_t[rec_idx].cuda()
    c2_b = cont2_target_t[torch.tensor(eval_local_idx)].cuda()
    root_emb_b = root_emb_table[r_b]
    cont_emb_b = root_emb_table[c_b]
    with torch.no_grad():
        z = cont2_model(h_b, root_emb_b, cont_emb_b)
        logits = lm_head(z.to(torch.bfloat16)).float()
    out = {}
    for k in ks:
        topk = logits.topk(k, dim=-1).indices  # [N, k]
        out[k] = float((topk == c2_b.unsqueeze(-1)).any(dim=-1).float().mean().item())
    return out


def main(vlm_name, n_test, max_tokens):
    cfg = VLM_CONFIGS[vlm_name]
    print(f"[medusa ablation: {vlm_name}, M={M}, K={K}]\n", flush=True)

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
    print(f"[collect {N_COLLECT} prompts × {GEN_LEN}]", flush=True)
    samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        N_COLLECT, seed=43)
    h_t_list, root_id_list, cont_target_list = [], [], []
    prompt_idx_list, pos_list = [], []
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
    prompt_idx_arr = np.array(prompt_idx_list); pos_arr = np.array(pos_list)
    print(f"  collected n={h_t_tensor.shape[0]}\n", flush=True)

    # Split
    rng = np.random.RandomState(0)
    unique_p = np.unique(prompt_idx_arr)
    n_eval = max(1, int(round(len(unique_p) * 0.20)))
    eval_p = set(rng.choice(unique_p, n_eval, replace=False).tolist())
    eval_mask = np.array([prompt_idx_arr[i] in eval_p for i in range(len(prompt_idx_arr))])
    train_idx = np.where(~eval_mask)[0]
    from collections import defaultdict
    train_by_p = defaultdict(list)
    for i in train_idx: train_by_p[prompt_idx_arr[i]].append(int(i))
    train_p_list = list(train_by_p.keys())

    # Cont2 target lookup
    rec_by_p_pos = {}
    for i in range(len(prompt_idx_arr)):
        rec_by_p_pos[(int(prompt_idx_arr[i]), int(pos_arr[i]))] = i
    cont2_target_list, valid_idx_for_c2 = [], []
    for i in range(len(prompt_idx_arr)):
        key = (int(prompt_idx_arr[i]), int(pos_arr[i]) + 1)
        if key in rec_by_p_pos:
            cont2_target_list.append(root_id_t[rec_by_p_pos[key]].item())
            valid_idx_for_c2.append(i)
    cont2_target_t = torch.tensor(cont2_target_list, dtype=torch.long)
    valid_idx_arr = np.array(valid_idx_for_c2)

    train_idx_c2 = np.array([i for i, ridx in enumerate(valid_idx_for_c2)
                              if not eval_mask[ridx]])
    train_by_p_c2 = defaultdict(list)
    for i in train_idx_c2:
        train_by_p_c2[prompt_idx_arr[valid_idx_for_c2[i]]].append(int(i))
    train_p_c2_list = list(train_by_p_c2.keys())

    # ---- Train Cont (shared by both) ----
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    n1_cont = V5e0_Cont(dim=D, alpha_init=cfg["alpha"]).to("cuda")
    opt = torch.optim.AdamW(n1_cont.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"[train Cont {EPOCHS} epochs]", flush=True)
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
            loss = F.cross_entropy(lm_head(z.to(torch.bfloat16)).float(), c_b)
            loss.backward(); opt.step()
    n1_cont.eval()
    print(f"  Cont done\n", flush=True)

    # ---- Train BOTH Cont2 variants on identical data ----
    def train_cont2(model_cls, name, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        m = model_cls(dim=D, alpha_init=cfg["alpha"]).to("cuda") if 'beta_init' not in model_cls.__init__.__code__.co_varnames \
            else model_cls(dim=D, alpha_init=cfg["alpha"], beta_init=cfg["beta"]).to("cuda")
        opt2 = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.01)
        print(f"[train {name} {EPOCHS} epochs]", flush=True)
        for ep in range(EPOCHS):
            m.train(); random.shuffle(train_p_c2_list)
            for bstart in range(0, len(train_p_c2_list), 8):
                bp = train_p_c2_list[bstart:bstart + 8]
                opt2.zero_grad()
                lidx = []
                for p in bp: lidx.extend(train_by_p_c2[p])
                if not lidx: continue
                rec_idxs = np.array([valid_idx_for_c2[i] for i in lidx])
                h_b = h_t_tensor[rec_idxs].cuda()
                r_b = root_id_t[rec_idxs].cuda()
                c_b = cont_target_t[rec_idxs].cuda()
                c2_b = cont2_target_t[torch.tensor(lidx)].cuda()
                root_emb_b = root_emb_table[r_b]
                cont_emb_b = root_emb_table[c_b]
                z = m(h_b, root_emb_b, cont_emb_b)
                loss = F.cross_entropy(lm_head(z.to(torch.bfloat16)).float(), c2_b)
                loss.backward(); opt2.step()
        m.eval()
        print(f"  {name} done", flush=True)
        return m

    n1_seq      = train_cont2(V5e0_Cont2,           "V5e-0 Cont2 (sequential)", 43)
    n1_parallel = train_cont2(V5e0_Cont2_Parallel,  "Medusa Cont2 (parallel)",  43)
    print()

    # ---- Eval cont2 accuracy on held-out (paired) ----
    seq_acc = eval_cont2_top_k(n1_seq, lm_head, root_emb_table,
                                 h_t_tensor, root_id_t, cont_target_t, cont2_target_t,
                                 valid_idx_arr, eval_mask)
    par_acc = eval_cont2_top_k(n1_parallel, lm_head, root_emb_table,
                                 h_t_tensor, root_id_t, cont_target_t, cont2_target_t,
                                 valid_idx_arr, eval_mask)
    print(f"[held-out cont2 top-K accuracy]")
    print(f"  V5e-0 sequential: top1={seq_acc[1]:.3f}, top3={seq_acc[3]:.3f}, top5={seq_acc[5]:.3f}, top10={seq_acc[10]:.3f}")
    print(f"  Medusa parallel : top1={par_acc[1]:.3f}, top3={par_acc[3]:.3f}, top5={par_acc[5]:.3f}, top10={par_acc[10]:.3f}")
    print()

    # ---- Walltime measurement (both, on same test prompts) ----
    eval_p_list = sorted(eval_p)
    test_samples = [samples[p] for p in eval_p_list if p < len(samples)][:n_test]
    if len(test_samples) < n_test:
        extra = load_llava_prompts(
            "./data/llava_messages_100k.jsonl",
            n_test * 3, seed=777)
        for s in extra:
            if all(s["image"] != t["image"] for t in test_samples):
                test_samples.append(s)
                if len(test_samples) >= n_test: break

    print(f"[walltime: {len(test_samples)} held-out prompts, {max_tokens} tokens]\n", flush=True)
    print(f"\n--- V5e-0 (Sequential Cont2) ---")
    seq_res = measure(model, n1_cont, n1_seq, lm_head, root_emb_table,
                      prepare_inputs, is_qwen, test_samples, M, K, max_tokens, "seq")
    print(f"\n--- Medusa-style (Parallel Cont2) ---")
    par_res = measure(model, n1_cont, n1_parallel, lm_head, root_emb_table,
                      prepare_inputs, is_qwen, test_samples, M, K, max_tokens, "par")

    def summarize(name, res):
        ar, ssd, nd2, nd3, nr, nt = res
        ar_m = float(np.mean(ar)) if ar else 0.0
        ssd_m = float(np.mean(ssd)) if ssd else 0.0
        sp = ssd_m / ar_m if ar_m > 0 else 0.0
        tpi = nt / max(1, nr)
        return {"label": name, "ar_tps": ar_m, "ssd_tps": ssd_m, "speedup": sp,
                "tpi": tpi, "d2_rate": nd2 / max(1, nr), "d3_rate": nd3 / max(1, nr),
                "n_rounds": nr, "n_tokens": nt,
                "cont2_top1": (seq_acc[1] if name == "V5e-0 seq" else par_acc[1]),
                "cont2_top3": (seq_acc[3] if name == "V5e-0 seq" else par_acc[3]),
                "cont2_top5": (seq_acc[5] if name == "V5e-0 seq" else par_acc[5])}

    seq_sum = summarize("V5e-0 seq", seq_res)
    par_sum = summarize("Medusa parallel", par_res)

    print("\n" + "=" * 72)
    print(f"  RESULTS ({vlm_name})")
    print("=" * 72)
    print(f"{'Method':<24}{'sp':>8}{'TPI':>8}{'d2':>8}{'d3':>8}{'c2 top1':>10}{'c2 top3':>10}")
    for r in (seq_sum, par_sum):
        print(f"{r['label']:<24}{r['speedup']:>8.3f}{r['tpi']:>8.3f}"
              f"{r['d2_rate']:>8.3f}{r['d3_rate']:>8.3f}"
              f"{r['cont2_top1']:>10.3f}{r['cont2_top3']:>10.3f}")
    print("=" * 72)
    return {"vlm": vlm_name, "v5e0_sequential": seq_sum, "medusa_parallel": par_sum}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", required=True, choices=list(VLM_CONFIGS.keys()))
    p.add_argument("--gpu", default="0")
    p.add_argument("--n_test", type=int, default=30)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    result = main(args.vlm, args.n_test, args.max_tokens)
    if args.out:
        with open(args.out, "w") as f: json.dump(result, f, indent=2)
        print(f"\n[saved → {args.out}]")
