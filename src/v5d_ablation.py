#!/usr/bin/env python3
"""V5d cross-VLM ablation: cross-attention over vision tokens vs V5e-0.

V5d head:
  z = CrossAttn(W_Q h_t, W_K V_vision, W_V V_vision; T)
  where V_vision is extracted from verifier's hidden states at image-token positions,
  and T is a learnable temperature.

  cont_logits = lm_head(z)

This is the "vision-aware drafter" that ViSpec/DREAM/MASSV use in spirit. We show
it underperforms V5e-0 (which uses h_t directly, no vision attention).

Variants implemented:
  V5d   — Pure vision attention (no h_t direct), as the cleanest "vision-aware" baseline
  V5e-1 — Vision only (Q = learnable param, no h_t)
  V5e-2 — h_t + mean-pool(V_vision), additive

Usage:
  python v5d_ablation.py --vlm llava-1.5-7b --variant v5d --gpu 0
"""
import os, sys, json, gc, time, random, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (
    VLM_CONFIGS, V5e0_Cont,
    to_tuple_kv, kv_to_cache, load_llava_prompts,
    prepare_qwen_inputs, prepare_llava_inputs,
)


# ============================================================
# V5d head: cross-attention with learnable temperature
# ============================================================
class V5d_Cont(nn.Module):
    """Cross-attention over vision tokens.
       z = CrossAttn(Q=W_Q h_t, K=W_K V_vis, V=W_V V_vis; T)
       cont_logits = lm_head(z)
    """
    def __init__(self, dim, T_init=30.0):
        super().__init__()
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.W_Q.weight)
        nn.init.eye_(self.W_K.weight)
        nn.init.eye_(self.W_V.weight)
        self.T = nn.Parameter(torch.tensor(T_init, dtype=torch.float32))
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, h_t, V_vis):
        """h_t: [B, D]  V_vis: [B, n_vis, D]"""
        Q = self.W_Q(h_t)                          # [B, D]
        K = self.W_K(V_vis)                        # [B, n_vis, D]
        V = self.W_V(V_vis)                        # [B, n_vis, D]
        attn = torch.einsum('bd,bnd->bn', Q, K) * self.scale / self.T   # [B, n_vis]
        attn = F.softmax(attn, dim=-1)
        z = torch.einsum('bn,bnd->bd', attn, V)    # [B, D]
        return z


class V5e2_Cont(nn.Module):
    """V5e-2: h_t + alpha·E[r] + gamma·mean_pool(V_vis)."""
    def __init__(self, dim, alpha=30.0, gamma_init=1.0):
        super().__init__()
        self.W_Q = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.W_Q.weight); nn.init.zeros_(self.W_Q.bias)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, h_t, root_emb, V_vis):
        vis_pool = V_vis.mean(dim=1)               # [B, D]
        return self.W_Q(h_t + self.alpha * root_emb + self.gamma * vis_pool)


# ============================================================
# Extract V_vision from verifier prefix
# ============================================================
def find_image_token_positions(input_ids, image_token_id):
    """Find positions in input_ids where token == image_token_id (per-sample)."""
    # input_ids: [B, seq]
    matches = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    return matches.tolist()


# ============================================================
# Main: train V5d (or variant) + eval top-K
# ============================================================
def setup_vlm(vlm_name):
    cfg = VLM_CONFIGS[vlm_name]
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
        image_token_id = model.config.image_token_id
    elif cfg["model_class"] == "llava":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        processor = AutoProcessor.from_pretrained(cfg["model_id"])
        prepare_inputs = lambda s: prepare_llava_inputs(processor, s["image"], s["prompt"], "cuda")
        is_qwen = False
        image_token_id = model.config.image_token_index
    else:
        from transformers import LlavaNextForConditionalGeneration, AutoProcessor
        model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        processor = AutoProcessor.from_pretrained(cfg["model_id"])
        prepare_inputs = lambda s: prepare_llava_inputs(processor, s["image"], s["prompt"], "cuda")
        is_qwen = False
        image_token_id = model.config.image_token_index

    tc = getattr(model.config, "text_config", model.config)
    vocab = tc.vocab_size; D = tc.hidden_size
    emb = model.get_input_embeddings()
    if emb.weight.shape[0] > vocab:
        new = nn.Embedding(vocab, emb.weight.shape[1], device="cuda", dtype=emb.weight.dtype)
        new.weight.data.copy_(emb.weight.data[:vocab]); model.set_input_embeddings(new); emb = new
    lm_head = model.get_output_embeddings()
    for p in model.parameters(): p.requires_grad_(False)
    root_emb_table = emb.weight.detach().to(torch.float32)
    return model, processor, prepare_inputs, is_qwen, image_token_id, lm_head, root_emb_table, D, vocab


def collect_data_with_vision(model, prepare_inputs, is_qwen, image_token_id,
                              n_collect=100, gen_len=32, max_vis=256):
    """Collect (h_t, V_vis, root, cont_target) per record. V_vis is per-prompt."""
    from transformers.cache_utils import DynamicCache
    samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        n_collect, seed=43)

    h_t_list = []; root_id_list = []; cont_target_list = []
    prompt_idx_list = []; pos_list = []
    V_vis_per_prompt = {}   # prompt_idx -> tensor [n_vis, D]
    used = 0
    t0 = time.time()
    for s in samples:
        try:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            inputs = prepare_inputs(s)
            with torch.no_grad():
                out = model(**inputs, past_key_values=DynamicCache(),
                            use_cache=True, output_hidden_states=True, return_dict=True)
        except Exception: continue

        # Extract V_vision from last-layer hidden at image token positions
        input_ids = inputs["input_ids"]
        vis_positions = find_image_token_positions(input_ids, image_token_id)
        if not vis_positions:
            continue
        # subsample if too many
        if len(vis_positions) > max_vis:
            stride = len(vis_positions) // max_vis
            vis_positions = vis_positions[::stride][:max_vis]
        # extract hidden states at vision positions (last layer)
        full_h = out.hidden_states[-1][0]    # [seq, D]
        V_vis = full_h[vis_positions, :].float().detach().cpu()   # [n_vis, D]

        cur_kv = to_tuple_kv(out.past_key_values)
        h_t = out.hidden_states[-1][0, -1, :].float()
        v_logits = out.logits[0, -1, :].float()

        V_vis_per_prompt[used] = V_vis

        for pos in range(gen_len):
            v_root = int(v_logits.argmax().item())
            try:
                with torch.no_grad():
                    tok_inp = torch.tensor([[v_root]], device="cuda")
                    cur_len = cur_kv[0][0].shape[2]
                    if is_qwen:
                        pid = torch.tensor([[cur_len]], device="cuda").unsqueeze(0).expand(3, 1, -1)
                    else:
                        pid = torch.tensor([[cur_len]], device="cuda")
                    cp = torch.tensor([cur_len], device="cuda")
                    out_step = model(input_ids=tok_inp, past_key_values=kv_to_cache(cur_kv),
                                     position_ids=pid, cache_position=cp,
                                     use_cache=True, output_hidden_states=True, return_dict=True)
                    v_cont = int(out_step.logits[0, -1, :].argmax().item())
            except Exception: break
            h_t_list.append(h_t.detach().cpu())
            root_id_list.append(v_root)
            cont_target_list.append(v_cont)
            prompt_idx_list.append(used); pos_list.append(pos)
            h_t = out_step.hidden_states[-1][0, -1, :].float()
            cur_kv = to_tuple_kv(out_step.past_key_values)
            v_logits = out_step.logits[0, -1, :].float()
        used += 1
        if used % 25 == 0:
            print(f"  collected {used}/{n_collect}, n_records={len(h_t_list)}, "
                  f"elapsed {(time.time()-t0)/60:.1f}min", flush=True)
        if used >= n_collect: break

    return (torch.stack(h_t_list).to(torch.float32),
            torch.tensor(root_id_list, dtype=torch.long),
            torch.tensor(cont_target_list, dtype=torch.long),
            np.array(prompt_idx_list), np.array(pos_list),
            V_vis_per_prompt)


def split_train_eval(prompt_idx_arr, seed=0):
    rng = np.random.RandomState(seed)
    uniq = np.unique(prompt_idx_arr)
    n_eval_p = max(1, int(round(len(uniq) * 0.20)))
    eval_prompts = set(rng.choice(uniq, n_eval_p, replace=False).tolist())
    train_idx = np.array([i for i in range(len(prompt_idx_arr))
                           if prompt_idx_arr[i] not in eval_prompts])
    eval_idx = np.array([i for i in range(len(prompt_idx_arr))
                          if prompt_idx_arr[i] in eval_prompts])
    return train_idx, eval_idx


def train_v5d(head, h_t, root_id, cont_target, prompt_idx_arr, V_vis_per_prompt,
              train_idx, lm_head, root_emb_table=None, lr=5e-4, epochs=50, variant="v5d"):
    """Train V5d (or V5e-2) head. Per-prompt batching."""
    from collections import defaultdict
    by_p = defaultdict(list)
    for i in train_idx:
        by_p[int(prompt_idx_arr[i])].append(int(i))
    prompts = list(by_p.keys())

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    head.train()
    t0 = time.time()
    for ep in range(epochs):
        random.shuffle(prompts)
        for bstart in range(0, len(prompts), 4):
            bp = prompts[bstart:bstart + 4]
            opt.zero_grad()
            total_loss = 0; total_n = 0
            for p in bp:
                idx_list = by_p[p]
                idx = torch.tensor(idx_list, dtype=torch.long)
                h = h_t[idx].to("cuda")
                t = cont_target[idx].to("cuda")
                V_vis = V_vis_per_prompt[p].to("cuda")   # [n_vis, D]
                V_b = V_vis.unsqueeze(0).expand(h.shape[0], -1, -1)   # [n, n_vis, D]
                if variant == "v5d":
                    z = head(h, V_b)
                elif variant == "v5e2":
                    r = root_emb_table[root_id[idx]].to("cuda")
                    z = head(h, r, V_b)
                logits = lm_head(z.to(torch.bfloat16)).float()
                loss = F.cross_entropy(logits, t, reduction="sum")
                total_loss = total_loss + loss; total_n += len(idx_list)
            if total_n > 0:
                (total_loss / total_n).backward(); opt.step()
        if (ep+1) % 10 == 0:
            T_val = head.T.item() if hasattr(head, 'T') else "n/a"
            print(f"  ep {ep+1}: elapsed {(time.time()-t0)/60:.1f}min, T={T_val}", flush=True)
    head.eval()
    for p in head.parameters(): p.requires_grad_(False)


def eval_v5d(head, h_t, root_id, cont_target, prompt_idx_arr, V_vis_per_prompt,
             eval_idx, lm_head, root_emb_table=None, variant="v5d"):
    from collections import defaultdict
    by_p = defaultdict(list)
    for i in eval_idx:
        by_p[int(prompt_idx_arr[i])].append(int(i))
    hits = {k: 0 for k in [1, 3, 5, 10]}
    n = 0
    with torch.no_grad():
        for p, idx_list in by_p.items():
            idx = torch.tensor(idx_list, dtype=torch.long)
            h = h_t[idx].to("cuda")
            t = cont_target[idx].to("cuda")
            V_vis = V_vis_per_prompt[p].to("cuda")
            V_b = V_vis.unsqueeze(0).expand(h.shape[0], -1, -1)
            if variant == "v5d":
                z = head(h, V_b)
            elif variant == "v5e2":
                r = root_emb_table[root_id[idx]].to("cuda")
                z = head(h, r, V_b)
            logits = lm_head(z.to(torch.bfloat16)).float()
            top10 = logits.topk(10, -1).indices
            for k in [1, 3, 5, 10]:
                hits[k] += (top10[:, :k] == t.unsqueeze(-1)).any(-1).long().sum().item()
            n += len(idx_list)
    return {k: hits[k] / n for k in [1, 3, 5, 10]}


def main(vlm_name, variant, gpu):
    model, proc, prep_in, is_qwen, image_token_id, lm_head, root_emb_table, D, vocab = setup_vlm(vlm_name)
    print(f"[V5d ablation: vlm={vlm_name}, variant={variant}, D={D}, image_token_id={image_token_id}]\n")

    print("[collect data with V_vision]")
    h_t, root_id, cont_target, prompt_idx_arr, pos_arr, V_vis_per_prompt = \
        collect_data_with_vision(model, prep_in, is_qwen, image_token_id, n_collect=100)
    print(f"  records: {len(h_t)}, prompts with vision: {len(V_vis_per_prompt)}\n")

    train_idx, eval_idx = split_train_eval(prompt_idx_arr)
    print(f"  train: {len(train_idx)}, eval: {len(eval_idx)}\n")

    if variant == "v5d":
        head = V5d_Cont(dim=D, T_init=30.0).to("cuda")
    elif variant == "v5e2":
        head = V5e2_Cont(dim=D, alpha=30.0).to("cuda")
    else:
        raise ValueError(f"Unknown variant: {variant}")

    n_params = sum(p.numel() for p in head.parameters())
    print(f"[head params: {n_params/1e6:.2f}M]")

    print(f"\n[train {variant}]")
    train_v5d(head, h_t, root_id, cont_target, prompt_idx_arr, V_vis_per_prompt,
              train_idx, lm_head, root_emb_table=root_emb_table, variant=variant)
    if hasattr(head, 'T'):
        print(f"  final T = {head.T.item():.3f}")

    print(f"\n[eval {variant}]")
    eval_results = eval_v5d(head, h_t, root_id, cont_target, prompt_idx_arr, V_vis_per_prompt,
                              eval_idx, lm_head, root_emb_table=root_emb_table, variant=variant)
    print(f"  top-1: {eval_results[1]:.4f}, top-3: {eval_results[3]:.4f}, "
          f"top-5: {eval_results[5]:.4f}, top-10: {eval_results[10]:.4f}")

    out = {
        "vlm": vlm_name, "variant": variant, "dim": D,
        "image_token_id": image_token_id,
        "n_train": len(train_idx), "n_eval": len(eval_idx),
        "n_prompts_with_vision": len(V_vis_per_prompt),
        "params_M": n_params / 1e6,
        "eval_topk": eval_results,
        "final_T": head.T.item() if hasattr(head, 'T') else None,
    }
    save_path = f"./results/T1_3_{variant}_{vlm_name}.json"
    with open(save_path, "w") as f: json.dump(out, f, indent=2)
    print(f"\n[saved {save_path}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", required=True, choices=list(VLM_CONFIGS.keys()))
    p.add_argument("--variant", required=True, choices=["v5d", "v5e2"])
    p.add_argument("--gpu", default="0")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args.vlm, args.variant, args.gpu)
