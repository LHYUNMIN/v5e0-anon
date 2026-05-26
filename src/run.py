"""V5e-0: minimalist VLM speculative decoding drafter.

Pipeline (per VLM):
  1. Collect 200 prompts × 32 positions (h_t, root, cont, cont2) data
  2. Train V5e-0-Cont (Linear, identity init, CE, 100 epochs)
  3. Train V5e-0-Cont2 (Linear, identity init, CE, 100 epochs)
  4. Wall-clock measure with single-root depth-3 tree (M=5, K=3, 21 input tokens)

Architecture:
  V5e-0-Cont:  z  = W_Q1(h_t + α·E[root])     → cont prediction via lm_head
  V5e-0-Cont2: z2 = W_Q2(h_t + α·E[root] + β·E[cont])  → cont2 prediction via lm_head
  α = β = 30 (frozen scalars from prior SE-HD work)
  W_Q1, W_Q2: identity initialized, only trainable parameters

Inference (single-root depth-3 tree):
  1 anchor (in KV) + 1 root (verifier's argmax) + M=5 conts + 5×3=15 cont2's
  = 21 input tokens / round
  TPI ≈ 2.4-2.6, sp 1.6-2.2× wall-clock across 4 VLMs (mean 1.92×)

Usage:
  python run.py --vlm qwen2-2b   --gpu 0
  python run.py --vlm qwen2-7b   --gpu 0
  python run.py --vlm llava-1.5-7b --gpu 0
  python run.py --vlm llava-1.6-mistral-7b --gpu 0
"""
import os, sys, json, gc, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ============================================================================
# Configuration
# ============================================================================
N_COLLECT = 200    # training prompts
GEN_LEN   = 32     # AR positions per prompt → records = N_COLLECT × GEN_LEN
N_TEST    = 30     # held-out test prompts
MAX_TOKENS = 64    # tokens generated per prompt during measurement
EPOCHS    = 100
M = 5              # cont width per round
K = 3              # cont2 width per cont (symmetric)


VLM_CONFIGS = {
    "qwen2-2b":              {"model_id": "Qwen/Qwen2-VL-2B-Instruct",       "model_class": "qwen",       "alpha": 30.0, "beta": 30.0},
    "qwen2-7b":              {"model_id": "Qwen/Qwen2-VL-7B-Instruct",       "model_class": "qwen",       "alpha": 30.0, "beta": 30.0},
    "llava-1.5-7b":          {"model_id": "llava-hf/llava-1.5-7b-hf",         "model_class": "llava",      "alpha": 30.0, "beta": 30.0},
    "llava-1.6-mistral-7b":  {"model_id": "llava-hf/llava-v1.6-mistral-7b-hf", "model_class": "llava-next", "alpha": 30.0, "beta": 30.0},
}


# ============================================================================
# Drafter heads
# ============================================================================
class V5e0_Cont(nn.Module):
    """V5e-0-Cont: Linear over (h_t + α·root_emb)."""
    def __init__(self, dim, alpha_init=30.0):
        super().__init__()
        self.Q_proj = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.Q_proj.weight)
        nn.init.zeros_(self.Q_proj.bias)
        self.register_buffer("alpha", torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, h_t, root_emb):
        return self.Q_proj(h_t + self.alpha * root_emb)


class V5e0_Cont2(nn.Module):
    """V5e-0-Cont2: Linear over (h_t + α·root_emb + β·cont_emb)."""
    def __init__(self, dim, alpha_init=30.0, beta_init=30.0):
        super().__init__()
        self.Q_proj = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.Q_proj.weight)
        nn.init.zeros_(self.Q_proj.bias)
        self.register_buffer("alpha", torch.tensor(alpha_init, dtype=torch.float32))
        self.register_buffer("beta",  torch.tensor(beta_init,  dtype=torch.float32))

    def forward(self, h_t, root_emb, cont_emb):
        q = h_t + self.alpha * root_emb + self.beta * cont_emb
        return self.Q_proj(q)


# ============================================================================
# KV cache helpers (transformers 5.5.4 compatible)
# ============================================================================
def to_tuple_kv(kv):
    if kv is None:
        return None
    if isinstance(kv, tuple) and len(kv) > 0 and isinstance(kv[0], tuple):
        return kv
    from transformers.cache_utils import DynamicCache
    if isinstance(kv, DynamicCache):
        if hasattr(kv, 'key_cache') and hasattr(kv, 'value_cache'):
            return tuple((k, v) for k, v in zip(kv.key_cache, kv.value_cache))
        if hasattr(kv, 'layers'):
            return tuple((layer.keys, layer.values) for layer in kv.layers)
    if hasattr(kv, "to_legacy_cache"):
        return tuple((k, v) for k, v in kv.to_legacy_cache())
    return tuple((k, v) for k, v in kv)


def kv_to_cache(kv_tuple):
    from transformers.cache_utils import DynamicCache
    # transformers 5.x: DynamicCache(ddp_cache_data=...)
    # transformers 4.x: use from_legacy_cache or manual update
    try:
        return DynamicCache(ddp_cache_data=kv_tuple)
    except TypeError:
        # transformers 4.x fallback
        if hasattr(DynamicCache, 'from_legacy_cache'):
            return DynamicCache.from_legacy_cache(kv_tuple)
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv_tuple):
            cache.update(k, v, layer_idx)
        return cache


# ============================================================================
# Tree construction (single-root depth-3)
# ============================================================================
def build_tree_d3(M, K):
    """1 anchor (in KV, parent=-1) + 1 root + M conts + M×K cont2's.

    parents[0] = -1 (anchor in KV cache, before tree input)
    parents[1] = 0  (single root, child of anchor)
    parents[2..M+1] = 1 (M conts, children of root)
    parents[M+2..1+M+M*K] = (cont parent index) (cont2's)
    """
    parents = [-1, 0]
    depths  = [0, 1]
    for _ in range(M):
        parents.append(1); depths.append(2)
    for cont_i in range(M):
        cont_parent_idx = 2 + cont_i
        for _ in range(K):
            parents.append(cont_parent_idx)
            depths.append(3)
    return parents, depths


def build_attn_mask_and_pos(parents, depths, prompt_len, device, dtype, qwen=False):
    """Causal mask: each tree node attends to prefix + ancestor chain + self."""
    n_input = len(parents) - 1   # exclude anchor (in KV)
    seq_len = n_input
    total_len = prompt_len + seq_len
    mask = torch.full((seq_len, total_len), float('-inf'), device=device, dtype=dtype)
    mask[:, :prompt_len] = 0.0
    for i in range(1, len(parents)):
        in_idx = i - 1
        mask[in_idx, prompt_len + in_idx] = 0.0
        cur = parents[i]
        while cur != -1 and cur != 0:
            anc_in_idx = cur - 1
            mask[in_idx, prompt_len + anc_in_idx] = 0.0
            cur = parents[cur]
    pos_ids = torch.tensor(
        [prompt_len + depths[i] - 1 for i in range(1, len(parents))],
        device=device, dtype=torch.long
    )
    if qwen:
        pos_ids = pos_ids.unsqueeze(0).unsqueeze(0).expand(3, 1, -1).contiguous()
    else:
        pos_ids = pos_ids.unsqueeze(0)
    return mask.unsqueeze(0).unsqueeze(0), pos_ids


def prune_kv(kv_tuple, prompt_len, accepted_input_indices):
    """Keep prefix + accepted positions only."""
    new_kv = []
    if not accepted_input_indices:
        for k, v in kv_tuple:
            new_kv.append((k[..., :prompt_len, :].contiguous(),
                           v[..., :prompt_len, :].contiguous()))
        return tuple(new_kv)
    indices = torch.tensor(
        [prompt_len + i for i in accepted_input_indices],
        device=kv_tuple[0][0].device, dtype=torch.long
    )
    for k, v in kv_tuple:
        prefix_k = k[..., :prompt_len, :]
        prefix_v = v[..., :prompt_len, :]
        sel_k = k.index_select(-2, indices)
        sel_v = v.index_select(-2, indices)
        new_kv.append((torch.cat([prefix_k, sel_k], dim=-2).contiguous(),
                       torch.cat([prefix_v, sel_v], dim=-2).contiguous()))
    return tuple(new_kv)


# ============================================================================
# Data loading
# ============================================================================
def load_llava_prompts(path, n, seed):
    """Load (image, prompt) pairs from llava_messages_100k.jsonl."""
    rng = random.Random(seed)
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            msgs = d['messages']
            img = next((c['image'] for c in msgs[0]['content'] if c['type'] == 'image'), None)
            txt = next((c['text']  for c in msgs[0]['content'] if c['type'] == 'text'),  None)
            if img and txt and os.path.exists(img):
                rows.append({"image": img, "prompt": txt})
                if len(rows) >= n * 3:
                    break
    rng.shuffle(rows)
    return rows[:n]


def prepare_qwen_inputs(processor, image_path, prompt_text, device):
    sys.path.insert(0, "./vendor")
    from vlm_io import prepare_vlm_inputs
    return prepare_vlm_inputs(processor, image_path, prompt_text, device)


def prepare_llava_inputs(processor, image_path, prompt_text, device):
    image = Image.open(image_path).convert("RGB")
    conversation = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return processor(images=image, text=prompt, return_tensors="pt").to(device)


# ============================================================================
# Inference: SSD measurement loop
# ============================================================================
def measure_d3(model, n1_cont, n1_cont2, lm_head, root_emb_table,
                prepare_inputs, is_qwen, test_samples, M, K, max_tokens):
    """Wall-clock measurement: AR baseline vs V5e-0 SSD with depth-3 tree."""
    from transformers.cache_utils import DynamicCache
    parents, depths = build_tree_d3(M, K)
    n_input = len(parents) - 1   # 1 + M + M*K

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
        last_token = int(out.logits[0, -1, :].argmax().item())
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            with torch.no_grad():
                for _ in range(max_tokens):
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
        except Exception:
            continue
        torch.cuda.synchronize()
        ar_tps_list.append(max_tokens / (time.perf_counter() - t0))

        # ---------- V5e-0 SSD (single-root depth-3) ----------
        try:
            inputs = prepare_inputs(s)
            with torch.no_grad():
                out = model(**inputs, past_key_values=DynamicCache(),
                            use_cache=True, output_hidden_states=True, return_dict=True)
        except Exception:
            continue
        cur_kv = to_tuple_kv(out.past_key_values)
        prompt_len = cur_kv[0][0].shape[2]
        last_h_t = out.hidden_states[-1][0, -1, :].float()
        last_logits = out.logits[0, -1, :].float()

        generated_tokens = []
        n_rounds = 0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            with torch.no_grad():
                while len(generated_tokens) < max_tokens:
                    # Free-Root: r = argmax(verifier's last logits)
                    anchor_argmax = int(last_logits.argmax().item())
                    root_emb = root_emb_table[anchor_argmax].unsqueeze(0)

                    # V5e-0-Cont: top-M cont candidates
                    z = n1_cont(last_h_t.unsqueeze(0), root_emb)
                    cont_logits = lm_head(z.to(torch.bfloat16)).float()
                    cont_topm = cont_logits.topk(M, dim=-1).indices[0]

                    # V5e-0-Cont2: top-K cont2 per cont (batched over M)
                    h_t_b = last_h_t.unsqueeze(0).expand(M, -1)
                    root_emb_b = root_emb.expand(M, -1)
                    cont_embs = root_emb_table[cont_topm]
                    z2 = n1_cont2(h_t_b, root_emb_b, cont_embs)
                    cont2_logits = lm_head(z2.to(torch.bfloat16)).float()
                    cont2_topk = cont2_logits.topk(K, dim=-1).indices

                    # Build tree input: [root, conts..., cont2's...]
                    tree_input_ids = [anchor_argmax] + cont_topm.tolist()
                    for cont_i in range(M):
                        tree_input_ids.extend(cont2_topk[cont_i].tolist())
                    tree_input_t = torch.tensor([tree_input_ids], device="cuda")

                    mask, pos_ids = build_attn_mask_and_pos(
                        parents, depths, prompt_len, "cuda", model.dtype, qwen=is_qwen
                    )
                    cache_pos = torch.arange(prompt_len, prompt_len + n_input, device="cuda")
                    out_tree = model(input_ids=tree_input_t,
                                      past_key_values=kv_to_cache(cur_kv),
                                      attention_mask=mask, position_ids=pos_ids,
                                      cache_position=cache_pos,
                                      use_cache=True, output_hidden_states=True, return_dict=True)
                    tree_logits = out_tree.logits[0]
                    tree_hidden = out_tree.hidden_states[-1][0]
                    kv_after = to_tuple_kv(out_tree.past_key_values)

                    # Verify: cont match against verifier's argmax at root[0] position
                    cont_argmax = int(tree_logits[0].argmax().item())
                    accepted_cont_idx = None
                    for c_j in range(M):
                        pos = 1 + c_j
                        if tree_input_ids[pos] == cont_argmax:
                            accepted_cont_idx = pos
                            break

                    is_first = (n_rounds == 0)
                    base = [anchor_argmax] if is_first else []   # bug-fix: skip anchor in rounds > 0

                    if accepted_cont_idx is not None:
                        cont_position_in_array = accepted_cont_idx - 1
                        cont2_start_idx = 1 + M + cont_position_in_array * K
                        cont2_argmax = int(tree_logits[accepted_cont_idx].argmax().item())
                        accepted_cont2_idx = None
                        for c2_j in range(K):
                            pos = cont2_start_idx + c2_j
                            if tree_input_ids[pos] == cont2_argmax:
                                accepted_cont2_idx = pos
                                break
                        if accepted_cont2_idx is not None:
                            # depth-3 accept: 3 tokens (cont, cont2, bonus)
                            bonus = int(tree_logits[accepted_cont2_idx].argmax().item())
                            new_tokens = base + [cont_argmax, cont2_argmax, bonus]
                            accepted_indices = [0, accepted_cont_idx, accepted_cont2_idx]
                            new_last_logits = tree_logits[accepted_cont2_idx].float()
                            new_last_h_t = tree_hidden[accepted_cont2_idx].float()
                            n_d3 += 1
                        else:
                            # depth-2 accept: 2 tokens (cont, cont2 as bonus)
                            new_tokens = base + [cont_argmax, cont2_argmax]
                            accepted_indices = [0, accepted_cont_idx]
                            new_last_logits = tree_logits[accepted_cont_idx].float()
                            new_last_h_t = tree_hidden[accepted_cont_idx].float()
                            n_d2 += 1
                    else:
                        # depth-1 only: 1 token (cont_argmax = verifier's argmax at root)
                        new_tokens = base + [cont_argmax]
                        accepted_indices = [0]
                        new_last_logits = tree_logits[0].float()
                        new_last_h_t = tree_hidden[0].float()

                    cur_kv = prune_kv(kv_after, prompt_len, accepted_indices)
                    prompt_len += len(accepted_indices)
                    last_logits = new_last_logits
                    last_h_t = new_last_h_t
                    generated_tokens.extend(new_tokens)
                    n_rounds += 1
        except Exception as e:
            print(f"  prompt {s_i} SSD failed: {e}", flush=True)
            continue
        torch.cuda.synchronize()
        ssd_tps_list.append(len(generated_tokens) / (time.perf_counter() - t0))
        n_rounds_total += n_rounds; n_tokens_total += len(generated_tokens)

    return ar_tps_list, ssd_tps_list, n_d2, n_d3, n_rounds_total, n_tokens_total


# ============================================================================
# Main pipeline: collect → train Cont → train Cont2 → measure
# ============================================================================
def main(vlm_name, save_ckpt=None, n_test=None, max_tokens=None):
    if n_test is not None:
        global N_TEST
        N_TEST = n_test
    if max_tokens is not None:
        global MAX_TOKENS
        MAX_TOKENS = max_tokens
    cfg = VLM_CONFIGS[vlm_name]
    print(f"[VLM: {vlm_name}, single-root depth-3 M={M}, K={K}]\n", flush=True)

    sys.path.insert(0, "./vendor")
    from runtime_env import strip_user_site_packages
    strip_user_site_packages()
    from transformers.cache_utils import DynamicCache

    # ---------- Load verifier ----------
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
    print(f"  hidden_dim={hidden_dim}, vocab={vocab_size}\n", flush=True)

    # ---------- Phase 1: collect training data ----------
    print(f"[collect {N_COLLECT} prompts × {GEN_LEN}]")
    samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        N_COLLECT, seed=43)

    h_t_list = []; root_id_list = []; cont_target_list = []
    prompt_idx_list = []; pos_list = []
    used = 0; t0 = time.time()
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
        if used % 50 == 0:
            print(f"  {used}/{N_COLLECT}, n={len(h_t_list)}, "
                  f"elapsed {(time.time()-t0)/60:.1f}min", flush=True)
        if used >= N_COLLECT:
            break

    h_t_tensor = torch.stack(h_t_list).to(torch.float32)
    root_id_t = torch.tensor(root_id_list, dtype=torch.long)
    cont_target_t = torch.tensor(cont_target_list, dtype=torch.long)
    prompt_idx_arr = np.array(prompt_idx_list)
    pos_arr = np.array(pos_list)
    print(f"  collected n={h_t_tensor.shape[0]}\n")

    # ---------- 80/20 prompt-level split ----------
    rng = np.random.RandomState(0)
    unique_prompts = np.unique(prompt_idx_arr)
    n_eval_p = max(1, int(round(len(unique_prompts) * 0.20)))
    eval_prompts = set(rng.choice(unique_prompts, n_eval_p, replace=False).tolist())
    train_idx = np.array([i for i in range(len(prompt_idx_arr))
                           if prompt_idx_arr[i] not in eval_prompts])

    from collections import defaultdict
    train_by_p = defaultdict(list)
    for i in train_idx:
        train_by_p[prompt_idx_arr[i]].append(int(i))
    train_prompts_list = list(train_by_p.keys())

    # ---------- Phase 2: train V5e-0-Cont ----------
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    n1_cont = V5e0_Cont(dim=hidden_dim, alpha_init=cfg["alpha"]).to("cuda")
    opt = torch.optim.AdamW(n1_cont.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"[train V5e-0-Cont {EPOCHS} epochs]")
    t0 = time.time()
    for ep in range(EPOCHS):
        n1_cont.train()
        random.shuffle(train_prompts_list)
        for bstart in range(0, len(train_prompts_list), 8):
            bp = train_prompts_list[bstart:bstart + 8]
            opt.zero_grad()
            total_loss = 0; total_n = 0
            for p in bp:
                recs = train_by_p[p]
                h  = h_t_tensor[recs].to("cuda")
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
    print(f"  trained in {(time.time()-t0)/60:.1f}min\n")

    # ---------- Phase 3: build cont2 tuples ----------
    print("[construct cont2 tuples]")
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
    eval_cont2_recs = [r for r in cont2_recs if r["prompt_idx"] in eval_prompts]
    print(f"  cont2 tuples: train {len(train_cont2_recs)}, eval {len(eval_cont2_recs)}\n")

    # ---------- Phase 4: train V5e-0-Cont2 ----------
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
    t0 = time.time()
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
    print(f"  trained in {(time.time()-t0)/60:.1f}min\n")

    # ---------- Phase 5: wall-clock measurement ----------
    test_samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        N_TEST, seed=999)

    n_input = 1 + M + M*K
    print(f"[walltime test M={M}, K={K}, tree={n_input} input tokens, {N_TEST} prompts × {MAX_TOKENS}]")
    ar_list, ssd_list, n_d2, n_d3, n_rounds, n_tokens = measure_d3(
        model, n1_cont, n1_cont2, lm_head, root_emb_table, prepare_inputs,
        is_qwen, test_samples, M, K, MAX_TOKENS)
    sp = [s/a for a, s in zip(ar_list, ssd_list)]
    ar_a = np.array(ar_list); ssd_a = np.array(ssd_list); sp_a = np.array(sp)
    d2_rate = n_d2 / n_rounds if n_rounds > 0 else 0
    d3_rate = n_d3 / n_rounds if n_rounds > 0 else 0
    tpi = n_tokens / n_rounds if n_rounds > 0 else 0

    print(f"\n{'='*70}")
    print(f"{vlm_name} V5e-0 single-root depth-3 (M={M}, K={K})")
    print(f"{'='*70}")
    print(f"  AR:    {ar_a.mean():.2f} ± {ar_a.std():.2f} t/s")
    print(f"  SSD:   {ssd_a.mean():.2f} ± {ssd_a.std():.2f} t/s")
    print(f"  sp:    {sp_a.mean():.3f} ± {sp_a.std():.3f}")
    print(f"        median {np.median(sp_a):.3f}, range [{sp_a.min():.2f}, {sp_a.max():.2f}]")
    print(f"  TPI {tpi:.2f}, d2 {d2_rate:.3f}, d3 {d3_rate:.3f}")

    out = {
        "vlm": vlm_name, "M": M, "K": K, "n_input_tokens": n_input,
        "ar_tps":  {"mean": float(ar_a.mean()),  "std": float(ar_a.std())},
        "ssd_tps": {"mean": float(ssd_a.mean()), "std": float(ssd_a.std())},
        "sp": {"mean": float(sp_a.mean()), "std": float(sp_a.std()),
               "median": float(np.median(sp_a)),
               "min": float(sp_a.min()), "max": float(sp_a.max())},
        "tpi": tpi, "d2_rate": d2_rate, "d3_rate": d3_rate,
        "n_prompts": len(ar_list),
        "raw": {"ar": ar_list, "ssd": ssd_list, "sp": sp},
    }
    # Suffix output with max_tokens if non-default, to keep separate measurements
    _tok_suffix = f"_t{MAX_TOKENS}" if MAX_TOKENS != 64 else ""
    save_path = f"./results/path1_v5e0_d3_4vlm_{vlm_name}{_tok_suffix}.json"
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved {save_path}]")

    # Save trained drafter checkpoint (for infer.py)
    if save_ckpt:
        torch.save({
            "vlm": vlm_name, "hidden_dim": hidden_dim,
            "alpha": cfg["alpha"], "beta": cfg["beta"],
            "W_Q1":      n1_cont.Q_proj.weight.detach().cpu(),
            "W_Q1_bias": n1_cont.Q_proj.bias.detach().cpu(),
            "W_Q2":      n1_cont2.Q_proj.weight.detach().cpu(),
            "W_Q2_bias": n1_cont2.Q_proj.bias.detach().cpu(),
        }, save_ckpt)
        print(f"[saved drafter ckpt {save_ckpt}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", required=True, choices=list(VLM_CONFIGS.keys()))
    p.add_argument("--gpu", default="0")
    p.add_argument("--save_ckpt", default=None,
                   help="path to save trained drafter heads (.pt) for infer.py")
    p.add_argument("--n_test", type=int, default=None,
                   help="override N_TEST (number of held-out test prompts)")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="override MAX_TOKENS (generation length per prompt)")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args.vlm, save_ckpt=args.save_ckpt, n_test=args.n_test, max_tokens=args.max_tokens)
