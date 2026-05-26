#!/usr/bin/env python3
"""V5e-0 inference: Free-Root + Linear cont heads + single-root depth-3 tree.

Just specify --model_path and --ckpt to run inference. The verifier class
(Qwen2-VL / LLaVA / LLaVA-Next) is auto-detected from the model's config.

Usage:
  python infer.py \
      --model_path Qwen/Qwen2-VL-2B-Instruct \
      --ckpt ./checkpoints/v5e0_qwen2-2b.pt \
      --image /path/to/image.jpg \
      --prompt "Describe this image."
"""
import os, sys, json, time, argparse
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers.cache_utils import DynamicCache

# =============================================================================
# 1. Drafter heads
# =============================================================================
class V5e0_Cont(nn.Module):
    """Depth-2 head: z = W_Q1(h_t + α·E[r*]),  cont_logits = lm_head(z)."""
    def __init__(self, dim, alpha=30.0):
        super().__init__()
        self.W_Q = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.W_Q.weight); nn.init.zeros_(self.W_Q.bias)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
    def forward(self, h_t, root_emb):
        return self.W_Q(h_t + self.alpha * root_emb)

class V5e0_Cont2(nn.Module):
    """Depth-3 head: z2 = W_Q2(h_t + α·E[r*] + β·E[c]),  cont2_logits = lm_head(z2)."""
    def __init__(self, dim, alpha=30.0, beta=30.0):
        super().__init__()
        self.W_Q = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.W_Q.weight); nn.init.zeros_(self.W_Q.bias)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("beta",  torch.tensor(beta,  dtype=torch.float32))
    def forward(self, h_t, root_emb, cont_emb):
        return self.W_Q(h_t + self.alpha * root_emb + self.beta * cont_emb)


# =============================================================================
# 2. Single-root depth-3 tree management
# =============================================================================
class SSDTree:
    """Single-root tree: 1 anchor (KV) + 1 root + M conts + M*K cont2's.
       parents[0]=-1 (anchor), parents[1]=0 (root), parents[2..M+1]=1 (conts),
       parents[M+2..]=cont parent (cont2's).
    """
    def __init__(self, M=5, K=3):
        self.M, self.K = M, K
        parents, depths = [-1, 0], [0, 1]
        for _ in range(M):
            parents.append(1); depths.append(2)
        for cont_i in range(M):
            for _ in range(K):
                parents.append(2 + cont_i); depths.append(3)
        self.parents, self.depths = parents, depths
        self.n_input = len(parents) - 1     # 1 + M + M*K
        self.cont2_start = [1 + M + i*K for i in range(M)]

    def build_mask_and_pos(self, prompt_len, device, dtype, qwen=False, rope_delta=0):
        """Each tree node attends to: prefix + self + ancestor chain (siblings masked).

        `rope_delta` shifts position_ids (used for Qwen-VL MRoPE: position_id
        for text after the image is cache_position + rope_deltas, not the raw
        cache_position).
        """
        seq = self.n_input
        mask = torch.full((seq, prompt_len + seq), float('-inf'), device=device, dtype=dtype)
        mask[:, :prompt_len] = 0.0
        for i in range(1, len(self.parents)):
            in_idx = i - 1
            mask[in_idx, prompt_len + in_idx] = 0.0
            cur = self.parents[i]
            while cur != -1 and cur != 0:
                mask[in_idx, prompt_len + cur - 1] = 0.0
                cur = self.parents[cur]
        pos = torch.tensor(
            [prompt_len + self.depths[i] - 1 + rope_delta for i in range(1, len(self.parents))],
            device=device, dtype=torch.long,
        )
        if qwen:
            pos = pos.unsqueeze(0).unsqueeze(0).expand(3, 1, -1).contiguous()
        else:
            pos = pos.unsqueeze(0)
        return mask.unsqueeze(0).unsqueeze(0), pos

    def verify(self, tree_input_ids, tree_logits):
        """Greedy verification along single-root depth-3 tree.

        Returns: (cont_argmax, accepted_cont_idx_or_None, cont2_argmax_or_None,
                  accepted_cont2_idx_or_None, bonus_token, accepted_indices)
        """
        cont_argmax = int(tree_logits[0].argmax().item())
        accepted_cont_idx = None
        cont_position = -1
        for j in range(self.M):
            pos = 1 + j
            if tree_input_ids[pos] == cont_argmax:
                accepted_cont_idx, cont_position = pos, j
                break

        if accepted_cont_idx is None:
            bonus = cont_argmax
            return cont_argmax, None, None, None, bonus, [0]

        cont2_argmax = int(tree_logits[accepted_cont_idx].argmax().item())
        accepted_cont2_idx = None
        cont2_start = self.cont2_start[cont_position]
        for a in range(self.K):
            pos = cont2_start + a
            if tree_input_ids[pos] == cont2_argmax:
                accepted_cont2_idx = pos
                break

        if accepted_cont2_idx is None:
            bonus = cont2_argmax
            return cont_argmax, accepted_cont_idx, cont2_argmax, None, bonus, \
                   [0, accepted_cont_idx]

        bonus = int(tree_logits[accepted_cont2_idx].argmax().item())
        return cont_argmax, accepted_cont_idx, cont2_argmax, accepted_cont2_idx, bonus, \
               [0, accepted_cont_idx, accepted_cont2_idx]

    @staticmethod
    def prune_kv(kv_tuple, prompt_len, accepted_indices):
        """Keep prefix + accepted tree positions only."""
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


def to_tuple_kv(kv):
    if isinstance(kv, tuple) and len(kv) and isinstance(kv[0], tuple): return kv
    if isinstance(kv, DynamicCache):
        if hasattr(kv, 'key_cache'):  return tuple(zip(kv.key_cache, kv.value_cache))
        if hasattr(kv, 'layers'):     return tuple((l.keys, l.values) for l in kv.layers)
    return tuple((k, v) for k, v in kv)

def kv_to_cache(kv_tuple):
    return DynamicCache(ddp_cache_data=kv_tuple)


# =============================================================================
# 3. Auto-detection of verifier class
# =============================================================================
def detect_verifier_class(model_path):
    """Auto-detect VLM class from the model's HuggingFace config.

    Returns: one of 'qwen2-vl', 'llava', 'llava-next' (extensible).
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    archs = cfg.architectures or []
    arch  = archs[0] if archs else ""
    name  = arch.lower()
    if "qwen2vl" in name or "qwen2_vl" in name or "qwen2-vl" in name:
        return "qwen2-vl"
    if "llavanext" in name or "llava_next" in name or "llava-next" in name:
        return "llava-next"
    if "llava" in name:
        return "llava"
    raise ValueError(
        f"Unrecognized VLM architecture '{arch}'. Supported: Qwen2-VL, LLaVA, LLaVA-Next.\n"
        f"To add a new VLM, extend `detect_verifier_class` and `load_verifier`."
    )

def load_verifier(model_path):
    """Load a verifier model + processor by auto-detecting its class."""
    vclass = detect_verifier_class(model_path)
    if vclass == "qwen2-vl":
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        V = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa").to("cuda").eval()
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        is_qwen = True
    elif vclass == "llava":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        V = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        proc = AutoProcessor.from_pretrained(model_path); is_qwen = False
    elif vclass == "llava-next":
        from transformers import LlavaNextForConditionalGeneration, AutoProcessor
        V = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        proc = AutoProcessor.from_pretrained(model_path); is_qwen = False
    for p in V.parameters(): p.requires_grad_(False)
    return V, proc, vclass, is_qwen


# =============================================================================
# 4. V5e-0 inference wrapper
# =============================================================================
class V5e0:
    """V5e-0 drafter wrapper. Encapsulates verifier + Cont/Cont2 heads + tree."""
    def __init__(self, verifier, processor, cont, cont2, vclass, is_qwen, tree=None):
        self.V = verifier
        self.proc = processor
        self.cont, self.cont2 = cont, cont2
        self.vclass = vclass
        self.is_qwen = is_qwen
        self.tree = tree or SSDTree(M=5, K=3)
        self.lm_head = verifier.get_output_embeddings()
        self.E = verifier.get_input_embeddings().weight.detach().to(torch.float32)
        # MRoPE rope_deltas captured during prefill (Qwen-VL family); applied
        # to position_ids of every decode-mode step so MRoPE positions land on
        # the correct post-image-expansion text positions.
        self.rope_delta = 0

    @torch.no_grad()
    def prepare(self, image_path, prompt_text):
        """Build verifier inputs (image + text) per VLM-specific protocol."""
        if self.vclass == "qwen2-vl":
            # Use external helper if available, else fallback to inline
            try:
                sys.path.insert(0, "./vendor")
                from vlm_io import prepare_vlm_inputs
                return prepare_vlm_inputs(self.proc, image_path, prompt_text, "cuda")
            except Exception:
                img = Image.open(image_path).convert("RGB")
                msg = [{"role": "user", "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text}]}]
                txt = self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                return self.proc(images=img, text=txt, return_tensors="pt").to("cuda")
        else:   # llava / llava-next
            img = Image.open(image_path).convert("RGB")
            conv = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt_text}]}]
            txt = self.proc.apply_chat_template(conv, add_generation_prompt=True)
            return self.proc(images=img, text=txt, return_tensors="pt").to("cuda")

    @torch.no_grad()
    def prefill(self, inputs):
        """Run verifier on prompt; return (kv, h_t, last_logits, prompt_len).
        Also captures `rope_deltas` for Qwen-VL family so MRoPE position_ids
        are correct in subsequent decode-mode steps.
        """
        out = self.V(**inputs, past_key_values=DynamicCache(),
                     use_cache=True, output_hidden_states=True, return_dict=True)
        kv = to_tuple_kv(out.past_key_values)
        h_t = out.hidden_states[-1][0, -1, :].float()
        ll  = out.logits[0, -1, :].float()
        rd = getattr(out, "rope_deltas", None)
        if self.is_qwen and rd is not None:
            try:    self.rope_delta = int(rd.flatten()[0].item())
            except Exception: self.rope_delta = 0
        else:
            self.rope_delta = 0
        return kv, h_t, ll, kv[0][0].shape[2]

    @torch.no_grad()
    def step(self, kv, h_t, last_logits, prompt_len, is_first):
        """One SSD round. Returns (new_tokens, kv, h_t, last_logits, prompt_len)."""
        # (a) Free-Root: r* = argmax(last_logits)
        r_star = int(last_logits.argmax().item())
        root_emb = self.E[r_star].unsqueeze(0)

        # (b) V5e-0-Cont: top-M cont candidates
        z = self.cont(h_t.unsqueeze(0), root_emb)
        cont_logits = self.lm_head(z.to(torch.bfloat16)).float()
        conts = cont_logits.topk(self.tree.M, dim=-1).indices[0].tolist()

        # (c) V5e-0-Cont2: top-K cont2 per cont (batched over M)
        h_b = h_t.unsqueeze(0).expand(self.tree.M, -1)
        r_b = root_emb.expand(self.tree.M, -1)
        c_emb = self.E[torch.tensor(conts, device=h_t.device)]
        z2 = self.cont2(h_b, r_b, c_emb)
        cont2_logits = self.lm_head(z2.to(torch.bfloat16)).float()
        cont2_tops = cont2_logits.topk(self.tree.K, dim=-1).indices    # [M, K]

        # (d) Verifier tree forward
        tree_ids = [r_star] + conts
        for i in range(self.tree.M):
            tree_ids.extend(cont2_tops[i].tolist())
        x = torch.tensor([tree_ids], device=h_t.device)
        mask, pos = self.tree.build_mask_and_pos(prompt_len, x.device, self.V.dtype,
                                                  qwen=self.is_qwen,
                                                  rope_delta=self.rope_delta)
        cache_pos = torch.arange(prompt_len, prompt_len + self.tree.n_input, device=x.device)
        out = self.V(input_ids=x, past_key_values=kv_to_cache(kv),
                     attention_mask=mask, position_ids=pos, cache_position=cache_pos,
                     use_cache=True, output_hidden_states=True, return_dict=True)
        tree_logits = out.logits[0]
        tree_hidden = out.hidden_states[-1][0]
        kv_after = to_tuple_kv(out.past_key_values)

        # (e) Greedy verify
        cont_arg, acc_cont, cont2_arg, acc_cont2, bonus, acc_idx = \
            self.tree.verify(tree_ids, tree_logits)

        # (f) Build output (anchor only on first round)
        out_tokens = []
        if is_first: out_tokens.append(r_star)
        if acc_cont2 is not None:
            out_tokens += [cont_arg, cont2_arg, bonus]
        elif acc_cont is not None:
            out_tokens += [cont_arg, bonus]
        else:
            out_tokens += [bonus]

        # (g) KV pruning + state update
        kv_new = self.tree.prune_kv(kv_after, prompt_len, acc_idx)
        last_pos = acc_idx[-1]
        h_t_new = tree_hidden[last_pos].float()
        last_logits_new = tree_logits[last_pos].float()
        return out_tokens, kv_new, h_t_new, last_logits_new, prompt_len + len(acc_idx)


# =============================================================================
# 5. Generation functions
# =============================================================================
@torch.no_grad()
def generate_ssd(v5e0, inputs, max_tokens=64, stream=False):
    """V5e-0 SSD generation. Returns (tokens, wall_time_sec, n_per_round)."""
    kv, h_t, ll, prompt_len = v5e0.prefill(inputs)
    out_tokens = []
    n_per_round = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = 0
    while len(out_tokens) < max_tokens:
        new_toks, kv, h_t, ll, prompt_len = v5e0.step(kv, h_t, ll, prompt_len, is_first=(r==0))
        out_tokens.extend(new_toks)
        n_per_round.append(len(new_toks))
        if stream:
            print(v5e0.proc.tokenizer.decode(new_toks), end='', flush=True)
        r += 1
    torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
    if stream: print()
    return out_tokens[:max_tokens], elapsed, n_per_round


@torch.no_grad()
def generate_ar(verifier, processor, inputs, is_qwen, max_tokens=64, stream=False):
    """Greedy AR baseline."""
    out = verifier(**inputs, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
    kv = to_tuple_kv(out.past_key_values)
    last = int(out.logits[0, -1, :].argmax().item())
    tokens = [last]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(max_tokens - 1):
        x = torch.tensor([[last]], device="cuda")
        cur = kv[0][0].shape[2]
        pid = torch.tensor([[cur]], device="cuda")
        if is_qwen: pid = pid.unsqueeze(0).expand(3, 1, -1)
        cache_pos = torch.tensor([cur], device="cuda")
        out = verifier(input_ids=x, past_key_values=kv_to_cache(kv),
                       position_ids=pid, cache_position=cache_pos, use_cache=True, return_dict=True)
        last = int(out.logits[0, -1, :].argmax().item())
        kv = to_tuple_kv(out.past_key_values)
        tokens.append(last)
        if stream: print(processor.tokenizer.decode([last]), end='', flush=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
    if stream: print()
    return tokens, elapsed


# =============================================================================
# 6. Compare AR vs SSD
# =============================================================================
def compare(v5e0, image_path, prompt, max_tokens=64, runs=3, stream=True):
    inputs = v5e0.prepare(image_path, prompt)

    # warmup
    _ = generate_ar(v5e0.V, v5e0.proc, inputs, v5e0.is_qwen, max_tokens=8)
    _ = generate_ssd(v5e0, inputs, max_tokens=8)

    print(f"\n{'='*50}\n[AR baseline]\n{'='*50}")
    ar_tokens, ar_t = generate_ar(v5e0.V, v5e0.proc, inputs, v5e0.is_qwen, max_tokens, stream=stream)
    ar_tps_list = [max_tokens / ar_t]
    for _ in range(runs - 1):
        _, t = generate_ar(v5e0.V, v5e0.proc, inputs, v5e0.is_qwen, max_tokens, stream=False)
        ar_tps_list.append(max_tokens / t)

    print(f"\n{'='*50}\n[V5e-0 SSD]\n{'='*50}")
    ssd_tokens, ssd_t, n_per_round = generate_ssd(v5e0, inputs, max_tokens, stream=stream)
    ssd_tps_list = [len(ssd_tokens) / ssd_t]
    tpi_list     = [len(ssd_tokens) / len(n_per_round)] if n_per_round else [1.0]
    for _ in range(runs - 1):
        gen, t, npr = generate_ssd(v5e0, inputs, max_tokens, stream=False)
        ssd_tps_list.append(len(gen) / t)
        tpi_list.append(len(gen) / len(npr) if npr else 1.0)

    ar_tps  = float(np.mean(ar_tps_list))
    ssd_tps = float(np.mean(ssd_tps_list))
    tpi     = float(np.mean(tpi_list))
    print(f"\n{'='*50}")
    print(f"AR:   {ar_tps:6.2f} tok/s")
    print(f"SSD:  {ssd_tps:6.2f} tok/s ({tpi:.2f} tok/iter)")
    print(f"sp:   {ssd_tps/ar_tps:.3f}x")
    print(f"{'='*50}")


# =============================================================================
# 7. Loader: assemble V5e0 from --model_path + --ckpt
# =============================================================================
def load_v5e0(model_path, ckpt_path):
    """Auto-load verifier (by HuggingFace ID or local path) + drafter heads.

    Args:
      model_path: HuggingFace ID (e.g. "Qwen/Qwen2-VL-2B-Instruct") or local path.
      ckpt_path:  path to drafter weights saved by run.py with --save_ckpt.

    Returns: V5e0 instance ready for compare() / generate_ssd().
    """
    sys.path.insert(0, "./vendor")
    try:
        from runtime_env import strip_user_site_packages; strip_user_site_packages()
    except Exception:
        pass

    V, proc, vclass, is_qwen = load_verifier(model_path)
    print(f"[loaded verifier: {model_path} ({vclass})]", flush=True)

    tc = getattr(V.config, "text_config", V.config)
    D, vocab = tc.hidden_size, tc.vocab_size
    emb = V.get_input_embeddings()
    # NOTE: Qwen2-VL has extra embedding rows for special tokens (image, video) ABOVE
    # vocab_size. Trimming the embedding to vocab_size breaks the model's multimodal
    # processing. Keep the full embedding intact.

    # Load drafter state and sanity-check dim
    state = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    ckpt_dim = state.get("hidden_dim", state["W_Q1"].shape[0])
    if ckpt_dim != D:
        raise ValueError(
            f"Drafter checkpoint dim {ckpt_dim} does not match verifier hidden_dim {D}. "
            f"The checkpoint was trained for a different VLM."
        )
    alpha = state.get("alpha", 30.0)
    beta  = state.get("beta",  30.0)

    cont  = V5e0_Cont( dim=D, alpha=alpha).to("cuda")
    cont2 = V5e0_Cont2(dim=D, alpha=alpha, beta=beta).to("cuda")
    cont.W_Q.weight.data.copy_(state["W_Q1"].float())
    cont.W_Q.bias.data.copy_(state["W_Q1_bias"].float())
    cont2.W_Q.weight.data.copy_(state["W_Q2"].float())
    cont2.W_Q.bias.data.copy_(state["W_Q2_bias"].float())
    cont.eval(); cont2.eval()
    for p in cont.parameters():  p.requires_grad_(False)
    for p in cont2.parameters(): p.requires_grad_(False)

    print(f"[loaded drafter: {ckpt_path} (dim={D}, alpha={alpha}, beta={beta})]", flush=True)
    return V5e0(V, proc, cont, cont2, vclass=vclass, is_qwen=is_qwen)


# =============================================================================
# 8. CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    p.add_argument("--model_path", required=True,
                   help="HuggingFace ID (e.g. 'Qwen/Qwen2-VL-2B-Instruct') or local model path")
    p.add_argument("--ckpt", required=True,
                   help="Drafter checkpoint produced by run.py --save_ckpt")
    p.add_argument("--image", required=True, help="Image file path")
    p.add_argument("--prompt", default="Describe the image.")
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--no_stream", action="store_true")
    p.add_argument("--gpu", default="0")
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    v5e0 = load_v5e0(a.model_path, a.ckpt)
    compare(v5e0, a.image, a.prompt, max_tokens=a.max_tokens, runs=a.runs, stream=not a.no_stream)
