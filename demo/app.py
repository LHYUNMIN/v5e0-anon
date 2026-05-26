"""V5e-0 Gradio demo: side-by-side AR vs V5e-0 inference on Qwen2-VL-2B.

Run locally:
    python demo/app.py
Open http://localhost:7860 in a browser.

To deploy as an anonymous HuggingFace Space:
    1. Create a new HF account with no identifying info.
    2. Create a Space (gradio template), MIT license.
    3. Push this file + a small download script that fetches checkpoints/v5e0_qwen2-2b.pt.
"""
import os
import sys
import time
import gc

import gradio as gr
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

# Make src/ and vendor/ importable.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "vendor"))

from vlm_io import prepare_vlm_inputs
from run import (
    V5e0_Cont, V5e0_Cont2, to_tuple_kv, kv_to_cache,
    build_tree_d3, build_attn_mask_and_pos, prune_kv,
)


# -------- Load model + drafter once --------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
CKPT = os.environ.get("V5E0_CKPT", os.path.join(ROOT, "checkpoints", "v5e0_qwen2-2b.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
M, K, MAX_TOK = 5, 3, 64

print(f"[demo] Loading {MODEL_ID} on {DEVICE}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    attn_implementation="sdpa",
).to(DEVICE).eval()
proc = AutoProcessor.from_pretrained(MODEL_ID)
D = model.config.hidden_size
emb = model.get_input_embeddings()
lm_head = model.get_output_embeddings()
root_emb_table = emb.weight.detach().to(torch.float32)
for p in model.parameters():
    p.requires_grad_(False)

ckpt = torch.load(CKPT, weights_only=False, map_location=DEVICE)
n1c = V5e0_Cont(dim=D, alpha_init=30.0).to(DEVICE)
n1c.Q_proj.weight.data = ckpt["W_Q1"].to(DEVICE).float()
n1c.Q_proj.bias.data = ckpt["W_Q1_bias"].to(DEVICE).float()
n1c2 = V5e0_Cont2(dim=D, alpha_init=30.0, beta_init=30.0).to(DEVICE)
n1c2.Q_proj.weight.data = ckpt["W_Q2"].to(DEVICE).float()
n1c2.Q_proj.bias.data = ckpt["W_Q2_bias"].to(DEVICE).float()
parents, depths = build_tree_d3(M, K)
n_input = len(parents) - 1
print("[demo] V5e-0 drafter loaded.")


def _prep(image, prompt):
    """Image may be a PIL.Image or a filepath."""
    if image is None:
        return prepare_vlm_inputs(proc, None, prompt, DEVICE)
    if hasattr(image, "save"):
        # PIL Image: save to a temp path so the Qwen processor can re-open it.
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name)
        return prepare_vlm_inputs(proc, tmp.name, prompt, DEVICE)
    return prepare_vlm_inputs(proc, image, prompt, DEVICE)


def gen_ar(image, prompt, max_tok=MAX_TOK):
    inputs = _prep(image, prompt)
    with torch.no_grad():
        out = model(**inputs, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
    cur_kv = to_tuple_kv(out.past_key_values)
    last_tok = int(out.logits[0, -1, :].argmax())
    generated = [last_tok]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_tok):
            tok_inp = torch.tensor([[last_tok]], device=DEVICE)
            cur_len = cur_kv[0][0].shape[2]
            pid = torch.tensor([[cur_len]], device=DEVICE).unsqueeze(0).expand(3, 1, -1)
            cp = torch.tensor([cur_len], device=DEVICE)
            o = model(input_ids=tok_inp, past_key_values=kv_to_cache(cur_kv),
                      position_ids=pid, cache_position=cp,
                      use_cache=True, return_dict=True)
            last_tok = int(o.logits[0, -1, :].argmax())
            cur_kv = to_tuple_kv(o.past_key_values)
            generated.append(last_tok)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    text = proc.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, elapsed, max_tok / elapsed


def gen_v5e0(image, prompt, max_tok=MAX_TOK):
    inputs = _prep(image, prompt)
    with torch.no_grad():
        out = model(**inputs, past_key_values=DynamicCache(),
                    use_cache=True, output_hidden_states=True, return_dict=True)
    cur_kv = to_tuple_kv(out.past_key_values)
    prompt_len = cur_kv[0][0].shape[2]
    last_h = out.hidden_states[-1][0, -1, :].float()
    last_lg = out.logits[0, -1, :].float()
    generated = []
    n_rounds = 0
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        while len(generated) < max_tok:
            anchor = int(last_lg.argmax())
            root_emb = root_emb_table[anchor].unsqueeze(0).to(DEVICE)
            z = n1c(last_h.unsqueeze(0).to(DEVICE), root_emb)
            cont_logits = lm_head(z.to(torch.bfloat16) if DEVICE == "cuda" else z).float()
            cont_topm = cont_logits.topk(M, dim=-1).indices[0]
            h_b = last_h.unsqueeze(0).expand(M, -1).to(DEVICE)
            re_b = root_emb.expand(M, -1)
            ce_b = root_emb_table[cont_topm].to(DEVICE)
            z2 = n1c2(h_b, re_b, ce_b)
            c2_lg = lm_head(z2.to(torch.bfloat16) if DEVICE == "cuda" else z2).float()
            c2_topk = c2_lg.topk(K, dim=-1).indices
            tree_list = [anchor] + cont_topm.tolist()
            for ci in range(M):
                tree_list.extend(c2_topk[ci].tolist())
            tree_ids = torch.tensor([tree_list], device=DEVICE)
            mask, pos_ids = build_attn_mask_and_pos(
                parents, depths, prompt_len, DEVICE,
                torch.bfloat16 if DEVICE == "cuda" else torch.float32, qwen=True,
            )
            cache_pos = torch.arange(prompt_len, prompt_len + n_input, device=DEVICE)
            outt = model(input_ids=tree_ids, past_key_values=kv_to_cache(cur_kv),
                         attention_mask=mask, position_ids=pos_ids,
                         cache_position=cache_pos,
                         use_cache=True, output_hidden_states=True, return_dict=True)
            tlog = outt.logits[0]
            thid = outt.hidden_states[-1][0]
            kv_after = to_tuple_kv(outt.past_key_values)
            cont_arg = int(tlog[0].argmax())
            acc_c = next((1 + cj for cj in range(M) if tree_list[1 + cj] == cont_arg), None)
            is_first = (n_rounds == 0)
            base = [anchor] if is_first else []
            if acc_c is not None:
                c2_start = 1 + M + (acc_c - 1) * K
                c2_arg = int(tlog[acc_c].argmax())
                acc_c2 = next((c2_start + j for j in range(K) if tree_list[c2_start + j] == c2_arg), None)
                if acc_c2 is not None:
                    bonus = int(tlog[acc_c2].argmax())
                    new_toks = base + [cont_arg, c2_arg, bonus]
                    acc_idx = [0, acc_c, acc_c2]
                    new_lg = tlog[acc_c2].float(); new_h = thid[acc_c2].float()
                else:
                    new_toks = base + [cont_arg, c2_arg]
                    acc_idx = [0, acc_c]
                    new_lg = tlog[acc_c].float(); new_h = thid[acc_c].float()
            else:
                new_toks = base + [cont_arg]
                acc_idx = [0]
                new_lg = tlog[0].float(); new_h = thid[0].float()
            cur_kv = prune_kv(kv_after, prompt_len, acc_idx)
            prompt_len += len(acc_idx)
            last_lg = new_lg; last_h = new_h
            generated.extend(new_toks)
            n_rounds += 1
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    text = proc.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, elapsed, len(generated) / elapsed


def demo_fn(image, prompt):
    if not prompt:
        return "(please enter a prompt)", "(please enter a prompt)", ""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    ar_text, ar_t, ar_tps = gen_ar(image, prompt)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    v5_text, v5_t, v5_tps = gen_v5e0(image, prompt)
    sp = ar_t / v5_t
    stats = (
        f"AR baseline   : {ar_t:.2f}s ({ar_tps:.1f} tok/s)\n"
        f"V5e-0 (ours)  : {v5_t:.2f}s ({v5_tps:.1f} tok/s)\n"
        f"Speedup        : {sp:.2f}×"
    )
    return ar_text, v5_text, stats


with gr.Blocks(title="V5e-0 demo") as demo:
    gr.Markdown(
        "# V5e-0: Minimalist VLM Speculative Decoding\n"
        "Side-by-side **autoregressive (AR)** vs.\ **V5e-0** generation on Qwen2-VL-2B.\n"
        "V5e-0 is a 4.72M-parameter drafter (two D×D linear heads) added on top of the frozen verifier.\n"
        "Upload an image and ask a question — both modes will produce the same answer; V5e-0 is faster."
    )
    with gr.Row():
        image_in = gr.Image(label="Image", type="pil")
        with gr.Column():
            prompt_in = gr.Textbox(label="Prompt", value="Describe the image.")
            run_btn = gr.Button("Generate (AR + V5e-0)", variant="primary")
            stats_out = gr.Textbox(label="Wall-clock", lines=4)
    with gr.Row():
        ar_out = gr.Textbox(label="AR output", lines=8)
        v5_out = gr.Textbox(label="V5e-0 output", lines=8)
    run_btn.click(demo_fn, [image_in, prompt_in], [ar_out, v5_out, stats_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
