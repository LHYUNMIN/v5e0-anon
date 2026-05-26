# V5e-0 Demo

Side-by-side autoregressive (AR) vs. V5e-0 inference on Qwen2-VL-2B.

## Run locally

```bash
# 1. Make sure you have a trained drafter checkpoint at checkpoints/v5e0_qwen2-2b.pt
#    (run `python src/run.py --vlm qwen2-2b` once to produce it)
python demo/app.py
```

Then open <http://localhost:7860> in a browser.

## Deploy as anonymous HuggingFace Space

1. Create a fresh HF account using an email address with no identifying information.
2. Create a new Space (gradio template, MIT license).
3. Upload `demo/app.py` + add a small download script that fetches the trained drafter checkpoint, e.g.,
   ```python
   # In Space init
   from huggingface_hub import hf_hub_download
   import os
   os.makedirs("checkpoints", exist_ok=True)
   ckpt_path = hf_hub_download(
       repo_id="anonymous-user/v5e0-checkpoints",
       filename="v5e0_qwen2-2b.pt",
       local_dir="checkpoints",
   )
   ```
4. Set the Space hardware to a T4 small or A10G small (Qwen2-VL-2B fits in <8 GB on bf16).

## Offline demo recording

`demo_recording.mp4` (to be added) — a 30-second screen capture of the gradio interface running the demo on three example prompts (description, OCR, counting). This serves as a fallback if the live Space is unreachable during the review window.
