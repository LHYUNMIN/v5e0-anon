# V5e-0 Web Demo (AR vs SSD side-by-side)

A web page that **replays** pre-recorded autoregressive (AR) vs V5e-0 speculative decoding (SSD) generations side by side. The right pane (V5e-0) emits tokens in bursts of 2-4 per verifier forward, finishing roughly 1.5-2× faster than the left pane (AR).

The demo ships with 5 pre-recorded LLaVA-1.5-7B samples (speedups 1.02-2.06×). Reviewers can either:
- **View the live URL** (see "Deployment" below) — single click, no setup.
- **Run locally**:
  ```bash
  cd demo/web
  python -m http.server 8000
  # open http://localhost:8000
  ```

## Files

```
demo/
├── README.md                # this file
├── record_inference.py      # records AR + SSD with per-token timestamps → sample_*.json
├── aggregate_samples.py     # combines sample_*.json → web/samples.json
├── server.py                # optional FastAPI live-inference server (for fresh demos)
└── web/                     # static replay site
    ├── index.html
    ├── style.css
    ├── main.js              # plays back samples.json with the original timings
    └── samples.json         # 5 pre-recorded examples (LLaVA-1.5-7B, ~1.5 MB)
```

## What viewers see

1. **Header**: V5e-0: VLM Speculative Decoding
2. **Controls**: example selector, Play / Reset, Speed (1× / 2× / 4× / 8×)
3. **Image + prompt** rendered above the comparison
4. **Side-by-side panes**:
   - Left "AR Baseline" — tokens appear one at a time
   - Right "V5e-0 SSD" — tokens appear in **green bursts of 2-4** = the multi-token accept events
5. **Per-pane metrics**: elapsed ms, tokens emitted, tokens/sec
6. **Speedup banner**: big number, e.g. "2.06×"
7. **Footer**: brief explanation that green bursts = source of speedup

## Deployment options (anonymous-friendly)

### Option 1 (recommended for ACL/EMNLP anonymity): Cloudflare Pages

Free, anonymous URL like `https://v5e0-anon-demo.pages.dev/`, no GitHub username in the URL.

1. Sign up at <https://dash.cloudflare.com/sign-up/pages> with an anonymous email.
2. **Create a project** → **Direct Upload** → drag the `demo/web/` folder.
3. Project name: e.g. `v5e0-anon-demo`.
4. Deploy → get URL `https://v5e0-anon-demo.pages.dev/`.
5. Paste this URL into the paper's Code release paragraph.

### Option 2: Netlify drop

1. Go to <https://app.netlify.com/drop>.
2. Drag the entire `demo/web/` folder onto the page.
3. Get an instant URL like `https://random-name-1234.netlify.app/`.
4. Optionally rename in Site settings → Site name (e.g. `v5e0-anon-demo`) → final URL `https://v5e0-anon-demo.netlify.app/`.

### Option 3: GitHub Pages (less anonymous — URL contains GitHub username)

If anonymity tolerance allows the source GitHub URL:

1. In the source repo, **Settings → Pages**.
2. Source: **Deploy from a branch** → Branch: `main` → Folder: `/demo/web`.
3. Save. URL: `https://<github-username>.github.io/<repo>/`.

Note: this exposes the GitHub username in the URL, so reviewers could in principle navigate to the source GitHub repo. Most anonymous-submission guidelines accept this because anonymous.4open.science already mirrors the code anonymously, but if maximal anonymity is required prefer Option 1 or 2.

### Option 4: Inside the anonymous repository (no hosting)

Reviewers download the anonymous repository tarball and open `demo/web/index.html` locally. No live URL is needed. The repository's README points to this directory.

## Re-recording fresh samples (optional)

```bash
# Train a drafter first (see top-level README)
python ../src/run.py --vlm qwen2-2b --gpu 0

# Record one example
python record_inference.py \
    --model_path Qwen/Qwen2-VL-2B-Instruct \
    --ckpt ../checkpoints/v5e0_qwen2-2b.pt \
    --image /path/to/img.jpg \
    --prompt "Describe this image." \
    --max_tokens 64 \
    --include_image_base64 \
    --output sample_1.json

# Repeat for more samples, then aggregate
python aggregate_samples.py . web/samples.json
```

## Live inference mode (FastAPI)

For interactive demos where users upload their own images (requires the trained drafter checkpoint + a GPU):

```bash
pip install fastapi uvicorn python-multipart
python server.py \
    --model_path Qwen/Qwen2-VL-2B-Instruct \
    --ckpt ../checkpoints/v5e0_qwen2-2b.pt \
    --host 0.0.0.0 --port 8080
# Open http://localhost:8080
```

The frontend connects via WebSocket and streams tokens in real-time. The included `web/main.js` is for static replay; live mode requires a small JS extension — see `server.py` for the server-side protocol.
