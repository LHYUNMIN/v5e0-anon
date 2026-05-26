# V5e-0 Web Demo

A side-by-side, browser-based comparison of **AR (autoregressive)** vs **V5e-0 SSD** token streaming on a Qwen2-VL-2B verifier. Users upload an image and type a prompt; both modes generate live and the speedup is shown in the header.

## Live URL

> 🔗 **https://saying-lion-rather-fare.trycloudflare.com/**

The right pane (V5e-0) emits tokens in green bursts of 2--4 per verifier forward — those bursts are the multi-token accept events that produce the wall-clock speedup.

## Files

```
demo/
├── README.md                # this file
├── server.py                # FastAPI live-inference server (WebSocket-based)
└── web/                     # browser UI
    ├── index.html
    ├── style.css
    ├── main.js              # handles both live WebSocket mode and pre-recorded replay
    └── samples.json         # 5 pre-recorded LLaVA-1.5-7B examples (for visitors without an image)
```

## Self-host

Requires a CUDA GPU + a trained drafter checkpoint (see top-level README).

```bash
pip install fastapi uvicorn python-multipart
python demo/server.py \
    --model_path Qwen/Qwen2-VL-2B-Instruct \
    --ckpt checkpoints/v5e0_qwen2-2b.pt \
    --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

The server:
- Serves `web/` as static assets on `/`.
- Accepts WebSocket connections at `/ws` carrying `{prompt, image_b64}`.
- Streams two parallel `{stream: "ar" | "ssd", token: "...", elapsed_ms: ...}` messages per emitted token.
- Sends a final `{stream, done: true, total_tokens, total_ms, tokens_per_sec}` summary per stream.

See `server.py` for the full message schema.

## Expose publicly (anonymous URL)

Any tunneling service works. The live URL above uses a Cloudflare Quick Tunnel (no signup, anonymous `*.trycloudflare.com` URL):

```bash
# Download cloudflared (single static binary)
curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ./cloudflared
chmod +x ./cloudflared

# Run server first (above), then in another shell:
./cloudflared tunnel --url http://localhost:8000
# prints something like: https://random-words-here.trycloudflare.com
```

The tunnel URL is anonymous (no GitHub username, no account name) and stays up as long as the cloudflared process runs.

Alternatives:
- `ngrok http 8000` — requires free ngrok signup.
- `npx localtunnel --port 8000` — quick but URL is rate-limited.
- Cloudflare Pages / Netlify drop with only the `web/` folder, if you want the replay-only mode without the live server.
