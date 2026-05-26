# V5e-0: Minimalist VLM Speculative Decoding

Anonymous code release for the EMNLP submission *"V5e-0: A Minimalist Self-Speculative Decoding Framework for Vision-Language Models"*.

V5e-0 is a self-speculative decoding (SSD) framework that drafts candidate continuations using only the verifier's **text-side hidden state** — no visual features in the drafter, no auxiliary teacher, no transformer-block drafter. The drafter consists of **two D×D linear continuation heads** (Cont + Cont2), trained per verifier in roughly one minute on a single A100, and uses the verifier's greedy argmax as a **parameter-free root** (Free-Root) inside a depth-3 tree (M=5, K=3, 21 input tokens).

Across **fifteen verifier VLMs** spanning four families (Qwen-VL, LLaVA, PaliGemma, Idefics2) and adding GLM-4V / InternVL3.5 / Pixtral, V5e-0 obtains an average **1.89× wall-clock speedup** over greedy autoregressive decoding while preserving answer-level accuracy on six vision-grounded benchmarks.

---

## Repository layout

```
v5e0-anon/
├── README.md                       # this file
├── requirements.txt                # pinned dependencies
├── scripts/
│   ├── prepare_data.sh             # COCO + LLaVA-Instruct download
│   ├── train_all_vlms.sh           # 15-VLM training batch
│   └── reproduce_paper_tables.sh   # per-table reproduction commands
├── src/
│   ├── run.py                      # main pipeline: collect → train Cont → train Cont2 → measure
│   ├── measure.py                  # measurement-only entry point
│   ├── infer.py                    # single-prompt inference helper
│   ├── medusa_ablation.py          # Medusa-VLM (parallel Cont2) vs Hydra-VLM baselines
│   ├── v5d_ablation.py             # vision-aware (cross-attention) drafter variant
│   ├── llm_v5e0.py                 # text-only LLM cross-modality control
│   ├── depth4_ablation.py          # depth-4 tree sweep
│   ├── statistical_tests.py        # paired-prompt significance tests
│   ├── theorem_empirical.py        # empirical check of Proposition 1
│   ├── lossless.py                 # bf16 token-level fidelity check
│   ├── make_figures.py             # figure generation
│   └── measurements/               # round-3/4/5 follow-up measurement scripts
│       ├── multi_benchmark_extended.py   # DocVQA / ChartQA / VQAv2 n=200
│       ├── mmbench_fix.py                # MMBench n=200
│       ├── mmmu_vqav2_fix.py             # MMMU n=200 + VQAv2 acc fix
│       ├── topp_preliminary.py           # top-p ∈ {1.0, 0.9, 0.7} preliminary
│       └── eagle_vlm_v2.py               # EAGLE-VLM Light head-to-head
├── vendor/
│   ├── vlm_io.py                   # minimal prepare_vlm_inputs helper
│   └── runtime_env.py              # user-site-packages stripping utility
├── results/                        # 19 measurement JSONs (raw outputs)
├── prompts/                        # train/eval prompt IDs (seed 43 train, 999 eval)
├── demo/                           # gradio demo + offline recording
└── figures/                        # paper figure source PDFs/JPGs
```

---

## Quickstart

### 1. Install dependencies

```bash
# Python 3.10 recommended (we tested with 3.10.x)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `transformers==5.5.4` and `qwen-vl-utils`. Newer transformers may require minor patches to the `DynamicCache` helpers (see `src/run.py:to_tuple_kv`).

### 2. Prepare data

Run the provided script (downloads COCO 2017 train images + LLaVA-Instruct-150K messages):

```bash
bash scripts/prepare_data.sh
```

The script populates `data/coco/train2017/` and `data/llava_messages_100k.jsonl`. The JSONL is the same file used by the V5e-0 training pipeline; each line carries the messages content with the image filename.

### 3. Train a drafter (one VLM)

```bash
# Qwen2-VL-2B example. Trains both Cont and Cont2 heads, then measures sp on 30 held-out prompts.
python src/run.py --vlm qwen2-2b --gpu 0
```

The script:
1. Collects 200 prompts × 32 positions = 6,400 records of `(h_t, root, cont, cont2)` greedy rollouts.
2. Trains V5e-0-Cont (Linear, identity init, CE, 100 epochs) — ~30 s on A100.
3. Trains V5e-0-Cont2 (Linear, identity init, CE, 100 epochs) — ~30 s on A100.
4. Saves the drafter checkpoint to `checkpoints/v5e0_<vlm>.pt` (~5 MB).
5. Wall-clock measures vs.\ greedy AR on 30 held-out LLaVA-Instruct prompts.

Supported `--vlm` keys (see `src/run.py:VLM_CONFIGS`): `qwen2-2b`, `qwen2-7b`, `qwen2.5-7b`, `llava-1.5-7b`, `llava-1.6-mistral-7b`, `llava-1.6-vicuna-7b`, `llava-onevision-7b`, `idefics2-8b`, `paligemma-3b`, `paligemma2-3b`, `paligemma2-10b`. Newer VLMs (GLM-4V, InternVL3.5, Pixtral, Qwen3-VL) are added in `src/run_new_vlms.py`.

### 4. Reproduce paper tables

```bash
bash scripts/reproduce_paper_tables.sh
```

The script chains the relevant measurement scripts and writes one JSON per table into `results/`.

---

## Ablations and rebuttal measurements

| Concern (rebuttal round) | Script | Output JSON |
|---|---|---|
| Medusa-VLM vs Hydra-VLM (R3) | `src/medusa_ablation.py` | `results/medusa_ablation_*.json` (in original pipeline) |
| V5d cross-attention (R3 Q4) | `src/v5d_ablation.py` | `results/v5d_ablation_*.json` |
| Precision sensitivity fp16/bf16/fp32 (R3 Q1) | (inline in measurement) | `results/precision_sensitivity.json` |
| DocVQA/ChartQA n=200 + VQAv2 n=200 (R4) | `src/measurements/multi_benchmark_clean.py` | `results/multi_benchmark_n200_clean.json` |
| MMBench n=200 (R4) | `src/measurements/mmbench_fix.py` | `results/mmbench_n200.json` |
| MMMU n=200 (R4) | `src/measurements/mmmu_vqav2_fix.py` | `results/mmmu_n200_vqav2_acc.json` |
| Top-p preliminary (R5 Phase 1) | `src/measurements/topp_preliminary.py` | `results/topp_preliminary.json` |
| EAGLE-VLM Light h2h (R5 Phase 2) | `src/measurements/eagle_vlm_v2.py` | `results/eagle_vlm_light.json` |

All raw measurement JSONs are checked in under `results/` so reviewers can verify the exact numbers reported in the paper without re-running.

---

## Reproducing the headline result (one command)

```bash
# Qwen2-VL-2B end-to-end on a single A100, ~3 minutes total
python src/run.py --vlm qwen2-2b --gpu 0
```

Expected output (subject to bf16 numerical variation):

```
[train V5e-0-Cont]   trained in ~0.5 min  Cont top-1 (eval) 0.88
[train V5e-0-Cont2]  trained in ~0.5 min  Cont2 top-1 (eval) 0.88
[measure n=30, 64-tok greedy]
  AR     :  36.4 t/s
  V5e-0  :  69.8 t/s
  sp     :  1.91×  ± 0.18
```

---

## Demo

`demo/web/` is a static side-by-side replay viewer for AR vs.\ V5e-0. It ships with five pre-recorded LLaVA-1.5-7B examples (speedups 1.02--2.06×) so reviewers can see the speedup mechanism without running inference.

**Quick local view:**
```bash
cd demo/web && python -m http.server 8000
# open http://localhost:8000
```

A hosted version (no signup required for the viewer) is linked in the paper's Code release paragraph. The hosted URL is anonymous (no GitHub username or author info). See `demo/README.md` for full deployment options (Cloudflare Pages / Netlify / GitHub Pages / static-only).

For interactive live inference (upload your own image), `demo/server.py` runs a FastAPI server that streams AR and SSD tokens over WebSocket; requires a GPU and a trained drafter checkpoint.

---

## Citation

Available after de-anonymization.

---

## License

This code release is distributed under the MIT License (see `LICENSE`). All model weights and checkpoints used in this work are obtained from public HuggingFace repositories of the original VLMs; please respect each model's own license when redistributing weights.
