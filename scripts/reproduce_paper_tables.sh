#!/usr/bin/env bash
#
# Reproduce the per-table measurement JSONs reported in the paper.
# Each block writes one JSON into results/.
#
# Prerequisites:
#   - Drafter checkpoints in checkpoints/ (run scripts/train_all_vlms.sh first)
#   - COCO + LLaVA-Instruct data in data/ (run scripts/prepare_data.sh first)

set -e
mkdir -p results

echo "[1/8] Main 15-VLM throughput sweep  (Table tab:main_4vlm in paper)"
for vlm in qwen2-2b qwen2-7b llava-1.5-7b llava-1.6-mistral-7b paligemma-3b paligemma2-3b paligemma2-10b idefics2-8b; do
  python src/measure.py --vlm "$vlm" --gpu 0
done

echo "[2/8] Vision-grounded benchmarks n=200 (DocVQA / ChartQA / VQAv2)"
python src/measurements/multi_benchmark_clean.py

echo "[3/8] MMBench n=200"
python src/measurements/mmbench_fix.py

echo "[4/8] MMMU n=200 + VQAv2 acc fix"
python src/measurements/mmmu_vqav2_fix.py

echo "[5/8] Top-p preliminary (top-p ∈ {1.0, 0.9, 0.7})"
python src/measurements/topp_preliminary.py

echo "[6/8] EAGLE-VLM Light head-to-head"
python src/measurements/eagle_vlm_v2.py

echo "[7/8] V5d vision-aware ablation"
python src/v5d_ablation.py --vlm qwen2-2b --gpu 0

echo "[8/8] Medusa-VLM vs Hydra-VLM ablation"
python src/medusa_ablation.py --vlm qwen2-2b --gpu 0
python src/medusa_ablation.py --vlm qwen2-7b --gpu 0
python src/medusa_ablation.py --vlm llava-1.5-7b --gpu 0

echo "All reproduction runs done. Outputs in results/."
