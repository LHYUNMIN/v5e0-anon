#!/usr/bin/env bash
#
# Train V5e-0 drafters across all evaluated VLMs.
# Each VLM takes ~3-5 minutes on a single A100; total ~1 hour for 15 VLMs.

set -e
mkdir -p checkpoints

VLMS=(
  qwen2-2b
  qwen2-7b
  qwen2.5-7b
  llava-1.5-7b
  llava-1.6-mistral-7b
  llava-1.6-vicuna-7b
  llava-onevision-7b
  idefics2-8b
  paligemma-3b
  paligemma2-3b
  paligemma2-10b
)

for vlm in "${VLMS[@]}"; do
  echo "==================== $vlm ===================="
  python src/run.py --vlm "$vlm" --gpu 0
done

# Newer VLMs (separate entry point due to custom modeling code)
NEW_VLMS=(
  qwen3-vl-4b
  glm-4v-9b
  internvl3.5
  pixtral-12b
)
for vlm in "${NEW_VLMS[@]}"; do
  echo "==================== $vlm ===================="
  python src/run_new_vlms.py --vlm "$vlm" --gpu 0
done

echo "All 15 VLMs trained. Checkpoints in checkpoints/."
