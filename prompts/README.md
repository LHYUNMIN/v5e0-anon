# Prompt IDs

Drafter training and evaluation use disjoint prompt sets drawn from LLaVA-Instruct-150K.

| Split | Seed | Count | How to reproduce |
|---|---|---|---|
| Train (drafter data collection) | 43 | 200 prompts × 32 positions = 6,400 records | `random.Random(43).shuffle(rows)`, take first 200 records with `os.path.exists(image_path)` |
| Held-out evaluation | 999 | 30 prompts | `random.Random(999).shuffle(rows)`, take prompts not in the train split |

Both seeds and the JSONL line ordering are deterministic given `data/llava_messages_100k.jsonl` produced by `scripts/prepare_data.sh`. We do not check in the exact JSON files here because they contain image paths that depend on the user's local data layout; reproducing the splits from the seeds is trivial.

For the benchmark task evaluations (TextVQA, OCRBench, DocVQA, ChartQA, VQAv2, MMBench, MMMU) we use HuggingFace `datasets` streaming with the first n samples per split — see the `src/measurements/*.py` scripts for the exact dataset names and splits.
