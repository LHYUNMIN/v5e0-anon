#!/usr/bin/env bash
#
# Prepare COCO 2017 train images + LLaVA-Instruct-150K messages for V5e-0 training.
#
# After this script completes:
#   data/coco/train2017/<*.jpg>           (118k COCO train images)
#   data/llava_messages_100k.jsonl        (LLaVA-Instruct-150K messages, JSONL)
#
# Total disk: ~20 GB.

set -e
mkdir -p data/coco data/

# ---- COCO 2017 train images ----
if [ ! -d data/coco/train2017 ]; then
  echo "[prepare_data] Downloading COCO 2017 train images..."
  cd data/coco
  wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip
  unzip -q train2017.zip
  rm train2017.zip
  cd ../..
  echo "[prepare_data] COCO 2017 train images: data/coco/train2017/"
else
  echo "[prepare_data] data/coco/train2017/ already exists, skipping COCO download."
fi

# ---- LLaVA-Instruct-150K messages ----
# We provide the JSONL via the LLaVA-Instruct-150K HuggingFace dataset.
# Each line is: {"messages": [{"role": "user", "content": [{"type":"image","image":"<absolute_path>"}, {"type":"text","text":"..."}]}, ...]}
# Image paths in the JSONL must be absolute paths into data/coco/train2017/.
if [ ! -f data/llava_messages_100k.jsonl ]; then
  echo "[prepare_data] Building data/llava_messages_100k.jsonl from HuggingFace..."
  python -c "
import json, os
from datasets import load_dataset
COCO_DIR = os.path.abspath('data/coco/train2017')
ds = load_dataset('liuhaotian/LLaVA-Instruct-150K', split='train', streaming=True)
out_path = 'data/llava_messages_100k.jsonl'
n = 0
with open(out_path, 'w') as f:
    for s in ds:
        if n >= 100000: break
        img_name = s.get('image')
        if not img_name: continue
        img_path = os.path.join(COCO_DIR, img_name)
        if not os.path.exists(img_path): continue
        convs = s.get('conversations', [])
        if not convs: continue
        # Take the first human turn as the prompt.
        prompt_text = None
        for c in convs:
            if c.get('from') == 'human':
                txt = c.get('value', '').replace('<image>','').strip()
                if txt:
                    prompt_text = txt
                    break
        if not prompt_text: continue
        msg = {'messages': [{'role':'user','content':[
            {'type':'image','image': img_path},
            {'type':'text', 'text':  prompt_text},
        ]}]}
        f.write(json.dumps(msg) + '\n')
        n += 1
print(f'[prepare_data] wrote {n} JSONL records to {out_path}')
"
else
  echo "[prepare_data] data/llava_messages_100k.jsonl already exists, skipping."
fi

echo "[prepare_data] Done."
