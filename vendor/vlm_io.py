"""Minimal VLM input preparation helper.

Anonymized standalone version of prepare_vlm_inputs used by V5e-0 training/measurement.
Original lived alongside a heavier inference scaffold; here we keep only the helper
that `run.py` and ablation scripts actually call.
"""
from typing import Any, Dict, Optional

import torch
from qwen_vl_utils import process_vision_info


def prepare_vlm_inputs(processor, image_path: Optional[str], prompt: str, device) -> Dict[str, Any]:
    """Build a chat-template + image batch for a Qwen-VL-style processor.

    Args:
        processor: HuggingFace AutoProcessor for the verifier VLM (Qwen-VL family).
        image_path: filesystem path to the image, or None for text-only.
        prompt:    user text content.
        device:    target torch device.
    """
    content = []
    if image_path:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
