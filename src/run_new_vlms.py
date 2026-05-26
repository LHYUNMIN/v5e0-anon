#!/usr/bin/env python3
"""Run V5e-0 training + measurement on additional VLMs (Phi-3.5-V, MiniCPM-V).

Adds 5th, 6th VLMs to the paper's cross-architecture coverage.

Usage:
  python run_new_vlms.py --vlm phi-3.5-v   --gpu 0 --save_ckpt ./checkpoints/v5e0_phi35v.pt
  python run_new_vlms.py --vlm minicpm-v   --gpu 0 --save_ckpt ./checkpoints/v5e0_minicpmv.pt
"""
import os, sys, json, gc, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (
    V5e0_Cont, V5e0_Cont2, MAX_TOKENS, N_TEST, N_COLLECT, GEN_LEN, EPOCHS, M, K,
    to_tuple_kv, kv_to_cache, load_llava_prompts,
    build_tree_d3, build_attn_mask_and_pos, prune_kv, measure_d3,
)


# Additional VLM configs
NEW_VLM_CONFIGS = {
    "phi-3.5-v":    {"model_id": "microsoft/Phi-3.5-vision-instruct",
                     "class": "phi3v",     "alpha": 30.0, "beta": 30.0},
    "minicpm-v":    {"model_id": "openbmb/MiniCPM-V-2_6",
                     "class": "minicpm",   "alpha": 30.0, "beta": 30.0},
    "llava-onevision": {"model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
                        "class": "llava-ov", "alpha": 30.0, "beta": 30.0},
    "llava-ov-0.5b":   {"model_id": "lmms-lab/llava-onevision-qwen2-0.5b-ov",
                        "class": "llava-ov", "alpha": 30.0, "beta": 30.0},
    "internvl-1b":     {"model_id": "OpenGVLab/InternVL2-1B",
                        "class": "internvl", "alpha": 30.0, "beta": 30.0},
    "internvl-2b":     {"model_id": "OpenGVLab/InternVL2-2B",
                        "class": "internvl", "alpha": 30.0, "beta": 30.0},
    "llava-next-vicuna": {"model_id": "llava-hf/llava-v1.6-vicuna-7b-hf",
                          "class": "llava-next", "alpha": 30.0, "beta": 30.0},
    "idefics2":     {"model_id": "HuggingFaceM4/idefics2-8b",
                     "class": "idefics2", "alpha": 30.0, "beta": 30.0},
    "qwen2.5-vl":   {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                     "class": "qwen25vl", "alpha": 30.0, "beta": 30.0},
    "paligemma":    {"model_id": "google/paligemma-3b-mix-448",
                     "class": "paligemma", "alpha": 30.0, "beta": 30.0},
    "paligemma2-3b": {"model_id": "google/paligemma2-3b-mix-448",
                      "class": "paligemma", "alpha": 30.0, "beta": 30.0},
    "paligemma2-10b": {"model_id": "google/paligemma2-10b-mix-448",
                       "class": "paligemma", "alpha": 30.0, "beta": 30.0},
    "qwen3-vl-4b":   {"model_id": "Qwen/Qwen3-VL-4B-Instruct",
                      "class": "qwen3vl", "alpha": 30.0, "beta": 30.0},
    "qwen3-vl-8b":   {"model_id": "Qwen/Qwen3-VL-8B-Instruct",
                      "class": "qwen3vl", "alpha": 30.0, "beta": 30.0},
    "glm-4v-9b":     {"model_id": "THUDM/glm-4v-9b",
                      "class": "glm4v", "alpha": 30.0, "beta": 30.0},
    "internvl3.5-8b":{"model_id": "OpenGVLab/InternVL3_5-8B",
                      "class": "internvl3", "alpha": 30.0, "beta": 30.0},
    "gemma-3-12b":   {"model_id": "google/gemma-3-12b-it",
                      "class": "gemma3", "alpha": 30.0, "beta": 30.0},
    "pixtral-12b":   {"model_id": "mistral-community/pixtral-12b",
                      "class": "pixtral", "alpha": 30.0, "beta": 30.0},
}


def prepare_gemma3_inputs(processor, image_path, prompt_text, device):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    return inputs.to(device, dtype=torch.bfloat16) if hasattr(inputs, 'to') else \
           {k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device))
            for k, v in inputs.items()}


def prepare_pixtral_inputs(processor, image_path, prompt_text, device):
    image = Image.open(image_path).convert("RGB")
    chat = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "content": prompt_text}]}]
    text = processor.apply_chat_template(chat, add_generation_prompt=True)
    return processor(text=text, images=[image], return_tensors="pt").to(device)


def prepare_qwen3vl_inputs(processor, image_path, prompt_text, device):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)


def _glm4v_batch_encode_plus(self, batch_ids, padding=None, truncation=None,
                             max_length=None, return_tensors=None,
                             is_split_into_words=True, add_special_tokens=False,
                             **kwargs):
    """Replacement for tokenizer.batch_encode_plus (removed in transformers >=5).

    Accepts pre-tokenized integer ID sequences (as the GLM-4V chat template emits)
    and returns the standard BatchEncoding dict with input_ids and attention_mask.
    """
    if isinstance(batch_ids[0], int):
        batch_ids = [batch_ids]
    max_len = max(len(ids) for ids in batch_ids)
    pad_id = getattr(self, "pad_token_id", None) or 0
    input_ids, attn_mask = [], []
    for ids in batch_ids:
        pad_n = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_n)
        attn_mask.append([1] * len(ids) + [0] * pad_n)
    if return_tensors == "pt":
        import torch as _t
        return {
            "input_ids": _t.tensor(input_ids, dtype=_t.long),
            "attention_mask": _t.tensor(attn_mask, dtype=_t.long),
        }
    return {"input_ids": input_ids, "attention_mask": attn_mask}


def prepare_glm4v_inputs(tokenizer, image_path, prompt_text, device):
    image = Image.open(image_path).convert("RGB")
    # transformers >=5 removed batch_encode_plus; inject our shim if missing
    if not hasattr(tokenizer.__class__, "batch_encode_plus") or \
       "_glm4v_batch_encode_plus" not in str(getattr(tokenizer.__class__, "batch_encode_plus", "")):
        tokenizer.__class__.batch_encode_plus = _glm4v_batch_encode_plus
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": prompt_text}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt",
        return_dict=True)
    if isinstance(inputs, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    return inputs.to(device)


def prepare_idefics2_inputs(processor, image_path, prompt_text, device):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt_text}]}]
    txt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(images=image, text=txt, return_tensors="pt").to(device)


def prepare_paligemma_inputs(processor, image_path, prompt_text, device):
    image = Image.open(image_path).convert("RGB")
    # PaliGemma expects <image> token explicitly in prompt
    full_prompt = f"<image> {prompt_text}"
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
    # PaliGemma uses bfloat16 pixel_values
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    return inputs


def prepare_phi3v_inputs(processor, image_path, prompt_text, device):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": f"<|image_1|>\n{prompt_text}"}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to(device)
    return inputs


def prepare_internvl_inputs(tokenizer, image_path, prompt_text, device):
    """InternVL2 input preparation. Uses fixed 448x448 single tile.
    Returns dict with pixel_values, input_ids, attention_mask, image_flags.
    """
    import torchvision.transforms as T
    IMG_MEAN = (0.485, 0.456, 0.406); IMG_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device, dtype=torch.bfloat16)

    # 448x448 with patch=14 and downsample_ratio=0.5 → 32×32/4 = 256 image tokens
    num_image_token = 256
    IMG_CONTEXT = "<IMG_CONTEXT>"
    IMG_START = "<img>"
    IMG_END = "</img>"
    image_block = IMG_START + IMG_CONTEXT * num_image_token + IMG_END

    # InternVL2 chat template (InternLM2-style)
    template = (
        f"<|im_start|>user\n{image_block}\n{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    model_inputs = tokenizer(template, return_tensors="pt").to(device)

    return {
        "pixel_values": pixel_values,
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "image_flags": torch.tensor([1], dtype=torch.long, device=device),
    }


def prepare_llava_ov_inputs(processor, image_path, prompt_text, device):
    image = Image.open(image_path).convert("RGB")
    conversation = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt_text}]}]
    txt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return processor(images=image, text=txt, return_tensors="pt").to(device)


def setup_new_vlm(vlm_name):
    """Load a new VLM and return (model, processor, prep_inputs_fn, is_qwen-like)."""
    cfg = NEW_VLM_CONFIGS[vlm_name]
    cls = cfg["class"]

    if cls == "phi3v":
        from transformers import AutoModelForCausalLM, AutoProcessor
        # Phi-3.5-V doesn't support SDPA; use eager attention
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="eager",
        ).to("cuda").eval()
        proc = AutoProcessor.from_pretrained(cfg["model_id"], trust_remote_code=True, num_crops=4)
        prep = lambda s: prepare_phi3v_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = False
    elif cls == "llava-ov":
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            ignore_mismatched_sizes=True,
        ).to("cuda").eval()
        proc = AutoProcessor.from_pretrained(cfg["model_id"])
        prep = lambda s: prepare_llava_ov_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = False
    elif cls == "internvl":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        proc = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        # InternVL2 requires img_context_token_id to be set for image-token lookup
        model.img_context_token_id = proc.convert_tokens_to_ids("<IMG_CONTEXT>")
        prep = lambda s: prepare_internvl_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = False
    elif cls == "minicpm":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa"
        ).to("cuda").eval()
        proc = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        prep = None
        is_qwen = False
    elif cls == "llava-next":
        from transformers import LlavaNextForConditionalGeneration, AutoProcessor
        model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        proc = AutoProcessor.from_pretrained(cfg["model_id"])
        prep = lambda s: prepare_llava_ov_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = False
    elif cls == "idefics2":
        from transformers import Idefics2ForConditionalGeneration, AutoProcessor
        model = Idefics2ForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        proc = AutoProcessor.from_pretrained(cfg["model_id"])
        prep = lambda s: prepare_idefics2_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = False
    elif cls == "qwen25vl":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa"
        ).to("cuda").eval()
        proc = AutoProcessor.from_pretrained(cfg["model_id"], trust_remote_code=True)
        # Use Qwen-VL-style preparation (compare_vlm_ssd_infer.prepare_vlm_inputs)
        sys.path.insert(0, "./vendor")
        from compare_vlm_ssd_infer import prepare_vlm_inputs
        prep = lambda s: prepare_vlm_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = True   # Qwen-style 3D position ids
    elif cls == "paligemma":
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        proc = AutoProcessor.from_pretrained(cfg["model_id"])
        prep = lambda s: prepare_paligemma_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = False
    elif cls == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa"
        ).to("cuda").eval()
        proc = AutoProcessor.from_pretrained(cfg["model_id"], trust_remote_code=True)
        prep = lambda s: prepare_qwen3vl_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = True   # Qwen-style 3D position ids
    elif cls == "internvl3":
        from transformers import AutoModel, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        # Patch missing post_init attrs + route AR/tree calls to language_model
        InternVLCls = get_class_from_dynamic_module(
            "modeling_internvl_chat.InternVLChatModel", cfg["model_id"])
        if not getattr(InternVLCls, "_v5e0_patched", False):
            _orig_init = InternVLCls.__init__
            def _patched_init(self, *args, **kwargs):
                _orig_init(self, *args, **kwargs)
                for attr, default in [("all_tied_weights_keys", {}),
                                      ("_tp_plan", {}), ("_ep_plan", {}), ("_pp_plan", {}),
                                      ("_keep_in_fp32_modules", set()),
                                      ("_keep_in_fp32_modules_strict", set()),
                                      ("_no_split_modules", set())]:
                    if not hasattr(self, attr): setattr(self, attr, default)
            InternVLCls.__init__ = _patched_init
            # Wrap forward: if no pixel_values, bypass ViT and route directly
            # to self.language_model (Qwen-style transformer that handles
            # cache_position, DynamicCache, and 4D attention masks natively).
            _orig_fwd = InternVLCls.forward
            def _patched_forward(self, pixel_values=None, input_ids=None,
                                  attention_mask=None, position_ids=None,
                                  image_flags=None, past_key_values=None,
                                  labels=None, use_cache=None,
                                  output_attentions=None, output_hidden_states=None,
                                  return_dict=None, **_extra):
                if pixel_values is None:
                    # AR / tree forward: skip ViT, go straight to LM
                    return self.language_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache, output_hidden_states=output_hidden_states,
                        return_dict=return_dict if return_dict is not None else True,
                        **_extra,
                    )
                return _orig_fwd(
                    self, pixel_values=pixel_values, input_ids=input_ids,
                    attention_mask=attention_mask, position_ids=position_ids,
                    image_flags=image_flags, past_key_values=past_key_values,
                    labels=labels, use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            InternVLCls.forward = _patched_forward
            InternVLCls._v5e0_patched = True
        model = AutoModel.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        proc = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        model.img_context_token_id = proc.convert_tokens_to_ids("<IMG_CONTEXT>")
        prep = lambda s: prepare_internvl_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = False
    elif cls == "gemma3":
        from transformers import Gemma3ForConditionalGeneration, AutoProcessor
        # transformers 5.5.4 + torch 2.5: sliding-window mask uses or_mask_function
        # which requires torch>=2.6; eager attention bypasses that code path.
        model = Gemma3ForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        ).to("cuda").eval()
        proc = AutoProcessor.from_pretrained(cfg["model_id"])
        prep = lambda s: prepare_gemma3_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = False
    elif cls == "pixtral":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).to("cuda").eval()
        proc = AutoProcessor.from_pretrained(cfg["model_id"])
        prep = lambda s: prepare_pixtral_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = False
    elif cls == "glm4v":
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        # GLM-4V's custom modeling_chatglm.py was written for transformers <5;
        # we patch the missing config attr and inject post_init-equivalent attrs.
        cfg_obj = AutoConfig.from_pretrained(cfg["model_id"], trust_remote_code=True)
        if not hasattr(cfg_obj, "max_length"):
            cfg_obj.max_length = getattr(cfg_obj, "seq_length", 8192)
        # Pre-resolve the remote class and patch __init__ to mimic post_init() attrs
        ChatGLMCls = get_class_from_dynamic_module(
            "modeling_chatglm.ChatGLMForConditionalGeneration",
            cfg["model_id"])
        if not getattr(ChatGLMCls, "_v5e0_patched", False):
            _orig_init = ChatGLMCls.__init__
            def _patched_init(self, config, *args, **kwargs):
                _orig_init(self, config, *args, **kwargs)
                if not hasattr(self, "all_tied_weights_keys"):
                    self.all_tied_weights_keys = {}
                if not hasattr(self, "_tp_plan"):
                    self._tp_plan = {}
                if not hasattr(self, "_ep_plan"):
                    self._ep_plan = {}
                if not hasattr(self, "_pp_plan"):
                    self._pp_plan = {}
                if not hasattr(self, "_keep_in_fp32_modules"):
                    self._keep_in_fp32_modules = set()
                if not hasattr(self, "_keep_in_fp32_modules_strict"):
                    self._keep_in_fp32_modules_strict = set()
                if not hasattr(self, "_no_split_modules"):
                    self._no_split_modules = set()
            ChatGLMCls.__init__ = _patched_init
            # Bypass top-level forward: convert (a) 4D float tree-attn mask → bool
            # full_attention_mask, (b) DynamicCache → legacy tuple, then call
            # self.transformer directly to thread full_attention_mask through.
            from transformers.modeling_outputs import CausalLMOutputWithPast as _COWP
            def _patched_forward(self, input_ids=None, images=None,
                                 position_ids=None, attention_mask=None,
                                 past_key_values=None, inputs_embeds=None,
                                 labels=None, use_cache=None,
                                 output_attentions=None, output_hidden_states=None,
                                 return_dict=None, return_last_logit=False,
                                 **_ignored):
                full_attention_mask = None
                if attention_mask is not None and attention_mask.dtype != torch.bool \
                        and attention_mask.dim() == 4:
                    # Tree mask: additive float (-inf or 0) on [1, 1, Q, KV]
                    full_attention_mask = (attention_mask < -1e9)
                    attention_mask = None
                # DynamicCache → legacy tuple-of-tuples (GLM-4V indexes by layer)
                if past_key_values is not None:
                    from transformers.cache_utils import DynamicCache as _DC
                    if isinstance(past_key_values, _DC):
                        if hasattr(past_key_values, "key_cache") and \
                                len(past_key_values.key_cache) > 0:
                            past_key_values = tuple(
                                (k, v) for k, v in zip(past_key_values.key_cache,
                                                       past_key_values.value_cache))
                        elif hasattr(past_key_values, "layers") and \
                                len(past_key_values.layers) > 0:
                            past_key_values = tuple(
                                (l.keys, l.values) for l in past_key_values.layers)
                        else:
                            past_key_values = None
                use_cache = use_cache if use_cache is not None else self.config.use_cache
                if return_dict is None:
                    return_dict = self.config.use_return_dict
                transformer_outputs = self.transformer(
                    input_ids=input_ids, images=images, position_ids=position_ids,
                    attention_mask=attention_mask,
                    full_attention_mask=full_attention_mask,
                    past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states = transformer_outputs[0] if not return_dict \
                    else transformer_outputs.last_hidden_state
                if return_last_logit:
                    hidden_states = hidden_states[:, -1:]
                lm_logits = self.transformer.output_layer(hidden_states)
                if not return_dict:
                    return (lm_logits,) + transformer_outputs[1:]
                return _COWP(
                    loss=None, logits=lm_logits,
                    past_key_values=transformer_outputs.past_key_values,
                    hidden_states=transformer_outputs.hidden_states,
                    attentions=transformer_outputs.attentions,
                )
            ChatGLMCls.forward = _patched_forward
            ChatGLMCls._v5e0_patched = True
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], config=cfg_obj, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True,
            attn_implementation="eager"
        ).to("cuda").eval()
        proc = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        prep = lambda s: prepare_glm4v_inputs(proc, s["image"], s["prompt"], "cuda")
        is_qwen = False
    return model, proc, prep, is_qwen, cls


def main(vlm_name, save_ckpt, n_test):
    cfg = NEW_VLM_CONFIGS[vlm_name]
    print(f"[NEW VLM: {vlm_name}, model_id: {cfg['model_id']}]\n", flush=True)

    sys.path.insert(0, "./vendor")
    from runtime_env import strip_user_site_packages; strip_user_site_packages()
    from transformers.cache_utils import DynamicCache

    model, processor, prepare_inputs, is_qwen, cls = setup_new_vlm(vlm_name)
    if prepare_inputs is None:
        print(f"  ERROR: {cls} model uses non-standard API; skipping for now.")
        return

    # Detect hidden_dim, vocab (handle multiple config nesting conventions)
    tc = getattr(model.config, "text_config", None) \
         or getattr(model.config, "llm_config", None) \
         or model.config
    vocab = getattr(tc, "vocab_size", None) or getattr(model.config, "vocab_size", None)
    hidden_dim = getattr(tc, "hidden_size", None) or getattr(model.config, "hidden_size", None)
    print(f"  hidden_dim={hidden_dim}, vocab={vocab}\n", flush=True)

    emb = model.get_input_embeddings()
    if emb.weight.shape[0] > vocab:
        new = nn.Embedding(vocab, emb.weight.shape[1], device="cuda", dtype=emb.weight.dtype)
        new.weight.data.copy_(emb.weight.data[:vocab])
        model.set_input_embeddings(new); emb = new
    lm_head = model.get_output_embeddings()
    # GLM-4V exposes the LM head at model.transformer.output_layer
    if lm_head is None and hasattr(model, "transformer") \
            and hasattr(model.transformer, "output_layer"):
        lm_head = model.transformer.output_layer
    # InternVL3.x exposes LM head at model.language_model.lm_head
    if lm_head is None and hasattr(model, "language_model"):
        lm_head = model.language_model.get_output_embeddings() \
            or getattr(model.language_model, "lm_head", None)
    for p in model.parameters(): p.requires_grad_(False)
    root_emb_table = emb.weight.detach().to(torch.float32)

    # ---------- Data collection ----------
    print(f"[collect {N_COLLECT} prompts × {GEN_LEN}]")
    samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        N_COLLECT, seed=43)

    h_t_list = []; root_id_list = []; cont_target_list = []
    prompt_idx_list = []; pos_list = []
    used = 0; t0 = time.time()
    for s in samples:
        try:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            inputs = prepare_inputs(s)
            with torch.no_grad():
                out = model(**inputs, past_key_values=DynamicCache(),
                            use_cache=True, output_hidden_states=True, return_dict=True)
        except Exception as e:
            print(f"  skip prompt (load error): {e}", flush=True); continue
        cur_kv = to_tuple_kv(out.past_key_values)
        h_t = out.hidden_states[-1][0, -1, :].float()
        v_logits = out.logits[0, -1, :].float()
        for pos in range(GEN_LEN):
            v_root = int(v_logits.argmax().item())
            try:
                with torch.no_grad():
                    tok = torch.tensor([[v_root]], device="cuda")
                    cur_len = cur_kv[0][0].shape[2]
                    pid = torch.tensor([[cur_len]], device="cuda")
                    cp = torch.tensor([cur_len], device="cuda")
                    out_step = model(input_ids=tok, past_key_values=kv_to_cache(cur_kv),
                                     position_ids=pid, cache_position=cp,
                                     use_cache=True, output_hidden_states=True, return_dict=True)
                    v_cont = int(out_step.logits[0, -1, :].argmax().item())
            except Exception as e:
                if used == 0:  # print only for first sample to debug
                    import traceback
                    print(f"  inner loop break at pos {pos}: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                break
            h_t_list.append(h_t.detach().cpu())
            root_id_list.append(v_root)
            cont_target_list.append(v_cont)
            prompt_idx_list.append(used); pos_list.append(pos)
            h_t = out_step.hidden_states[-1][0, -1, :].float()
            cur_kv = to_tuple_kv(out_step.past_key_values)
            v_logits = out_step.logits[0, -1, :].float()
        used += 1
        if used % 50 == 0:
            print(f"  {used}/{N_COLLECT}, n={len(h_t_list)}, "
                  f"elapsed {(time.time()-t0)/60:.1f}min", flush=True)
        if used >= N_COLLECT: break

    h_t_tensor = torch.stack(h_t_list).to(torch.float32)
    root_id_t = torch.tensor(root_id_list, dtype=torch.long)
    cont_target_t = torch.tensor(cont_target_list, dtype=torch.long)
    prompt_idx_arr = np.array(prompt_idx_list)
    pos_arr = np.array(pos_list)
    print(f"  collected n={h_t_tensor.shape[0]}\n")

    # ---------- Train V5e-0-Cont + Cont2 (sequential) ----------
    from collections import defaultdict
    rng = np.random.RandomState(0)
    unique = np.unique(prompt_idx_arr)
    n_eval_p = max(1, int(round(len(unique) * 0.20)))
    eval_prompts = set(rng.choice(unique, n_eval_p, replace=False).tolist())
    train_idx = np.array([i for i in range(len(prompt_idx_arr))
                           if prompt_idx_arr[i] not in eval_prompts])
    train_by_p = defaultdict(list)
    for i in train_idx: train_by_p[prompt_idx_arr[i]].append(int(i))
    train_prompts_list = list(train_by_p.keys())

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    n1_cont = V5e0_Cont(dim=hidden_dim, alpha_init=cfg["alpha"]).to("cuda")
    opt = torch.optim.AdamW(n1_cont.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"[train V5e-0-Cont {EPOCHS} epochs]")
    t0 = time.time()
    for ep in range(EPOCHS):
        n1_cont.train()
        random.shuffle(train_prompts_list)
        for bstart in range(0, len(train_prompts_list), 8):
            bp = train_prompts_list[bstart:bstart + 8]
            opt.zero_grad()
            total_loss = 0; total_n = 0
            for p in bp:
                recs = train_by_p[p]
                h = h_t_tensor[recs].to("cuda")
                re = root_emb_table[root_id_t[recs]].to("cuda")
                tg = cont_target_t[recs].to("cuda")
                z = n1_cont(h, re)
                logits = lm_head(z.to(torch.bfloat16)).float()
                loss = F.cross_entropy(logits, tg, reduction="sum")
                total_loss = total_loss + loss; total_n += len(recs)
            if total_n > 0:
                (total_loss / total_n).backward(); opt.step()
    n1_cont.eval()
    for p in n1_cont.parameters(): p.requires_grad_(False)
    print(f"  trained in {(time.time()-t0)/60:.1f}min\n")

    # Cont2 tuples
    cont2_recs = []
    for i in range(len(prompt_idx_arr) - 1):
        if prompt_idx_arr[i+1] == prompt_idx_arr[i] and pos_arr[i+1] == pos_arr[i] + 1:
            cont2_recs.append({
                "h_idx": i, "root_id": int(root_id_t[i]),
                "cont_id": int(cont_target_t[i]),
                "cont2_target": int(cont_target_t[i+1]),
                "prompt_idx": int(prompt_idx_arr[i]),
            })
    train_c2 = [r for r in cont2_recs if r["prompt_idx"] not in eval_prompts]
    train_h_idx = torch.tensor([r["h_idx"] for r in train_c2], dtype=torch.long)
    train_root = torch.tensor([r["root_id"] for r in train_c2], dtype=torch.long)
    train_cont = torch.tensor([r["cont_id"] for r in train_c2], dtype=torch.long)
    train_cont2 = torch.tensor([r["cont2_target"] for r in train_c2], dtype=torch.long)
    train_p2 = torch.tensor([r["prompt_idx"] for r in train_c2], dtype=torch.long)
    train_by_p2 = defaultdict(list)
    for i, p in enumerate(train_p2.tolist()): train_by_p2[p].append(i)
    train_prompts2 = list(train_by_p2.keys())

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    n1_cont2 = V5e0_Cont2(dim=hidden_dim, alpha_init=cfg["alpha"],
                          beta_init=cfg["beta"]).to("cuda")
    opt2 = torch.optim.AdamW(n1_cont2.parameters(), lr=5e-4, weight_decay=0.01)
    print(f"[train V5e-0-Cont2 {EPOCHS} epochs]")
    t0 = time.time()
    for ep in range(EPOCHS):
        n1_cont2.train()
        random.shuffle(train_prompts2)
        for bstart in range(0, len(train_prompts2), 8):
            bp = train_prompts2[bstart:bstart + 8]
            opt2.zero_grad()
            total_loss = 0; total_n = 0
            for p in bp:
                idx_list = train_by_p2[p]
                idx = torch.tensor(idx_list, dtype=torch.long)
                h = h_t_tensor[train_h_idx[idx]].to("cuda")
                re = root_emb_table[train_root[idx]].to("cuda")
                ce = root_emb_table[train_cont[idx]].to("cuda")
                tg = train_cont2[idx].to("cuda")
                z = n1_cont2(h, re, ce)
                logits = lm_head(z.to(torch.bfloat16)).float()
                loss = F.cross_entropy(logits, tg, reduction="sum")
                total_loss = total_loss + loss; total_n += len(idx_list)
            if total_n > 0:
                (total_loss / total_n).backward(); opt2.step()
    n1_cont2.eval()
    for p in n1_cont2.parameters(): p.requires_grad_(False)
    print(f"  trained in {(time.time()-t0)/60:.1f}min\n")

    # ---------- Walltime measurement ----------
    test_samples = load_llava_prompts(
        "./data/llava_messages_100k.jsonl",
        n_test, seed=999)

    print(f"[walltime test M={M}, K={K}, {n_test} prompts × {MAX_TOKENS}]")
    ar_list, ssd_list, n_d2, n_d3, n_rounds, n_tokens = measure_d3(
        model, n1_cont, n1_cont2, lm_head, root_emb_table, prepare_inputs,
        is_qwen, test_samples, M, K, MAX_TOKENS)
    sp = [s/a for a, s in zip(ar_list, ssd_list)]
    ar_a = np.array(ar_list); ssd_a = np.array(ssd_list); sp_a = np.array(sp)
    d3_rate = n_d3 / n_rounds if n_rounds > 0 else 0
    tpi = n_tokens / n_rounds if n_rounds > 0 else 0

    print(f"\n{'='*70}")
    print(f"{vlm_name} V5e-0 single-root depth-3 (M={M}, K={K})")
    print(f"{'='*70}")
    print(f"  AR:    {ar_a.mean():.2f} ± {ar_a.std():.2f} t/s")
    print(f"  SSD:   {ssd_a.mean():.2f} ± {ssd_a.std():.2f} t/s")
    print(f"  sp:    {sp_a.mean():.3f} ± {sp_a.std():.3f}")
    print(f"  TPI {tpi:.2f}, d3 {d3_rate:.3f}")

    out = {
        "vlm": vlm_name, "model_id": cfg["model_id"], "hidden_dim": hidden_dim,
        "M": M, "K": K, "n_input_tokens": 1 + M + M*K,
        "ar_tps":  {"mean": float(ar_a.mean()),  "std": float(ar_a.std())},
        "ssd_tps": {"mean": float(ssd_a.mean()), "std": float(ssd_a.std())},
        "sp": {"mean": float(sp_a.mean()), "std": float(sp_a.std()),
               "median": float(np.median(sp_a)),
               "min": float(sp_a.min()), "max": float(sp_a.max())},
        "tpi": tpi, "d3_rate": d3_rate,
        "n_prompts": len(ar_list),
        "raw": {"ar": ar_list, "ssd": ssd_list, "sp": sp},
    }
    save_path = f"/tmp/path1_v5e0_d3_4vlm_{vlm_name}.json"
    with open(save_path, "w") as f: json.dump(out, f, indent=2)
    print(f"\n[saved {save_path}]")

    if save_ckpt:
        torch.save({
            "vlm": vlm_name, "hidden_dim": hidden_dim,
            "alpha": cfg["alpha"], "beta": cfg["beta"],
            "W_Q1": n1_cont.Q_proj.weight.detach().cpu(),
            "W_Q1_bias": n1_cont.Q_proj.bias.detach().cpu(),
            "W_Q2": n1_cont2.Q_proj.weight.detach().cpu(),
            "W_Q2_bias": n1_cont2.Q_proj.bias.detach().cpu(),
        }, save_ckpt)
        print(f"[saved drafter ckpt {save_ckpt}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", required=True)   # any key in NEW_VLM_CONFIGS
    p.add_argument("--gpu", default="0")
    p.add_argument("--save_ckpt", default=None)
    p.add_argument("--n_test", type=int, default=30)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args.vlm, args.save_ckpt, args.n_test)
