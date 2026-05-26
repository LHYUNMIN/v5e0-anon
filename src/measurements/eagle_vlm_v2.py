"""EAGLE-VLM Light v2: 1-layer transformer drafter with PROPER image+text data collection.

Fixes v1's bug where image paths from message content weren't extracted.
Uses same data loader as V5e-0 (run.py load_llava_messages function).
"""
import os, sys, json, gc, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, './v5e0')
sys.path.insert(0, './vendor')
from runtime_env import strip_user_site_packages; strip_user_site_packages()
import numpy as np
import random
from datasets import load_dataset
from transformers.cache_utils import DynamicCache
from run import (VLM_CONFIGS, to_tuple_kv, kv_to_cache,
                 build_tree_d3, build_attn_mask_and_pos, prune_kv,
                 prepare_qwen_inputs, prepare_llava_inputs,
                 N_COLLECT, GEN_LEN, EPOCHS, M, K)
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, LlavaForConditionalGeneration


def load_llava_msgs(path, n, seed=43):
    """Same as run.py — extract (image, prompt) pairs with existing image files."""
    rng = random.Random(seed)
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            msgs = d['messages']
            img = next((c['image'] for c in msgs[0]['content'] if c['type'] == 'image'), None)
            txt = next((c['text']  for c in msgs[0]['content'] if c['type'] == 'text'),  None)
            if img and txt and os.path.exists(img):
                rows.append({"image": img, "prompt": txt})
                if len(rows) >= n * 3: break
    rng.shuffle(rows)
    return rows[:n]


class EagleVLMDrafter(nn.Module):
    """1-layer transformer block (self-attn + FFN). Applied to short sequence ending in h_t."""
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, bias=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, seq):
        L = seq.shape[1]
        cmask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        x = self.ln1(seq)
        attn_out, _ = self.attn(x, x, x, attn_mask=cmask, need_weights=False)
        seq = seq + attn_out
        x = self.ln2(seq)
        seq = seq + self.ffn(x)
        return seq[:, -1, :]


def run_vlm(vlm_name):
    print(f'\n=========== {vlm_name} ===========', flush=True)
    cfg = VLM_CONFIGS[vlm_name]
    is_qwen = cfg['model_class'] == 'qwen'
    if is_qwen:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16, attn_implementation='sdpa'
        ).to('cuda').eval()
        processor = AutoProcessor.from_pretrained(cfg['model_id'], trust_remote_code=True)
        prep = lambda s: prepare_qwen_inputs(processor, s['image'], s['prompt'], 'cuda')
    elif cfg['model_class'] == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained(
            cfg['model_id'], torch_dtype=torch.bfloat16, attn_implementation='sdpa'
        ).to('cuda').eval()
        processor = AutoProcessor.from_pretrained(cfg['model_id'])
        prep = lambda s: prepare_llava_inputs(processor, s['image'], s['prompt'], 'cuda')
    else:
        raise ValueError(f'Unsupported model class: {cfg["model_class"]}')
    D = getattr(model.config, 'hidden_size', None) or model.config.text_config.hidden_size
    emb = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    root_emb_table = emb.weight.detach().to(torch.float32)
    for p in model.parameters(): p.requires_grad_(False)
    print(f'[{vlm_name}] D={D}', flush=True)

    # ---- Collect data (PROPER image+text) ----
    print(f'[collect {N_COLLECT}×{GEN_LEN} (image+text)]', flush=True)
    samples = load_llava_msgs('./data/llava_messages_100k.jsonl', N_COLLECT, seed=43)
    print(f'  loaded {len(samples)} samples', flush=True)

    h_t_list, root_id_list, cont_target_list = [], [], []
    prompt_idx_list, pos_list = [], []
    used = 0; t0 = time.time()
    for s in samples:
        try:
            gc.collect(); torch.cuda.empty_cache()
            inputs = prep(s)
            with torch.no_grad():
                out = model(**inputs, past_key_values=DynamicCache(),
                            use_cache=True, output_hidden_states=True, return_dict=True)
        except Exception as e:
            continue
        cur_kv = to_tuple_kv(out.past_key_values)
        h_t = out.hidden_states[-1][0,-1,:].float()
        v_logits = out.logits[0,-1,:].float()
        for pos in range(GEN_LEN):
            v_root_top1 = int(v_logits.argmax().item())
            try:
                with torch.no_grad():
                    tok_inp = torch.tensor([[v_root_top1]], device='cuda')
                    cur_len = cur_kv[0][0].shape[2]
                    pid = torch.tensor([[cur_len]], device='cuda')
                    if is_qwen: pid = pid.unsqueeze(0).expand(3,1,-1)
                    cp = torch.tensor([cur_len], device='cuda')
                    o = model(input_ids=tok_inp, past_key_values=kv_to_cache(cur_kv),
                              position_ids=pid, cache_position=cp,
                              use_cache=True, output_hidden_states=True, return_dict=True)
                    v_cont_top1 = int(o.logits[0,-1,:].argmax().item())
            except Exception:
                break
            h_t_list.append(h_t.detach().cpu())
            root_id_list.append(v_root_top1)
            cont_target_list.append(v_cont_top1)
            prompt_idx_list.append(used); pos_list.append(pos)
            h_t = o.hidden_states[-1][0,-1,:].float()
            cur_kv = to_tuple_kv(o.past_key_values)
            v_logits = o.logits[0,-1,:].float()
        used += 1
        if used % 50 == 0:
            print(f'  collected {used}/{N_COLLECT}, n={len(h_t_list)}, {(time.time()-t0)/60:.1f}min', flush=True)
        if used >= N_COLLECT: break

    h_t_tensor = torch.stack(h_t_list).to(torch.float32)
    root_id_t = torch.tensor(root_id_list, dtype=torch.long)
    cont_target_t = torch.tensor(cont_target_list, dtype=torch.long)
    prompt_idx_arr = np.array(prompt_idx_list)
    pos_arr = np.array(pos_list)
    print(f'  collected n={h_t_tensor.shape[0]}', flush=True)

    # 80/20 split
    rng = np.random.RandomState(0)
    unique_prompts = np.unique(prompt_idx_arr)
    n_eval = max(1, int(round(len(unique_prompts) * 0.20)))
    eval_prompts = set(rng.choice(unique_prompts, n_eval, replace=False).tolist())
    train_idx = np.array([i for i in range(len(prompt_idx_arr)) if prompt_idx_arr[i] not in eval_prompts])
    eval_idx = np.array([i for i in range(len(prompt_idx_arr)) if prompt_idx_arr[i] in eval_prompts])
    from collections import defaultdict
    train_by_p = defaultdict(list)
    for i in train_idx: train_by_p[prompt_idx_arr[i]].append(int(i))
    train_prompts_list = list(train_by_p.keys())

    # ---- Train EAGLE drafter (Cont step) ----
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    drafter = EagleVLMDrafter(dim=D, n_heads=8).to('cuda')
    n_params = sum(p.numel() for p in drafter.parameters())
    print(f'  EAGLE drafter: {n_params/1e6:.1f}M params', flush=True)
    opt = torch.optim.AdamW(drafter.parameters(), lr=5e-4, weight_decay=0.01)
    print(f'[train Cont step, {EPOCHS} epochs]', flush=True)
    t0 = time.time()
    for ep in range(EPOCHS):
        drafter.train()
        random.shuffle(train_prompts_list)
        for bstart in range(0, len(train_prompts_list), 8):
            bp = train_prompts_list[bstart:bstart+8]
            opt.zero_grad()
            total_loss = 0; total_n = 0
            for p in bp:
                recs = train_by_p[p]
                h = h_t_tensor[recs].to('cuda')
                re = root_emb_table[root_id_t[recs]].to('cuda')
                tg = cont_target_t[recs].to('cuda')
                seq = torch.stack([re, h], dim=1)
                z = drafter(seq)
                logits = lm_head(z.to(torch.bfloat16)).float()
                loss = F.cross_entropy(logits, tg, reduction='sum')
                total_loss = total_loss + loss; total_n += len(recs)
            if total_n > 0:
                (total_loss / total_n).backward(); opt.step()
        if (ep+1) % 25 == 0:
            print(f'  ep {ep+1}', flush=True)
    drafter.eval()
    for p in drafter.parameters(): p.requires_grad_(False)
    print(f'  trained in {(time.time()-t0)/60:.1f}min', flush=True)

    # Eval Cont top-1
    if len(eval_idx) > 0:
        with torch.no_grad():
            h = h_t_tensor[eval_idx].to('cuda')
            re = root_emb_table[root_id_t[eval_idx]].to('cuda')
            seq = torch.stack([re, h], dim=1)
            z = drafter(seq)
            logits = lm_head(z.to(torch.bfloat16)).float()
            pred = logits.argmax(dim=-1)
            tg = cont_target_t[eval_idx].to('cuda')
            top1 = (pred == tg).float().mean().item()
        print(f'  Cont top-1 (eval): {top1:.3f}', flush=True)
    else:
        top1 = -1.0

    # ---- Measure on TextVQA (consistent with prior V5e-0 OOD measurement) ----
    print(f'[measure on TextVQA n=30]', flush=True)
    parents, depths = build_tree_d3(M, K)
    n_input = len(parents) - 1

    ds = load_dataset('lmms-lab/textvqa', split='validation', streaming=True)
    test_samples = []
    for s in ds:
        if len(test_samples) >= 30: break
        if s.get('image') is None or not s.get('question'): continue
        test_samples.append({'image': s['image'], 'question': s['question']})

    def prep_textvqa(image, question):
        if is_qwen:
            from qwen_vl_utils import process_vision_info
            messages = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":question}]}]
            texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            return processor(text=[texts], images=image_inputs, return_tensors='pt', padding=True).to('cuda', torch.bfloat16)
        else:
            prompt_txt = f"USER: <image>\n{question} ASSISTANT:"
            return processor(text=prompt_txt, images=image, return_tensors='pt').to('cuda', torch.bfloat16)

    def gen_ar(image, question, max_tok=64):
        inputs = prep_textvqa(image, question)
        with torch.no_grad():
            out = model(**inputs, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
        cur_kv = to_tuple_kv(out.past_key_values)
        last_tok = int(out.logits[0,-1,:].argmax())
        generated = [last_tok]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(max_tok):
                tok_inp = torch.tensor([[last_tok]], device='cuda')
                cur_len = cur_kv[0][0].shape[2]
                pid = torch.tensor([[cur_len]], device='cuda')
                if is_qwen: pid = pid.unsqueeze(0).expand(3,1,-1)
                cp = torch.tensor([cur_len], device='cuda')
                o = model(input_ids=tok_inp, past_key_values=kv_to_cache(cur_kv),
                          position_ids=pid, cache_position=cp,
                          use_cache=True, return_dict=True)
                last_tok = int(o.logits[0,-1,:].argmax())
                cur_kv = to_tuple_kv(o.past_key_values)
                generated.append(last_tok)
        torch.cuda.synchronize()
        return generated, max_tok/(time.perf_counter()-t0)

    def gen_eagle(image, question, max_tok=64):
        inputs = prep_textvqa(image, question)
        with torch.no_grad():
            out = model(**inputs, past_key_values=DynamicCache(),
                        use_cache=True, output_hidden_states=True, return_dict=True)
        cur_kv = to_tuple_kv(out.past_key_values)
        prompt_len = cur_kv[0][0].shape[2]
        last_h = out.hidden_states[-1][0,-1,:].float()
        last_lg = out.logits[0,-1,:].float()
        generated = []
        n_rounds = 0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            while len(generated) < max_tok:
                anchor = int(last_lg.argmax())
                root_emb = root_emb_table[anchor].unsqueeze(0).to('cuda')
                seq_cont = torch.stack([root_emb, last_h.unsqueeze(0).to('cuda')], dim=1)
                z_c = drafter(seq_cont)
                cont_logits = lm_head(z_c.to(torch.bfloat16)).float()
                cont_topm = cont_logits.topk(M, dim=-1).indices[0]
                h_b = last_h.unsqueeze(0).expand(M,-1).to('cuda')
                re_b = root_emb.expand(M,-1)
                ce_b = root_emb_table[cont_topm].to('cuda')
                seq_c2 = torch.stack([re_b, h_b, ce_b], dim=1)
                z_c2 = drafter(seq_c2)
                c2_lg = lm_head(z_c2.to(torch.bfloat16)).float()
                c2_topk = c2_lg.topk(K, dim=-1).indices
                tree_list = [anchor] + cont_topm.tolist()
                for ci in range(M): tree_list.extend(c2_topk[ci].tolist())
                tree_ids = torch.tensor([tree_list], device='cuda')
                mask, pos_ids = build_attn_mask_and_pos(parents, depths, prompt_len, 'cuda', torch.bfloat16, qwen=is_qwen)
                cache_pos = torch.arange(prompt_len, prompt_len+n_input, device='cuda')
                outt = model(input_ids=tree_ids, past_key_values=kv_to_cache(cur_kv),
                              attention_mask=mask, position_ids=pos_ids,
                              cache_position=cache_pos,
                              use_cache=True, output_hidden_states=True, return_dict=True)
                tlog = outt.logits[0]; thid = outt.hidden_states[-1][0]
                kv_after = to_tuple_kv(outt.past_key_values)
                cont_arg = int(tlog[0].argmax())
                acc_c = next((1+cj for cj in range(M) if tree_list[1+cj]==cont_arg), None)
                is_first = (n_rounds == 0); base = [anchor] if is_first else []
                if acc_c is not None:
                    c2_start = 1 + M + (acc_c-1)*K
                    c2_arg = int(tlog[acc_c].argmax())
                    acc_c2 = next((c2_start+c2j for c2j in range(K) if tree_list[c2_start+c2j]==c2_arg), None)
                    if acc_c2 is not None:
                        bonus = int(tlog[acc_c2].argmax())
                        new_toks = base + [cont_arg, c2_arg, bonus]
                        acc_idx = [0, acc_c, acc_c2]
                        new_lg = tlog[acc_c2].float(); new_h = thid[acc_c2].float()
                    else:
                        new_toks = base + [cont_arg, c2_arg]
                        acc_idx = [0, acc_c]
                        new_lg = tlog[acc_c].float(); new_h = thid[acc_c].float()
                else:
                    new_toks = base + [cont_arg]; acc_idx = [0]
                    new_lg = tlog[0].float(); new_h = thid[0].float()
                cur_kv = prune_kv(kv_after, prompt_len, acc_idx)
                prompt_len += len(acc_idx)
                last_lg = new_lg; last_h = new_h
                generated.extend(new_toks); n_rounds += 1
        torch.cuda.synchronize()
        return generated, len(generated)/(time.perf_counter()-t0)

    ar_tps_list, ssd_tps_list = [], []
    for i, ts in enumerate(test_samples):
        try:
            gc.collect(); torch.cuda.empty_cache()
            _, ar_tps = gen_ar(ts['image'], ts['question'])
            ar_tps_list.append(ar_tps)
            gc.collect(); torch.cuda.empty_cache()
            _, ssd_tps = gen_eagle(ts['image'], ts['question'])
            ssd_tps_list.append(ssd_tps)
            if (i+1) % 10 == 0:
                ar_a = np.array(ar_tps_list); ssd_a = np.array(ssd_tps_list)
                print(f'  {i+1}/{len(test_samples)}: AR {ar_a.mean():.2f}, EAGLE {ssd_a.mean():.2f}, sp {(ssd_a/ar_a).mean():.3f}', flush=True)
        except Exception as e:
            print(f'  prompt {i} fail: {type(e).__name__}: {str(e)[:80]}', flush=True)
    ar_a = np.array(ar_tps_list); ssd_a = np.array(ssd_tps_list); sp = ssd_a/ar_a
    return {
        'n': len(ar_a), 'ar_tps': float(ar_a.mean()), 'ssd_tps': float(ssd_a.mean()),
        'sp_mean': float(sp.mean()), 'sp_std': float(sp.std()),
        'cont_top1': float(top1), 'drafter_params_M': n_params/1e6,
        'drafter_dim': D
    }


print('=== EAGLE-VLM Light v2 (PROPER image+text data) ===', flush=True)
results = {}
for vlm in ['qwen2-2b']:
    try:
        results[vlm] = run_vlm(vlm)
        print(f'\n=== {vlm} EAGLE-VLM Light v2 ===')
        r = results[vlm]
        print(f'  sp: {r["sp_mean"]:.3f} ± {r["sp_std"]:.3f}')
        print(f'  Cont top-1: {r["cont_top1"]:.3f}')
        print(f'  Drafter: {r["drafter_params_M"]:.1f}M')
    except Exception as e:
        print(f'{vlm} FAIL: {type(e).__name__}: {str(e)[:200]}', flush=True)
        import traceback; traceback.print_exc()
    gc.collect(); torch.cuda.empty_cache()
    with open('./results/eagle_vlm_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved ./results/eagle_vlm_v2.json after {vlm}')
print('\nAll done')
