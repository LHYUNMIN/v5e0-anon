"""MMBench-only re-measurement with correct dataset name (lmms-lab/MMBench_EN)."""
import os, sys, json, gc, time
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, './v5e0')
sys.path.insert(0, './vendor')
from runtime_env import strip_user_site_packages; strip_user_site_packages()
import numpy as np
from datasets import load_dataset
from transformers.cache_utils import DynamicCache
from run import (VLM_CONFIGS, V5e0_Cont, V5e0_Cont2, to_tuple_kv, kv_to_cache,
                 build_tree_d3, build_attn_mask_and_pos, prune_kv)
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

M, K, MAX_TOK = 5, 3, 64
N_TEST = 200

print('=== MMBench n=200 on Qwen2-VL-2B ===', flush=True)
cfg = VLM_CONFIGS['qwen2-2b']
model = Qwen2VLForConditionalGeneration.from_pretrained(
    cfg['model_id'], torch_dtype=torch.bfloat16, attn_implementation='sdpa'
).to('cuda').eval()
proc = AutoProcessor.from_pretrained(cfg['model_id'])
D = getattr(model.config, 'hidden_size', None) or model.config.text_config.hidden_size
emb = model.get_input_embeddings()
lm_head = model.get_output_embeddings()
root_emb_table = emb.weight.detach().to(torch.float32)
for p in model.parameters(): p.requires_grad_(False)

ckpt = torch.load('./checkpoints/v5e0_qwen2-2b.pt', weights_only=False)
n1c = V5e0_Cont(dim=D, alpha_init=30.0).to('cuda')
n1c.Q_proj.weight.data = ckpt['W_Q1'].to('cuda').float()
n1c.Q_proj.bias.data = ckpt['W_Q1_bias'].to('cuda').float()
n1c2 = V5e0_Cont2(dim=D, alpha_init=30.0, beta_init=30.0).to('cuda')
n1c2.Q_proj.weight.data = ckpt['W_Q2'].to('cuda').float()
n1c2.Q_proj.bias.data = ckpt['W_Q2_bias'].to('cuda').float()
parents, depths = build_tree_d3(M, K)
n_input = len(parents) - 1


def prep_input(image, question):
    from qwen_vl_utils import process_vision_info
    messages = [{"role":"user","content":[
        {"type":"image","image":image},{"type":"text","text":question}
    ]}]
    texts = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return proc(text=[texts], images=image_inputs, return_tensors='pt', padding=True).to('cuda', torch.bfloat16)


def gen_ar(image, question, max_tok=MAX_TOK):
    inputs = prep_input(image, question)
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
            pid = torch.tensor([[cur_len]], device='cuda').unsqueeze(0).expand(3,1,-1)
            cp = torch.tensor([cur_len], device='cuda')
            o = model(input_ids=tok_inp, past_key_values=kv_to_cache(cur_kv),
                      position_ids=pid, cache_position=cp,
                      use_cache=True, return_dict=True)
            last_tok = int(o.logits[0,-1,:].argmax())
            cur_kv = to_tuple_kv(o.past_key_values)
            generated.append(last_tok)
    torch.cuda.synchronize()
    return generated, max_tok/(time.perf_counter()-t0)


def gen_v5e0(image, question, max_tok=MAX_TOK):
    inputs = prep_input(image, question)
    with torch.no_grad():
        out = model(**inputs, past_key_values=DynamicCache(),
                    use_cache=True, output_hidden_states=True, return_dict=True)
    cur_kv = to_tuple_kv(out.past_key_values)
    prompt_len = cur_kv[0][0].shape[2]
    last_h = out.hidden_states[-1][0,-1,:].float()
    last_lg = out.logits[0,-1,:].float()
    generated = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    n_rounds = 0
    with torch.no_grad():
        while len(generated) < max_tok:
            anchor = int(last_lg.argmax())
            root_emb = root_emb_table[anchor].unsqueeze(0).to('cuda')
            z = n1c(last_h.unsqueeze(0).to('cuda'), root_emb)
            cont_logits = lm_head(z.to(torch.bfloat16)).float()
            cont_topm = cont_logits.topk(M, dim=-1).indices[0]
            h_b = last_h.unsqueeze(0).expand(M,-1).to('cuda')
            re_b = root_emb.expand(M,-1)
            ce_b = root_emb_table[cont_topm].to('cuda')
            z2 = n1c2(h_b, re_b, ce_b)
            c2_lg = lm_head(z2.to(torch.bfloat16)).float()
            c2_topk = c2_lg.topk(K, dim=-1).indices
            tree_list = [anchor] + cont_topm.tolist()
            for ci in range(M): tree_list.extend(c2_topk[ci].tolist())
            tree_ids = torch.tensor([tree_list], device='cuda')
            mask, pos_ids = build_attn_mask_and_pos(parents, depths, prompt_len, 'cuda', torch.bfloat16, qwen=True)
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


def mcq_extract(text):
    import re
    t = text.strip().upper()
    m = re.search(r'\b([A-D])\b', t)
    return m.group(1) if m else None


print('[loading MMBench n=200 from lmms-lab/MMBench_EN]', flush=True)
ds = load_dataset('lmms-lab/MMBench_EN', split='dev', streaming=True)
samples = []
for s in ds:
    if len(samples) >= N_TEST: break
    img = s.get('image')
    q = s.get('question')
    ans = s.get('answer')
    if img is None or not q or not ans: continue
    choices = []
    for letter in ['A','B','C','D']:
        v = s.get(letter)
        if v: choices.append(f"{letter}. {v}")
    if not choices: continue
    hint = s.get('hint', '')
    full_q = (f"{hint}\n" if hint else '') + f"{q}\n" + "\n".join(choices) + "\nAnswer with the letter only."
    samples.append({'image': img, 'question': full_q, 'answer': ans})
print(f'  {len(samples)} samples', flush=True)

ar_tps_list, ssd_tps_list = [], []
ar_texts, ssd_texts, gts = [], [], []
for i, s in enumerate(samples):
    try:
        gc.collect(); torch.cuda.empty_cache()
        ar_gen, ar_tps = gen_ar(s['image'], s['question'])
        ar_tps_list.append(ar_tps)
        ar_texts.append(proc.tokenizer.decode(ar_gen, skip_special_tokens=True).strip())
        gc.collect(); torch.cuda.empty_cache()
        ssd_gen, ssd_tps = gen_v5e0(s['image'], s['question'])
        ssd_tps_list.append(ssd_tps)
        ssd_texts.append(proc.tokenizer.decode(ssd_gen, skip_special_tokens=True).strip())
        gts.append(s['answer'])
        if (i+1) % 50 == 0:
            ar_a = np.array(ar_tps_list); ssd_a = np.array(ssd_tps_list)
            print(f'  {i+1}/{len(samples)}: AR {ar_a.mean():.2f}, V5e-0 {ssd_a.mean():.2f}, sp {(ssd_a/ar_a).mean():.3f}', flush=True)
    except Exception as e:
        print(f'  prompt {i} fail: {type(e).__name__}: {str(e)[:80]}', flush=True)

ar_letters = [mcq_extract(t) for t in ar_texts]
ssd_letters = [mcq_extract(t) for t in ssd_texts]
ar_acc = [1 if l == str(g).upper() else 0 for l, g in zip(ar_letters, gts) if l and g]
ssd_acc = [1 if l == str(g).upper() else 0 for l, g in zip(ssd_letters, gts) if l and g]
text_match = sum(1 for a,s in zip(ar_texts, ssd_texts) if a==s) / max(len(ar_texts), 1)
letter_match = sum(1 for a,s in zip(ar_letters, ssd_letters) if a==s and a is not None) / max(len(ar_letters), 1)
ar_a = np.array(ar_tps_list); ssd_a = np.array(ssd_tps_list); sp = ssd_a/ar_a
out = {'n': len(ar_a),
       'ar_tps': float(ar_a.mean()), 'ssd_tps': float(ssd_a.mean()),
       'sp_mean': float(sp.mean()), 'sp_std': float(sp.std()),
       'ar_acc': float(np.mean(ar_acc)) if ar_acc else None,
       'ssd_acc': float(np.mean(ssd_acc)) if ssd_acc else None,
       'text_match': float(text_match),
       'letter_match': float(letter_match)}
print(f'\n=== MMBench (n={len(ar_a)}) ===')
print(f'  AR: {ar_a.mean():.2f} t/s, V5e-0: {ssd_a.mean():.2f} t/s, sp: {sp.mean():.3f} ± {sp.std():.3f}')
if ar_acc:
    print(f'  AR acc: {100*np.mean(ar_acc):.1f}%, V5e-0 acc: {100*np.mean(ssd_acc):.1f}%, letter-match: {100*letter_match:.1f}%, text-match: {100*text_match:.1f}%')

with open('./results/mmbench_fix.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved ./results/mmbench_fix.json')
