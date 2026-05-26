"""Top-p / nucleus sampling preliminary measurement on Qwen2-VL-2B.

Reuses greedy-trained V5e-0 checkpoint; modifies verifier to sample with top-p.
Acceptance: tree token matches verifier's sample = accept; else reject.
Reports throughput and acceptance rate at top-p in {0.7, 0.9, 1.0 (=greedy ref)}.
"""
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
N_TEST = 30
TEMPERATURE = 1.0  # greedy-equivalent temperature; sampling controlled by top-p


def top_p_sample(logits, top_p, temperature=1.0):
    """Sample one token via top-p (nucleus) sampling. Returns sampled token id."""
    if top_p >= 1.0 and temperature == 1.0:
        # Plain multinomial
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs > top_p
    mask[..., 0] = False  # always keep top-1
    sorted_logits[mask] = -float('inf')
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1).item()
    return int(sorted_indices[sampled_idx].item())


print('=== Top-p preliminary on Qwen2-VL-2B ===', flush=True)
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


def gen_ar_topp(image, question, top_p, max_tok=MAX_TOK):
    inputs = prep_input(image, question)
    with torch.no_grad():
        out = model(**inputs, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
    cur_kv = to_tuple_kv(out.past_key_values)
    last_tok = top_p_sample(out.logits[0,-1,:].float(), top_p)
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
            last_tok = top_p_sample(o.logits[0,-1,:].float(), top_p)
            cur_kv = to_tuple_kv(o.past_key_values)
            generated.append(last_tok)
    torch.cuda.synchronize()
    return generated, max_tok/(time.perf_counter()-t0)


def gen_v5e0_topp(image, question, top_p, max_tok=MAX_TOK):
    """V5e-0 with verifier sampling at top-p; drafter still argmax (greedy-trained).
    Accept tree branch if verifier's sample equals drafter's prediction at that node.
    """
    inputs = prep_input(image, question)
    with torch.no_grad():
        out = model(**inputs, past_key_values=DynamicCache(),
                    use_cache=True, output_hidden_states=True, return_dict=True)
    cur_kv = to_tuple_kv(out.past_key_values)
    prompt_len = cur_kv[0][0].shape[2]
    last_h = out.hidden_states[-1][0,-1,:].float()
    last_lg = out.logits[0,-1,:].float()
    generated = []
    n_rounds = 0
    n_accept_c = 0
    n_accept_c2 = 0
    n_root_only = 0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        while len(generated) < max_tok:
            # Drafter is greedy: anchor = argmax (matches drafter training)
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
            # Verifier samples at root position
            cont_arg = top_p_sample(tlog[0].float(), top_p)
            acc_c = next((1+cj for cj in range(M) if tree_list[1+cj]==cont_arg), None)
            is_first = (n_rounds == 0); base = [anchor] if is_first else []
            if acc_c is not None:
                c2_start = 1 + M + (acc_c-1)*K
                c2_arg = top_p_sample(tlog[acc_c].float(), top_p)
                acc_c2 = next((c2_start+c2j for c2j in range(K) if tree_list[c2_start+c2j]==c2_arg), None)
                if acc_c2 is not None:
                    bonus = top_p_sample(tlog[acc_c2].float(), top_p)
                    new_toks = base + [cont_arg, c2_arg, bonus]
                    acc_idx = [0, acc_c, acc_c2]
                    new_lg = tlog[acc_c2].float(); new_h = thid[acc_c2].float()
                    n_accept_c += 1; n_accept_c2 += 1
                else:
                    new_toks = base + [cont_arg, c2_arg]
                    acc_idx = [0, acc_c]
                    new_lg = tlog[acc_c].float(); new_h = thid[acc_c].float()
                    n_accept_c += 1
            else:
                new_toks = base + [cont_arg]; acc_idx = [0]
                new_lg = tlog[0].float(); new_h = thid[0].float()
                n_root_only += 1
            cur_kv = prune_kv(kv_after, prompt_len, acc_idx)
            prompt_len += len(acc_idx)
            last_lg = new_lg; last_h = new_h
            generated.extend(new_toks); n_rounds += 1
    torch.cuda.synchronize()
    return generated, len(generated)/(time.perf_counter()-t0), {
        'rounds': n_rounds, 'root_only': n_root_only,
        'accept_c': n_accept_c, 'accept_c2': n_accept_c2
    }


# Load LLaVA-Instruct evaluation prompts (consistent with main eval)
print('[loading LLaVA-Instruct n=30]', flush=True)
ds = load_dataset('liuhaotian/LLaVA-Instruct-150K', split='train', streaming=True)
import random
random.seed(999)
samples = []
ds_iter = iter(ds)
seen = 0
while len(samples) < N_TEST and seen < 5000:
    s = next(ds_iter)
    seen += 1
    if s.get('image') is None: continue
    # LLaVA-Instruct image is a filename; need to load from COCO
    img_filename = s['image']
    if not img_filename.endswith('.jpg'): continue
    # We don't have COCO images locally; skip and use a different dataset
    break

# Fall back to TextVQA images
print('[loading TextVQA n=30 instead]', flush=True)
ds = load_dataset('lmms-lab/textvqa', split='validation', streaming=True)
samples = []
for s in ds:
    if len(samples) >= N_TEST: break
    img = s.get('image')
    q = s.get('question')
    if img is None or not q: continue
    samples.append({'image': img, 'question': q})
print(f'  {len(samples)} prompts', flush=True)

results = {}
for top_p in [1.0, 0.9, 0.7]:
    print(f'\n=== top_p = {top_p} ===', flush=True)
    torch.manual_seed(42)
    ar_tps_list, ssd_tps_list = [], []
    rounds_total, acc_c_total, acc_c2_total, root_only_total = 0, 0, 0, 0
    for i, s in enumerate(samples):
        try:
            gc.collect(); torch.cuda.empty_cache()
            torch.manual_seed(42 + i)
            _, ar_tps = gen_ar_topp(s['image'], s['question'], top_p)
            ar_tps_list.append(ar_tps)
            gc.collect(); torch.cuda.empty_cache()
            torch.manual_seed(42 + i)
            _, ssd_tps, stats = gen_v5e0_topp(s['image'], s['question'], top_p)
            ssd_tps_list.append(ssd_tps)
            rounds_total += stats['rounds']
            acc_c_total += stats['accept_c']
            acc_c2_total += stats['accept_c2']
            root_only_total += stats['root_only']
            if (i+1) % 10 == 0:
                ar_a = np.array(ar_tps_list); ssd_a = np.array(ssd_tps_list)
                print(f'  {i+1}/{len(samples)}: AR {ar_a.mean():.2f}, V5e-0 {ssd_a.mean():.2f}, sp {(ssd_a/ar_a).mean():.3f}', flush=True)
        except Exception as e:
            print(f'  prompt {i} fail: {type(e).__name__}: {str(e)[:80]}', flush=True)
    ar_a = np.array(ar_tps_list); ssd_a = np.array(ssd_tps_list); sp = ssd_a/ar_a
    results[f'top_p_{top_p}'] = {
        'n': len(ar_a),
        'ar_tps': float(ar_a.mean()), 'ssd_tps': float(ssd_a.mean()),
        'sp_mean': float(sp.mean()), 'sp_std': float(sp.std()),
        'rounds_total': rounds_total,
        'accept_c_rate': acc_c_total / max(rounds_total, 1),
        'accept_c2_rate': acc_c2_total / max(rounds_total, 1),
        'root_only_rate': root_only_total / max(rounds_total, 1)
    }
    print(f'  AR: {ar_a.mean():.2f} t/s, V5e-0: {ssd_a.mean():.2f} t/s, sp: {sp.mean():.3f} ± {sp.std():.3f}')
    print(f'  acc_c rate: {acc_c_total/max(rounds_total,1):.3f}, acc_c2 rate: {acc_c2_total/max(rounds_total,1):.3f}')

with open('./results/topp_preliminary.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved ./results/topp_preliminary.json')
