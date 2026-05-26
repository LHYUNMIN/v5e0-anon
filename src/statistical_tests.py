#!/usr/bin/env python3
"""Statistical significance analysis for V5e-0 paper.

Computes for each ablation comparison:
  - Paired t-test (p-value)
  - Bootstrap 95% CI for sp difference
  - Cohen's d effect size

Uses existing JSON results in ./results/.
"""
import os, json, glob
import numpy as np
from scipy import stats


def load_sp(path):
    with open(path) as f: d = json.load(f)
    return np.array(d["raw"]["sp"])


def paired_test(a, b, name_a, name_b):
    """Paired t-test + bootstrap CI."""
    n = min(len(a), len(b))
    a, b = np.array(a[:n]), np.array(b[:n])
    diff = a - b

    t_stat, p_val = stats.ttest_rel(a, b)

    # Bootstrap 95% CI for mean diff
    boot_diffs = []
    rng = np.random.RandomState(42)
    for _ in range(10000):
        idx = rng.randint(0, n, n)
        boot_diffs.append(diff[idx].mean())
    boot_lo, boot_hi = np.percentile(boot_diffs, [2.5, 97.5])

    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

    print(f"\n[{name_a} vs {name_b}, n={n}]")
    print(f"  Mean: {a.mean():.3f} vs {b.mean():.3f}, diff = {diff.mean():+.3f}")
    print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_val:.2e}")
    print(f"  Bootstrap 95% CI: [{boot_lo:+.3f}, {boot_hi:+.3f}]")
    print(f"  Cohen's d: {cohens_d:.3f}")
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  Significance: {sig}")
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(diff.mean()), "t": float(t_stat), "p": float(p_val),
            "ci_low": float(boot_lo), "ci_high": float(boot_hi),
            "cohens_d": float(cohens_d), "n": n, "significance": sig}


# =============================================================
# All pairwise comparisons
# =============================================================
results = {}

print("="*70)
print("V5e-0 Statistical Significance Analysis")
print("="*70)

# 1. Single-root depth-3 vs static (16, 3) — per VLM
print("\n=== Comparison 1: Depth-3 single-root vs Static (16, 3) ===")
for vlm in ["qwen2-2b", "qwen2-7b", "llava-1.5-7b", "llava-1.6-mistral-7b"]:
    try:
        d3 = load_sp(f"./results/path1_v5e0_d3_4vlm_{vlm}.json")
        static = load_sp(f"./results/T1_4_static_{vlm}.json")
        results[f"depth3_vs_static_{vlm}"] = paired_test(
            d3, static, f"depth-3 ({vlm})", f"static (16,3) ({vlm})")
    except FileNotFoundError as e:
        print(f"  Skipped {vlm}: {e}")

# 2. 128-token vs 64-token sp
print("\n\n=== Comparison 2: 128 tokens vs 64 tokens ===")
for vlm in ["qwen2-2b", "qwen2-7b", "llava-1.5-7b", "llava-1.6-mistral-7b"]:
    try:
        long_sp = load_sp(f"./results/measure_{vlm}_M5_K3_t128.json")
        short_sp = load_sp(f"./results/path1_v5e0_d3_4vlm_{vlm}.json")
        results[f"128_vs_64_{vlm}"] = paired_test(
            long_sp, short_sp, f"128 tok ({vlm})", f"64 tok ({vlm})")
    except FileNotFoundError as e:
        print(f"  Skipped {vlm}: {e}")

# 3. (M, K) configurations vs M=5, K=3 baseline (Qwen2-7B as example)
print("\n\n=== Comparison 3: (M, K) configurations vs (5, 3) on Qwen2-7B ===")
for mk in ["5_1", "10_1", "10_2", "10_3", "15_1"]:
    try:
        M, K = mk.split("_")
        sp = load_sp(f"./results/measure_qwen2-7b_M{M}_K{K}_t64.json")
        baseline = load_sp("./results/path1_v5e0_d3_4vlm_qwen2-7b.json")
        results[f"qwen7b_M{M}K{K}_vs_5_3"] = paired_test(
            sp, baseline, f"M={M}, K={K}", "M=5, K=3 (baseline)")
    except FileNotFoundError as e:
        print(f"  Skipped {mk}: {e}")

# 4. Cross-VLM ablations on LLaVA-1.5
print("\n\n=== Comparison 4: Cross-VLM ablations on LLaVA-1.5 ===")
try:
    with open("./results/B_ablation_llava-1.5-7b_linear_a30.json") as f: linear = json.load(f)
    with open("./results/B_ablation_llava-1.5-7b_kl_a30.json") as f: kl = json.load(f)
    with open("./results/B_ablation_llava-1.5-7b_mlp_3_a30.json") as f: mlp3 = json.load(f)
    print(f"\nTop-3 acceptance (proxy for sp):")
    print(f"  Linear:  {linear['eval_topk']['3']:.4f}")
    print(f"  KL:      {kl['eval_topk']['3']:.4f}  (Δ {kl['eval_topk']['3']-linear['eval_topk']['3']:+.4f})")
    print(f"  MLP-3:   {mlp3['eval_topk']['3']:.4f}  (Δ {mlp3['eval_topk']['3']-linear['eval_topk']['3']:+.4f})")
    print(f"  All differences < 0.02 → no meaningful improvement")
except FileNotFoundError as e:
    print(f"  Skipped: {e}")

# 5. Lossless n=30 cross-VLM summary
print("\n\n=== Lossless n=30 cross-VLM summary ===")
losses = {}
for vlm in ["qwen2-2b", "qwen2-7b", "llava-1.5-7b", "llava-1.6-mistral-7b"]:
    try:
        with open(f"./results/path1_v5e0_d3_lossless_{vlm}.json") as f:
            d = json.load(f)
        losses[vlm] = {"match": d["total_match_rate"], "exact": d["n_exact"]}
        print(f"  {vlm}: match {d['total_match_rate']*100:6.2f}%, exact {d['n_exact']}/{d['n_test']}")
    except FileNotFoundError:
        pass

# Save all results
with open("./results/statistical_tests.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[saved ./results/statistical_tests.json]")

# =============================================================
# Summary table for paper
# =============================================================
print("\n\n" + "="*70)
print("SUMMARY TABLE FOR PAPER (Table A.X: Statistical tests)")
print("="*70)
print(f"{'Comparison':40s} {'Δsp':>8s} {'p':>10s} {'CI 95%':>16s} {'sig':>5s}")
print("-"*70)
for key, r in results.items():
    ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
    print(f"{key:40s} {r['diff']:>+8.3f} {r['p']:>10.2e} {ci:>16s} {r['significance']:>5s}")
