#!/usr/bin/env python3
"""Generate bf16 floor / first-divergence distribution plots from lossless data.

Two plots:
  1. Token match rate vs tree batch size (monotonic — bf16 floor)
  2. First-divergence position distribution per VLM

Both support Section 6 (Methodology / bf16 floor) of the paper.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 120

# ============================================================
# (1) Token match vs tree batch size
# ============================================================
# Data from Section 4.4 (paper) and T1.2 results
tree_data = {
    # (tree_size, token_match%, vlm, exact)
    ("single-root M=3 d2",  4,  70.47, "Qwen2-VL-2B", "5/10"),
    ("single-root M=15 d2", 16, 82.81, "Qwen2-VL-2B", "7/10"),
    ("single-root M=5 K=3 d3", 21, 88.44, "Qwen2-VL-2B", "8/10"),
    ("static (16, 3) d2",    64, 88.91, "Qwen2-VL-2B", "8/10"),
}

vlm_lossless = {
    "Qwen2-VL-2B":             88.44,
    "Qwen2-VL-7B":             96.09,
    "LLaVA-1.5-7B":            82.34,
    "LLaVA-1.6-Mistral-7B":    100.00,
}

# ---------- Plot 1: tree batch size monotonicity ----------
fig, ax = plt.subplots(figsize=(5.5, 3.5))
sizes = sorted({d[1] for d in tree_data})
match_by_size = {d[1]: d[2] for d in tree_data}
ax.plot(sizes, [match_by_size[s] for s in sizes], 'o-', color='C0', linewidth=2, markersize=8)
for d in tree_data:
    ax.annotate(d[0].split(' (')[0], (d[1], d[2]), textcoords="offset points",
                xytext=(8, -3), fontsize=8, color='gray')
ax.set_xlabel("Tree batch size (input tokens / round)")
ax.set_ylabel("Token-level match with AR (%)")
ax.set_title("bf16 lossless floor: monotonic in tree batch size (Qwen2-VL-2B)")
ax.set_ylim(60, 100)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./results/T2_5_bf16_floor.pdf", bbox_inches='tight')
plt.savefig("./results/T2_5_bf16_floor.png", bbox_inches='tight')
print("[saved ./results/T2_5_bf16_floor.pdf, .png]")
plt.close()

# ---------- Plot 2: cross-VLM lossless ----------
fig, ax = plt.subplots(figsize=(5.5, 3.5))
vlms = list(vlm_lossless.keys())
rates = [vlm_lossless[v] for v in vlms]
bars = ax.bar(range(len(vlms)), rates, color=['C0', 'C0', 'C1', 'C1'])
for i, r in enumerate(rates):
    ax.text(i, r + 0.5, f"{r:.2f}%", ha='center', fontsize=10)
ax.set_xticks(range(len(vlms)))
ax.set_xticklabels(["Qwen-2B", "Qwen-7B", "LLaVA-1.5", "LLaVA-1.6"], rotation=20)
ax.set_ylabel("Token-level match with AR (%)")
ax.set_title("Cross-VLM lossless verification (V5e-0 depth-3, n=10)")
ax.axhline(100, linestyle='--', color='gray', alpha=0.5, label="Exact AR")
ax.set_ylim(70, 105)
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("./results/T2_5_lossless_4vlm.pdf", bbox_inches='tight')
plt.savefig("./results/T2_5_lossless_4vlm.png", bbox_inches='tight')
print("[saved ./results/T2_5_lossless_4vlm.pdf, .png]")
plt.close()

# ---------- Plot 3: first-divergence distribution from T1.2 data ----------
loss_files = [
    "./results/path1_v5e0_d3_lossless.json",                          # qwen2-2b
    "./results/path1_v5e0_d3_lossless_qwen2-7b.json",
    "./results/path1_v5e0_d3_lossless_llava-1.5-7b.json",
    "./results/path1_v5e0_d3_lossless_llava-1.6-mistral-7b.json",
]
labels = ["Qwen-2B", "Qwen-7B", "LLaVA-1.5", "LLaVA-1.6"]

fig, ax = plt.subplots(figsize=(5.5, 3.5))
all_first_diffs = []
for path, lbl in zip(loss_files, labels):
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    fd = [p["first_diff"] for p in d["per_prompt"]]
    all_first_diffs.append((lbl, fd))

# Box plot
positions = list(range(1, len(all_first_diffs) + 1))
ax.boxplot([fd for _, fd in all_first_diffs], positions=positions, widths=0.5,
           patch_artist=True, boxprops=dict(facecolor='C0', alpha=0.5))
ax.set_xticks(positions)
ax.set_xticklabels([l for l, _ in all_first_diffs])
ax.set_ylabel("First-divergence position (vs AR)")
ax.set_title("First-divergence index distribution (10 prompts × 64 tokens)")
ax.set_ylim(0, 70)
ax.axhline(64, linestyle='--', color='gray', alpha=0.5, label='exact match (no divergence)')
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("./results/T2_5_first_diff.pdf", bbox_inches='tight')
plt.savefig("./results/T2_5_first_diff.png", bbox_inches='tight')
print("[saved ./results/T2_5_first_diff.pdf, .png]")
plt.close()

print("\nDone. 3 plots ready for paper Section 6:")
print("  T2_5_bf16_floor.pdf      — monotonic in tree size")
print("  T2_5_lossless_4vlm.pdf   — cross-VLM lossless rates")
print("  T2_5_first_diff.pdf      — first-divergence boxplot")
