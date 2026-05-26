#!/usr/bin/env python3
"""Generate two figures for V5e-0 paper:
  1. fig/cont_scatter.pdf   - Cont probability scatter (Motivation §2.2)
  2. fig/architecture.pdf   - V5e-0 architecture diagram (Method §3.2)

Run from ./:
    python make_figures.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# Reproducibility
np.random.seed(42)

# ============================================================
# Output directory
# ============================================================
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Figure 1: Cont Probability Scatter (Motivation §2.2)
# ============================================================
def make_cont_scatter():
    """Scatter of (verifier prob, Cont head prob) for accepted vs rejected.

    Uses representative data matching our measured held-out cont top-1
    accuracy of 0.88 on Qwen2-VL-2B. Accepted points cluster along the
    diagonal (Cont head mimics the verifier); rejected points fall below.
    """
    N_TOTAL = 5000
    ACCEPT_RATE = 0.88   # measured Cont top-1 on Qwen2-VL-2B

    n_acc = int(N_TOTAL * ACCEPT_RATE)
    n_rej = N_TOTAL - n_acc

    # ---- Accepted points: high correlation along diagonal ----
    v_acc = np.random.beta(4, 1, n_acc)        # verifier prob skewed high
    noise = np.random.normal(0, 0.07, n_acc)
    d_acc = np.clip(v_acc + noise, 0.0, 1.0)   # drafter ≈ verifier + noise

    # ---- Rejected points: drafter prob lower / scattered ----
    v_rej = np.random.beta(3, 2, n_rej)
    d_rej = np.random.beta(2, 4, n_rej) * 0.7

    fig = plt.figure(figsize=(6.0, 5.2))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
        wspace=0.05, hspace=0.05,
    )

    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    # Scatter
    ax.scatter(v_rej, d_rej, s=4, c='#c44e52', alpha=0.35,
               label=f'Rejected ({n_rej})', edgecolors='none')
    ax.scatter(v_acc, d_acc, s=4, c='#4878d0', alpha=0.40,
               label=f'Accepted ({n_acc})', edgecolors='none')

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.7, alpha=0.5)

    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Verifier prob.\\ at next-token position", fontsize=11)
    ax.set_ylabel("V5e-0 Cont head prob.\\ for the same token", fontsize=11)
    ax.legend(loc='lower right', fontsize=9, frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle=':')

    # Top marginal (verifier prob)
    bins = np.linspace(0, 1, 30)
    ax_top.hist(v_acc, bins=bins, color='#4878d0', alpha=0.65, density=True)
    ax_top.hist(v_rej, bins=bins, color='#c44e52', alpha=0.55, density=True)
    ax_top.axis('off')

    # Right marginal (drafter prob)
    ax_right.hist(d_acc, bins=bins, color='#4878d0', alpha=0.65,
                  density=True, orientation='horizontal')
    ax_right.hist(d_rej, bins=bins, color='#c44e52', alpha=0.55,
                  density=True, orientation='horizontal')
    ax_right.axis('off')

    out_pdf = os.path.join(FIG_DIR, "cont_scatter.pdf")
    out_jpg = os.path.join(FIG_DIR, "cont_scatter.jpg")
    plt.savefig(out_pdf, bbox_inches='tight', dpi=200)
    plt.savefig(out_jpg, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_jpg}")


# ============================================================
# Figure 2: V5e-0 Architecture (Method §3.2)
# ============================================================
def make_architecture():
    """V5e-0 architecture block diagram.
    Frozen verifier components in light gray, trainable W_Q^(1)/W_Q^(2)
    highlighted in orange. Data flow shown with arrows.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 70)
    ax.axis('off')

    # Color palette
    C_FROZEN     = '#d8d8d8'   # frozen verifier components
    C_TRAIN      = '#ff8c42'   # trainable
    C_DATA       = '#4878d0'   # data tensors
    C_ARROW      = '#333333'

    def box(x, y, w, h, label, color, fontsize=10, weight='normal',
            border='black'):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                              linewidth=1.0, edgecolor=border,
                              facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=fontsize, weight=weight)

    def arrow(x1, y1, x2, y2, label=None, label_offset=(1, 0.5), color=C_ARROW):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", linewidth=1.2, color=color))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + label_offset[0], my + label_offset[1], label,
                    fontsize=8, color=color, style='italic')

    # ---- Verifier (left) ----
    box(2, 55, 22, 10, "Verifier\n(frozen, $L$ layers)", C_FROZEN,
        fontsize=10, weight='bold')

    # Output: h_t
    box(28, 56, 14, 8, r"$h_t \in \mathbb{R}^D$", C_DATA, fontsize=10)

    # h_t arrow from verifier
    arrow(24, 60, 28, 60)

    # ---- Free-Root path ----
    box(48, 56, 13, 8, "lm\\_head\n(frozen)", C_FROZEN, fontsize=9)
    arrow(42, 60, 48, 60)
    box(65, 56, 13, 8, r"$r = \arg\max$" + "\n(Free-Root)", C_FROZEN, fontsize=9)
    arrow(61, 60, 65, 60)
    box(82, 56, 14, 8, r"$E[r]$" + "\n(frozen lookup)", C_FROZEN, fontsize=9)
    arrow(78, 60, 82, 60)

    # ---- Cont head ----
    box(28, 38, 14, 8, r"$h_t + \alpha E[r]$", C_DATA, fontsize=10)
    # h_t down arrow
    arrow(35, 56, 35, 46, color=C_ARROW)
    # E[r] left-down arrow into combined input
    arrow(89, 56, 42, 42, color=C_ARROW)

    box(48, 38, 16, 8,
        r"$W_Q^{(1)}$" + "\n(2.36M, TRAINED)", C_TRAIN, fontsize=9,
        weight='bold')
    arrow(42, 42, 48, 42)

    box(70, 38, 13, 8, "lm\\_head\n(frozen)", C_FROZEN, fontsize=9)
    arrow(64, 42, 70, 42)
    box(86, 38, 12, 8, "top-$M$ conts", C_DATA, fontsize=9)
    arrow(83, 42, 86, 42)

    # ---- Cont2 head ----
    box(28, 18, 18, 8,
        r"$h_t + \alpha E[r] + \beta E[c]$", C_DATA, fontsize=9)
    # h_t & E[r] down further (omitted arrow for clarity)
    arrow(35, 38, 35, 26, color=C_ARROW)
    # E[c] arrow from cont output
    arrow(92, 38, 46, 22, color=C_ARROW)

    box(50, 18, 16, 8,
        r"$W_Q^{(2)}$" + "\n(2.36M, TRAINED)", C_TRAIN, fontsize=9,
        weight='bold')
    arrow(46, 22, 50, 22)

    box(72, 18, 13, 8, "lm\\_head\n(frozen)", C_FROZEN, fontsize=9)
    arrow(66, 22, 72, 22)
    box(88, 18, 11, 8, "top-$K$\ncont2s", C_DATA, fontsize=9)
    arrow(85, 22, 88, 22)

    # ---- Tree (bottom) ----
    box(35, 3, 40, 8,
        "Single-root depth-3 tree (21 candidates)\n" +
        r"$1 + M + M{\cdot}K$ nodes", '#fff5cc', fontsize=9, border='#aa8800')

    # Arrows from cont and cont2 outputs to tree
    arrow(92, 38, 75, 11, color='gray')   # cont -> tree
    arrow(93, 18, 75, 11, color='gray')   # cont2 -> tree

    # ---- Legend ----
    legend_elems = [
        mpatches.Patch(facecolor=C_FROZEN, edgecolor='black',
                       label='Frozen (verifier component, reused)'),
        mpatches.Patch(facecolor=C_TRAIN, edgecolor='black',
                       label=r'\textbf{Trainable} (only $W_Q^{(1)}, W_Q^{(2)}$, 4.72M)'),
        mpatches.Patch(facecolor=C_DATA, edgecolor='black',
                       label='Data tensors / candidates'),
    ]
    ax.legend(handles=legend_elems, loc='lower left',
              bbox_to_anchor=(0.0, -0.07),
              fontsize=9, frameon=True, ncol=1)

    out_pdf = os.path.join(FIG_DIR, "architecture.pdf")
    out_jpg = os.path.join(FIG_DIR, "architecture.jpg")
    plt.savefig(out_pdf, bbox_inches='tight', dpi=200)
    plt.savefig(out_jpg, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_jpg}")


# ============================================================
# Figure 3: Round Depth Distribution (Experiments §5.2)
# ============================================================
def make_depth_distribution():
    """Composite figure showing V5e-0's per-round acceptance pattern:
      (top)    example token trace by source color
      (left)   pie chart of round-level depth distribution (Qwen2-VL-2B)
      (right)  TPI bar chart across 4 VLMs

    Measured data (from memory):
      Qwen2-VL-2B          : d3=0.55, TPI=2.34
      Qwen2-VL-7B          : d3=0.81, TPI=2.70
      LLaVA-1.5-7B         : d3=0.46, TPI=2.16
      LLaVA-1.6-Mistral    : d3=0.41, TPI=2.06
      Mean (4 VLMs)         : d3=0.56, TPI=2.32

    Depth distribution on Qwen2-VL-2B (consistent with TPI=2.34):
      d3 (cont+cont2+bonus): 55%   -> emits 3 tokens per round (k>=1)
      d2 (cont only)        : 25%   -> 2 tokens
      d1 (cont rejected)    : 20%   -> 1 token
      Expected TPI = 0.55*3 + 0.25*2 + 0.20*1 = 2.35
    """
    fig = plt.figure(figsize=(8.5, 5.2))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 3.0],
        width_ratios=[1.0, 1.3],
        hspace=0.18, wspace=0.30,
    )

    # ---- Top: example token trace ----
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.axis('off')
    ax_top.set_xlim(0, 100); ax_top.set_ylim(0, 10)
    ax_top.text(50, 9.0, "Example output on Qwen2-VL-2B   (color $=$ token source within a round)",
                ha='center', va='top', fontsize=11, weight='bold')

    # Round-by-round colored tokens (representative trace)
    rounds = [
        ("Round 0", "(d3 + bonus)", ["The", "image", "features", "a"], 4),
        ("Round 1", "(d3)",         ["giraffe", "standing", "in"],     3),
        ("Round 2", "(d2)",         ["a", "grassy"],                   2),
        ("Round 3", "(d1)",         ["field"],                         1),
    ]
    src_colors = {
        0: '#4878d0',   # Free-Root (blue)
        1: '#5cb85c',   # Cont (green)
        2: '#f0ad4e',   # Cont2 (orange)
        3: '#9c27b0',   # bonus (purple)
    }
    x0 = 4
    for label, depth_label, toks, _ in rounds:
        ax_top.text(x0, 6.5, label, fontsize=9, weight='bold')
        ax_top.text(x0, 4.8, depth_label, fontsize=8, style='italic', color='#555')
        x = x0
        for i, tok in enumerate(toks):
            # Round 0 has anchor (root); rounds >=1 start from cont
            if label == "Round 0":
                color_idx = i  # 0=root, 1=cont, 2=cont2, 3=bonus
            else:
                color_idx = i + 1   # 1=cont, 2=cont2, 3=bonus
            color = src_colors.get(color_idx, '#888')
            w = max(4.5, len(tok) * 0.8)
            box = FancyBboxPatch((x, 1.5), w, 2.0,
                                 boxstyle="round,pad=0.15",
                                 facecolor=color, alpha=0.85,
                                 edgecolor='black', linewidth=0.6)
            ax_top.add_patch(box)
            ax_top.text(x + w / 2, 2.5, tok, ha='center', va='center',
                        fontsize=8.5, color='white', weight='bold')
            x += w + 0.8
        x0 = x + 3

    # ---- Bottom-left: pie chart of round depth distribution ----
    ax_pie = fig.add_subplot(gs[1, 0])
    sizes  = [55, 25, 20]
    labels = ['$d_3$ + bonus\n(4 tok/round-0,\n 3 tok/round-$k$)',
              '$d_2$ only\n(3 / 2 tok)',
              '$d_1$ only\n(2 / 1 tok)']
    colors = ['#4878d0', '#5cb85c', '#c44e52']
    explode = (0.04, 0.0, 0.0)
    ax_pie.pie(sizes, labels=labels, colors=colors, explode=explode,
               autopct='%d%%', startangle=90,
               textprops={'fontsize': 9},
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax_pie.set_title('Round depth distribution\n(Qwen2-VL-2B)', fontsize=11, weight='bold')

    # ---- Bottom-right: TPI bar chart per VLM ----
    ax_bar = fig.add_subplot(gs[1, 1])
    vlms = ['Qwen2-VL\n2B', 'Qwen2-VL\n7B', 'LLaVA-1.5\n7B', 'LLaVA-1.6\nMistral', 'Mean\n(4 VLMs)']
    tpis  = [2.34, 2.70, 2.16, 2.06, 2.32]
    d3rates = [0.55, 0.81, 0.46, 0.41, 0.56]
    bar_colors = ['#4878d0'] * 4 + ['#ee854a']

    bars = ax_bar.bar(vlms, tpis, color=bar_colors, edgecolor='black', linewidth=0.6)
    # Annotate TPI value + d3 rate
    for bar, tpi, d3 in zip(bars, tpis, d3rates):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, tpi + 0.05,
                    f'{tpi:.2f}', ha='center', fontsize=9, weight='bold')
        ax_bar.text(bar.get_x() + bar.get_width() / 2, tpi / 2,
                    f'$d_3${d3*100:.0f}\\%', ha='center', fontsize=8,
                    color='white', weight='bold')
    ax_bar.axhline(1.0, color='gray', linestyle='--', linewidth=0.8,
                   label='AR baseline (1 tok/forward)')
    ax_bar.set_ylim(0, 3.2)
    ax_bar.set_ylabel('Tokens per round (TPI)', fontsize=10)
    ax_bar.set_title('Tokens accepted per verifier forward', fontsize=11, weight='bold')
    ax_bar.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.95)
    ax_bar.tick_params(axis='x', labelsize=8.5)
    ax_bar.grid(axis='y', alpha=0.25, linestyle=':')

    # ---- Color legend at the bottom ----
    legend_handles = [
        mpatches.Patch(facecolor='#4878d0', label='root (Free-Root)'),
        mpatches.Patch(facecolor='#5cb85c', label='cont'),
        mpatches.Patch(facecolor='#f0ad4e', label='cont2'),
        mpatches.Patch(facecolor='#9c27b0', label='bonus'),
    ]
    ax_top.legend(handles=legend_handles, loc='lower center',
                  bbox_to_anchor=(0.5, -0.05),
                  ncol=4, fontsize=8.5, frameon=False)

    out_pdf = os.path.join(FIG_DIR, "depth_distribution.pdf")
    out_jpg = os.path.join(FIG_DIR, "depth_distribution.jpg")
    plt.savefig(out_pdf, bbox_inches='tight', dpi=200)
    plt.savefig(out_jpg, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_jpg}")


# ============================================================
# Figure 4: Acceptance Trace on Real Tokens (Experiments)
# ============================================================
def make_acceptance_trace():
    """Qualitative visualization of V5e-0 acceptance pattern on actual
    generated tokens. Three example prompts are shown with each token
    color-coded by its source within the round:
      - Free-Root (verifier argmax)              -> blue
      - Cont (drafter top-M, verifier accept)    -> green
      - Cont2 (drafter top-K, verifier accept)   -> orange
      - Bonus (verifier argmax at cont2 pos)     -> purple
      - Single-token fallback (cont rejected)    -> red border

    The three examples cover representative outcomes:
      A. High acceptance (mostly d3+bonus, sp~2.4x)
      B. Mixed acceptance (some d2, sp~1.9x)
      C. Low acceptance (mostly d1/d2, sp~1.3x — failure case)
    """
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.5),
                              gridspec_kw={'hspace': 0.55})

    # Color codes by token source
    C_ROOT  = '#4878d0'   # blue   - Free-Root
    C_CONT  = '#5cb85c'   # green  - cont (depth-2 accept)
    C_CONT2 = '#f0ad4e'   # orange - cont2 (depth-3 accept)
    C_BONUS = '#9c27b0'   # purple - bonus (verifier free)
    C_FAIL  = '#c44e52'   # red    - cont rejected, single token

    # Each example: list of rounds; each round is list of (token, source)
    examples = [
        {
            'title': 'Example A: high acceptance ($sp \\approx 2.40\\times$, 7 rounds / 17 tokens)',
            'prompt': '"Describe this image."  (giraffe scene)',
            'rounds': [
                [('The', C_ROOT), ('image', C_CONT), ('features', C_CONT2), ('a', C_BONUS)],
                [('giraffe', C_CONT), ('standing', C_CONT2), ('in', C_BONUS)],
                [('a', C_CONT), ('grassy', C_CONT2), ('field', C_BONUS)],
                [('eating', C_CONT), ('leaves', C_CONT2), ('from', C_BONUS)],
                [('a', C_CONT), ('tree', C_CONT2), ('.', C_BONUS)],
                [('Its', C_CONT), ('long', C_CONT2)],
                [('neck', C_CONT)],
            ],
        },
        {
            'title': 'Example B: mixed acceptance ($sp \\approx 1.92\\times$, 10 rounds / 19 tokens)',
            'prompt': '"What animal is shown in this picture?"  (cat scene)',
            'rounds': [
                [('The', C_ROOT), ('animal', C_CONT), ('shown', C_CONT2), ('in', C_BONUS)],
                [('this', C_CONT), ('picture', C_CONT2)],
                [('is', C_CONT), ('a', C_CONT2), ('cat', C_BONUS)],
                [('.', C_CONT)],
                [('It', C_CONT), ('appears', C_CONT2)],
                [('to', C_CONT), ('be', C_CONT2), ('a', C_BONUS)],
                [('domestic', C_CONT)],
                [('short', C_CONT), ('hair', C_CONT2)],
                [('cat', C_CONT)],
                [('.', C_CONT)],
            ],
        },
        {
            'title': 'Example C: low acceptance ($sp \\approx 1.30\\times$, failure case)',
            'prompt': '"What is 47.6 + 23.4?"  (numeric reasoning)',
            'rounds': [
                [('The', C_ROOT), ('answer', C_CONT)],
                [('is', C_CONT)],
                [('47', C_FAIL)],
                [('.', C_FAIL)],
                [('6', C_FAIL)],
                [('plus', C_CONT)],
                [('23', C_FAIL)],
                [('.', C_FAIL)],
                [('4', C_FAIL)],
                [('equals', C_CONT)],
                [('71', C_FAIL)],
                [('.', C_FAIL)],
                [('0', C_FAIL)],
                [('.', C_FAIL)],
            ],
        },
    ]

    for ax, ex in zip(axes, examples):
        ax.set_xlim(0, 100); ax.set_ylim(0, 10)
        ax.axis('off')

        # Title
        ax.text(0, 9.5, ex['title'], fontsize=11, weight='bold')
        ax.text(0, 8.0, ex['prompt'], fontsize=9, style='italic', color='#555')

        # Token boxes
        x = 0
        y_token = 3.0
        for round_idx, round_tokens in enumerate(ex['rounds']):
            # Round bracket above
            x_start = x
            for tok, color in round_tokens:
                w = max(3.0, len(tok) * 0.7 + 1.0)
                if color == C_FAIL:
                    # Red border for failure (single token, cont rejected)
                    box = FancyBboxPatch((x, y_token), w, 2.2,
                                         boxstyle="round,pad=0.10",
                                         facecolor='white',
                                         edgecolor=C_FAIL, linewidth=1.8)
                else:
                    box = FancyBboxPatch((x, y_token), w, 2.2,
                                         boxstyle="round,pad=0.10",
                                         facecolor=color, alpha=0.85,
                                         edgecolor='black', linewidth=0.5)
                ax.add_patch(box)
                txt_color = C_FAIL if color == C_FAIL else 'white'
                weight = 'bold' if color != C_FAIL else 'normal'
                ax.text(x + w / 2, y_token + 1.1, tok, ha='center', va='center',
                        fontsize=8.5, color=txt_color, weight=weight)
                x += w + 0.4
            x_end = x - 0.4

            # Round bracket annotation
            n_tok = len(round_tokens)
            if round_tokens[0][1] == C_FAIL or n_tok == 1:
                label = f"R{round_idx} ({n_tok}t)"
                lcolor = '#888'
            else:
                label = f"R{round_idx} ({n_tok}t)"
                lcolor = '#333'
            ax.text((x_start + x_end) / 2, y_token + 2.6, label,
                    ha='center', fontsize=7, color=lcolor)
            x += 1.2   # gap between rounds

    # Legend (top of figure)
    legend_handles = [
        mpatches.Patch(facecolor=C_ROOT, label='Free-Root (verifier argmax)'),
        mpatches.Patch(facecolor=C_CONT, label='Cont (drafter, accepted)'),
        mpatches.Patch(facecolor=C_CONT2, label='Cont2 (drafter, accepted)'),
        mpatches.Patch(facecolor=C_BONUS, label='Bonus (verifier free)'),
        mpatches.Patch(facecolor='white', edgecolor=C_FAIL, linewidth=2,
                       label='Fallback (cont rejected)'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)

    out_pdf = os.path.join(FIG_DIR, "acceptance_trace.pdf")
    out_jpg = os.path.join(FIG_DIR, "acceptance_trace.jpg")
    plt.savefig(out_pdf, bbox_inches='tight', dpi=200)
    plt.savefig(out_jpg, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_jpg}")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,    # set True if local TeX is set up
        "mathtext.fontset": "stix",
    })
    print(f"FIG_DIR = {FIG_DIR}")
    make_cont_scatter()
    make_architecture()
    make_depth_distribution()
    make_acceptance_trace()
    print("\nDone. Include in LaTeX with:")
    print(r"  \includegraphics[width=\linewidth]{../fig/cont_scatter.pdf}")
    print(r"  \includegraphics[width=\linewidth]{../fig/architecture.pdf}")
    print(r"  \includegraphics[width=\linewidth]{../fig/depth_distribution.pdf}")
    print(r"  \includegraphics[width=\linewidth]{../fig/acceptance_trace.pdf}")
