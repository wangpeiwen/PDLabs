"""Roofline Model figure — publication quality, single focused plot."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

import config


# ── Color scheme ─────────────────────────────────────────────────────────────
_PREFILL_COLOR = "#2563EB"   # vivid blue
_DECODE_COLOR = "#DC2626"    # vivid red
_CEILING_COLOR = "#1E293B"   # dark slate
_RIDGE_COLOR = "#94A3B8"     # muted gray
_BG_COLOR = "#FAFBFC"
_GRID_COLOR = "#E2E8F0"


def create_figure(summaries, save_path=None):
    """Single Roofline figure with all prefill/decode data points."""
    gpu = config.get_gpu_specs()
    peak_gflops = gpu["peak_flops_fp16"] / 1e9
    peak_bw_gbs = gpu["peak_bw_bytes"] / 1e9
    ridge = gpu["ridge_point"]

    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    # ── Roofline ceiling ─────────────────────────────────────────────────────
    ai = np.logspace(-1, 4, 1000)
    roof = np.minimum(peak_gflops, peak_bw_gbs * ai)
    ax.plot(ai, roof, color=_CEILING_COLOR, linewidth=2.8, solid_capstyle="round",
            zorder=3)

    # Ceiling labels
    # Memory-bound slope label
    slope_x = ridge * 0.05
    slope_y = peak_bw_gbs * slope_x * 0.8
    ax.annotate(f"Peak BW = {peak_bw_gbs:.0f} GB/s",
                xy=(slope_x, slope_y), fontsize=8.5, color=_CEILING_COLOR,
                rotation=38, ha="center", va="bottom", fontstyle="italic")
    # Compute-bound flat label
    ax.annotate(f"Peak FP16 = {peak_gflops/1e3:.0f} TFLOPS",
                xy=(ridge * 8, peak_gflops * 1.08), fontsize=8.5,
                color=_CEILING_COLOR, ha="center", va="bottom", fontstyle="italic")

    # ── Ridge point ──────────────────────────────────────────────────────────
    ax.axvline(ridge, color=_RIDGE_COLOR, linestyle=":", linewidth=1.2,
               alpha=0.8, zorder=2)
    ax.scatter([ridge], [peak_gflops], marker="D", s=50, color=_RIDGE_COLOR,
               zorder=4, edgecolors="white", linewidth=0.8)
    ax.annotate(f"Ridge Point\nAI = {ridge:.0f}",
                xy=(ridge, peak_gflops), xytext=(ridge * 1.5, peak_gflops * 0.35),
                fontsize=8, color=_RIDGE_COLOR, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=_RIDGE_COLOR,
                                lw=0.8, connectionstyle="arc3,rad=-0.2"))

    # ── Shaded regions ───────────────────────────────────────────────────────
    ax.fill_between(ai, 0.01, roof, where=(ai < ridge),
                     alpha=0.04, color=_DECODE_COLOR, zorder=1)
    ax.fill_between(ai, 0.01, roof, where=(ai >= ridge),
                     alpha=0.04, color=_PREFILL_COLOR, zorder=1)

    # Region labels with text outline for readability
    txt_effect = [pe.withStroke(linewidth=3, foreground=_BG_COLOR)]
    ax.text(ridge * 0.03, peak_gflops * 0.4, "Memory\nBound",
            fontsize=14, color=_DECODE_COLOR, alpha=0.35, fontweight="bold",
            ha="center", va="center", path_effects=txt_effect)
    ax.text(ridge * 30, peak_gflops * 0.4, "Compute\nBound",
            fontsize=14, color=_PREFILL_COLOR, alpha=0.35, fontweight="bold",
            ha="center", va="center", path_effects=txt_effect)

    # ── Data points ──────────────────────────────────────────────────────────
    prefill = [s for s in summaries if s["phase"] == "prefill"]
    decode = [s for s in summaries if s["phase"] == "decode"]

    def _plot_phase(data, color, marker, label_prefix):
        ais = [s["arithmetic_intensity"] for s in data]
        perfs = [s["achieved_tflops"] * 1e3 for s in data]  # → GFLOPS
        sizes = [s["token_count"] for s in data]

        # Scale marker size by token count (log scale for visual balance)
        size_arr = np.array(sizes, dtype=float)
        marker_sizes = 60 + 200 * (np.log2(size_arr) - np.log2(size_arr.min())) / \
                       max(np.log2(size_arr.max()) - np.log2(size_arr.min()), 1)

        for i, s in enumerate(data):
            if ais[i] <= 0 or perfs[i] <= 0:
                continue
            ax.scatter(ais[i], perfs[i], c=color, marker=marker,
                       s=marker_sizes[i], edgecolors="white", linewidth=1.0,
                       alpha=0.9, zorder=6)
            # Label with token count
            ax.annotate(f"{s['token_count']}",
                        (ais[i], perfs[i]),
                        fontsize=7, color=color, fontweight="bold",
                        xytext=(6, -4), textcoords="offset points",
                        path_effects=[pe.withStroke(linewidth=2.5,
                                                     foreground=_BG_COLOR)])

    _plot_phase(prefill, _PREFILL_COLOR, "o", "Prefill")
    _plot_phase(decode, _DECODE_COLOR, "^", "Decode")

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity  (FLOP / Byte)", fontsize=11,
                  labelpad=8)
    ax.set_ylabel("Attainable Performance  (GFLOP/s)", fontsize=11,
                  labelpad=8)
    ax.set_xlim(5e-2, 2e4)
    ax.set_ylim(5e-1, peak_gflops * 3)

    # Grid
    ax.grid(True, which="major", color=_GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", color=_GRID_COLOR, linewidth=0.4, alpha=0.4)

    # Spines
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#475569", labelsize=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_PREFILL_COLOR,
               markersize=10, markeredgecolor="white", markeredgewidth=0.8,
               label="Prefill  (compute-bound)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=_DECODE_COLOR,
               markersize=10, markeredgecolor="white", markeredgewidth=0.8,
               label="Decode  (memory-bound)"),
        Line2D([0], [0], color=_CEILING_COLOR, linewidth=2.5,
               label="Roofline ceiling"),
    ]
    legend = ax.legend(handles=legend_elements, loc="lower right", fontsize=9.5,
                       frameon=True, fancybox=True, framealpha=0.9,
                       edgecolor="#CBD5E1")
    legend.get_frame().set_facecolor(_BG_COLOR)

    # ── Title ────────────────────────────────────────────────────────────────
    model_short = config.MODEL_NAME.split("/")[-1]
    ax.set_title(f"Roofline Analysis: Prefill vs Decode\n"
                 f"{gpu['name']}  ·  {model_short}  ·  FP16",
                 fontsize=13, fontweight="bold", color="#1E293B", pad=16)

    # ── Annotation box: summary stats ────────────────────────────────────────
    if prefill and decode:
        pf_best = max(prefill, key=lambda s: s["compute_utilization"])
        dc_best = max(decode, key=lambda s: s["memory_bw_utilization"])
        info = (f"Prefill (len={pf_best['token_count']}): "
                f"{pf_best['compute_utilization']*100:.1f}% compute util, "
                f"AI={pf_best['arithmetic_intensity']:.0f}\n"
                f"Decode  (len={dc_best['token_count']}): "
                f"{dc_best['memory_bw_utilization']*100:.1f}% BW util, "
                f"AI={dc_best['arithmetic_intensity']:.1f}")
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=7.5,
                color="#64748B", family="monospace", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=_BG_COLOR,
                          edgecolor="#CBD5E1", alpha=0.9))

    fig.tight_layout()
    path = save_path or f"{config.OUTPUT_DIR}/roofline.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=_BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")
