"""Figure 1: Execution Timeline (top) + Roofline Model (bottom)."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config

# ── Color palette for kernel categories ──────────────────────────────────────
CATEGORY_COLORS = {
    "gemm": "#4C72B0",
    "attention": "#DD8452",
    "layernorm": "#55A868",
    "activation": "#C44E52",
    "softmax": "#8172B3",
    "elementwise": "#937860",
    "other": "#AAAAAA",
}


def plot_timeline(ax, df, phase_label: str):
    """Horizontal stacked bar chart showing kernel execution over time."""
    # Sort by time contribution, keep top-N for readability
    df_sorted = df.nlargest(20, "cuda_time_ms")
    categories = df_sorted["category"].values
    times = df_sorted["cuda_time_ms"].values
    names = df_sorted["kernel_name"].apply(lambda n: n[:40]).values

    colors = [CATEGORY_COLORS.get(c, "#AAAAAA") for c in categories]

    # Waterfall: each bar starts where the previous ended
    lefts = np.zeros(len(times))
    lefts[1:] = np.cumsum(times[:-1])

    y_pos = np.arange(len(times))
    ax.barh(y_pos, times, left=lefts, color=colors, height=0.7, edgecolor="white", linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Cumulative Time (ms)", fontsize=9)
    ax.set_title(f"{phase_label} — Top-20 Kernels by Time", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    total = df["cuda_time_ms"].sum()
    ax.annotate(f"Total: {total:.1f} ms", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", fontsize=9, fontstyle="italic")


def plot_roofline(ax, prefill_df, decode_df):
    """Log-log Roofline model with prefill/decode kernel scatter."""
    gpu = config.get_gpu_specs()
    peak_gflops = gpu["peak_flops_fp16"] / 1e9
    peak_bw_gbs = gpu["peak_bw_bytes"] / 1e9  # GB/s → for slope
    ridge = gpu["ridge_point"]

    # Roofline ceiling
    ai_range = np.logspace(-2, 4, 500)
    roofline = np.minimum(peak_gflops, peak_bw_gbs * ai_range)
    ax.plot(ai_range, roofline, "k-", linewidth=2, label="Roofline ceiling")

    # Ridge point
    ax.axvline(ridge, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.annotate(f"Ridge = {ridge:.0f}", xy=(ridge, peak_gflops * 0.6),
                fontsize=8, color="gray", ha="left", rotation=90)

    # Shade regions
    ax.fill_between(ai_range, 0, roofline, where=(ai_range < ridge),
                     alpha=0.05, color="red", label="Memory-bound")
    ax.fill_between(ai_range, 0, roofline, where=(ai_range >= ridge),
                     alpha=0.05, color="blue", label="Compute-bound")

    # Scatter kernels
    def _scatter(df, color, marker, label):
        valid = df[(df["arithmetic_intensity"] > 0) & (df["achieved_gflops"] > 0)]
        if valid.empty:
            return
        sizes = np.clip(valid["cuda_time_ms"] * 5, 10, 300)
        ax.scatter(valid["arithmetic_intensity"], valid["achieved_gflops"],
                   c=color, marker=marker, s=sizes, alpha=0.7, edgecolors="white",
                   linewidth=0.5, label=label, zorder=5)

    _scatter(prefill_df, "#4C72B0", "o", "Prefill kernels")
    _scatter(decode_df, "#DD8452", "^", "Decode kernels")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPS/Byte)", fontsize=10)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=10)
    ax.set_title("Roofline Model — Prefill vs Decode", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e-1, peak_gflops * 2)
    ax.grid(True, which="both", alpha=0.2)


def create_figure(prefill_df, decode_df, save_path=None):
    """Produce the combined Timeline + Roofline figure."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

    # Top-left: prefill timeline
    ax_pf = fig.add_subplot(gs[0, 0])
    plot_timeline(ax_pf, prefill_df, "Prefill")

    # Top-right: decode timeline
    ax_dc = fig.add_subplot(gs[0, 1])
    plot_timeline(ax_dc, decode_df, "Decode")

    # Bottom: roofline (spans both columns)
    ax_rf = fig.add_subplot(gs[1, :])
    plot_roofline(ax_rf, prefill_df, decode_df)

    # Legend for kernel categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=k) for k, c in CATEGORY_COLORS.items()]
    fig.legend(handles=legend_elements, loc="upper center", ncol=len(CATEGORY_COLORS),
               fontsize=8, title="Kernel Category", title_fontsize=9,
               bbox_to_anchor=(0.5, 0.98))

    path = save_path or f"{config.OUTPUT_DIR}/timeline_roofline.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

