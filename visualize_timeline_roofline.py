"""Figure 1: Latency scaling (top) + Roofline Model (bottom)."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config


def plot_latency_scaling(ax_pf, ax_dc, summaries):
    """Bar charts showing latency scaling across configurations."""
    prefill = [s for s in summaries if s["phase"] == "prefill"]
    decode = [s for s in summaries if s["phase"] == "decode"]

    # Prefill: latency vs prompt length
    if prefill:
        labels = [str(s["token_count"]) for s in prefill]
        times = [s["elapsed_ms"] for s in prefill]
        bars = ax_pf.bar(labels, times, color="#4C72B0", edgecolor="white", width=0.6)
        for bar, t in zip(bars, times):
            ax_pf.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                       f"{t:.1f}", ha="center", va="bottom", fontsize=8)
        ax_pf.set_xlabel("Prompt Length (tokens)")
        ax_pf.set_ylabel("Latency (ms)")
        ax_pf.set_title("Prefill Latency", fontweight="bold")

    # Decode: latency vs decode length
    if decode:
        labels = [str(s["token_count"]) for s in decode]
        times = [s["elapsed_ms"] for s in decode]
        bars = ax_dc.bar(labels, times, color="#DD8452", edgecolor="white", width=0.6)
        for bar, t in zip(bars, times):
            ax_dc.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                       f"{t:.1f}", ha="center", va="bottom", fontsize=8)
        ax_dc.set_xlabel("Decode Length (tokens)")
        ax_dc.set_ylabel("Latency (ms)")
        ax_dc.set_title("Decode Latency", fontweight="bold")


def plot_roofline(ax, summaries):
    """Log-log Roofline model with prefill/decode points."""
    gpu = config.get_gpu_specs()
    peak_gflops = gpu["peak_flops_fp16"] / 1e9
    peak_bw_gbs = gpu["peak_bw_bytes"] / 1e9
    ridge = gpu["ridge_point"]

    # Roofline ceiling
    ai_range = np.logspace(-1, 4, 500)
    roofline = np.minimum(peak_gflops, peak_bw_gbs * ai_range)
    ax.plot(ai_range, roofline, "k-", linewidth=2.5, label="Roofline ceiling")

    # Ridge point
    ax.axvline(ridge, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.annotate(f"Ridge = {ridge:.0f}", xy=(ridge, peak_gflops * 0.5),
                fontsize=8, color="gray", ha="left", rotation=90)

    # Shade regions
    ax.fill_between(ai_range, 0.1, roofline, where=(ai_range < ridge),
                     alpha=0.06, color="red")
    ax.fill_between(ai_range, 0.1, roofline, where=(ai_range >= ridge),
                     alpha=0.06, color="blue")
    ax.text(ridge * 0.1, peak_gflops * 0.02, "Memory-bound",
            fontsize=9, color="#CC4444", fontstyle="italic")
    ax.text(ridge * 3, peak_gflops * 0.02, "Compute-bound",
            fontsize=9, color="#4444CC", fontstyle="italic")

    # Plot each summary as a point
    for s in summaries:
        ai = s["arithmetic_intensity"]
        achieved = s["achieved_tflops"] * 1e3  # TFLOPS → GFLOPS
        if ai <= 0 or achieved <= 0:
            continue
        color = "#4C72B0" if s["phase"] == "prefill" else "#DD8452"
        marker = "o" if s["phase"] == "prefill" else "^"
        label_text = f"{s['phase']} len={s['token_count']}"
        ax.scatter(ai, achieved, c=color, marker=marker, s=120,
                   edgecolors="white", linewidth=0.8, zorder=5)
        ax.annotate(label_text, (ai, achieved), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    # Legend
    ax.scatter([], [], c="#4C72B0", marker="o", s=80, label="Prefill")
    ax.scatter([], [], c="#DD8452", marker="^", s=80, label="Decode")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPS/Byte)", fontsize=10)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=10)
    ax.set_title("Roofline Model — Prefill vs Decode", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(1e-1, 1e4)
    ax.set_ylim(1e-1, peak_gflops * 2)
    ax.grid(True, which="both", alpha=0.2)


def create_figure(summaries, save_path=None):
    """Produce the combined Latency + Roofline figure."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.3], hspace=0.35, wspace=0.3)

    ax_pf = fig.add_subplot(gs[0, 0])
    ax_dc = fig.add_subplot(gs[0, 1])
    plot_latency_scaling(ax_pf, ax_dc, summaries)

    ax_rf = fig.add_subplot(gs[1, :])
    plot_roofline(ax_rf, summaries)

    gpu = config.get_gpu_specs()
    fig.suptitle(f"Prefill vs Decode — {gpu['name']}  |  {config.MODEL_NAME}",
                 fontsize=13, fontweight="bold", y=0.98)

    path = save_path or f"{config.OUTPUT_DIR}/timeline_roofline.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
