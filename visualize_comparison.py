"""Figure 2: Multi-dimensional bar chart comparison of Prefill vs Decode."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config


def create_figure(phase_summaries: list[dict], save_path=None):
    """2x2 grouped bar chart comparing prefill vs decode across 4 metrics.

    Args:
        phase_summaries: list of dicts from analyzer.compute_phase_summary,
                         each containing 'phase', metric values, and a label key.
    """
    # Separate by phase
    prefill = [s for s in phase_summaries if s["phase"] == "prefill"]
    decode = [s for s in phase_summaries if s["phase"] == "decode"]

    # Use the first entry of each for the primary comparison
    pf = prefill[0] if prefill else {}
    dc = decode[0] if decode else {}

    metrics = [
        ("elapsed_ms", "Latency (ms)", "Total Phase Latency"),
        ("tokens_per_second", "Tokens / s", "Throughput"),
        ("memory_bw_utilization", "Utilization (%)", "Memory Bandwidth Utilization"),
        ("compute_utilization", "Utilization (%)", "Compute Utilization"),
    ]

    palette = {"Prefill": "#4C72B0", "Decode": "#DD8452"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (key, ylabel, title) in zip(axes, metrics):
        pf_val = pf.get(key, 0)
        dc_val = dc.get(key, 0)

        # Convert utilization to percentage
        if "utilization" in key:
            pf_val *= 100
            dc_val *= 100

        x = np.arange(2)
        bars = ax.bar(x, [pf_val, dc_val],
                      color=[palette["Prefill"], palette["Decode"]],
                      width=0.5, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Prefill", "Decode"], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Value labels
        for bar, val in zip(bars, [pf_val, dc_val]):
            fmt = f"{val:.1f}" if val >= 1 else f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylim(0, max(pf_val, dc_val, 1e-6) * 1.25)
        sns.despine(ax=ax)

    # If we have multiple configs, add a sweep subplot
    if len(prefill) > 1 or len(decode) > 1:
        _add_sweep_annotation(fig, prefill, decode)

    gpu = config.get_gpu_specs()
    fig.suptitle(f"Prefill vs Decode — {gpu['name']}  |  {config.MODEL_NAME}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = save_path or f"{config.OUTPUT_DIR}/comparison_bars.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _add_sweep_annotation(fig, prefill_list, decode_list):
    """Add a text box summarizing how metrics scale across configs."""
    lines = ["Config sweep summary:"]
    for s in prefill_list:
        lines.append(f"  Prefill len={s['token_count']}: {s['elapsed_ms']:.1f}ms, "
                     f"{s['compute_utilization']*100:.1f}% compute")
    for s in decode_list:
        lines.append(f"  Decode  len={s['token_count']}: {s['elapsed_ms']:.1f}ms, "
                     f"{s['memory_bw_utilization']*100:.1f}% BW")
    fig.text(0.02, -0.02, "\n".join(lines), fontsize=7, family="monospace",
             va="top", transform=fig.transFigure)
