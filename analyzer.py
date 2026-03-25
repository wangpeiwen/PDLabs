"""Parse torch.profiler traces and compute Roofline metrics.

Strategy: torch.profiler's with_flops does not work for custom CUDA kernels
(flash attention, cutlass, etc.) used by vLLM. Instead we:
  1. Use profiler ONLY for kernel-level timing and classification
  2. Compute theoretical FLOPs and memory bytes from model architecture
  3. Distribute theoretical totals across kernels proportional to their time
"""

import re
import numpy as np
import pandas as pd

import config

# ── Kernel classification ────────────────────────────────────────────────────

_KERNEL_PATTERNS = [
    (r"(mm|gemm|cublas|cutlass)", "gemm"),
    (r"(flash|sdpa|attention|attn|fmha)", "attention"),
    (r"(layer_?norm|LayerNorm|rms_?norm)", "layernorm"),
    (r"(gelu|relu|silu|swiglu|activation)", "activation"),
    (r"(softmax)", "softmax"),
    (r"(elementwise|add_|mul_|copy_)", "elementwise"),
]


def classify_kernel(name: str) -> str:
    low = name.lower()
    for pattern, category in _KERNEL_PATTERNS:
        if re.search(pattern, low):
            return category
    return "other"


# ── Extract kernel events ────────────────────────────────────────────────────

def _get_cuda_time(evt) -> float:
    """Extract CUDA/device time from a profiler event, across torch versions."""
    for attr in ("self_device_time_total", "self_cuda_time_total",
                 "device_time_total", "cuda_time_total"):
        val = getattr(evt, attr, None)
        if val is not None and val > 0:
            return float(val)
    return float(getattr(evt, "self_cpu_time_total", 0))


_EMPTY_COLUMNS = [
    "kernel_name", "cuda_time_us", "cuda_time_ms", "count",
    "avg_time_ms", "flops", "bytes_transferred", "arithmetic_intensity",
    "achieved_gflops", "roofline_bound_gflops", "efficiency",
    "input_shapes", "category",
]


def extract_kernel_events(prof) -> pd.DataFrame:
    """Turn profiler key_averages into a DataFrame of CUDA kernel events."""
    key_avgs = prof.key_averages(group_by_input_shape=True)

    if key_avgs:
        sample = key_avgs[0]
        time_attrs = [a for a in dir(sample)
                      if ("time" in a or "cuda" in a or "device" in a)
                      and not a.startswith("_")]
        print(f"    [debug] FunctionEventAvg attrs: {time_attrs}")
        print(f"    [debug] Total events from key_averages: {len(key_avgs)}")

    rows = []
    for evt in key_avgs:
        cuda_time = _get_cuda_time(evt)
        if cuda_time == 0:
            continue
        rows.append({
            "kernel_name": evt.key,
            "cuda_time_us": cuda_time,
            "cuda_time_ms": cuda_time / 1000.0,
            "count": evt.count,
            "avg_time_ms": (cuda_time / evt.count) / 1000.0,
            "input_shapes": str(evt.input_shapes) if hasattr(evt, "input_shapes") and evt.input_shapes else "",
            "category": classify_kernel(evt.key),
        })

    print(f"    [debug] Extracted {len(rows)} kernel events with non-zero time")

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)
    return pd.DataFrame(rows)


# ── Theoretical FLOPs / Bytes calculation ────────────────────────────────────

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


def compute_theoretical_flops(seq_len: int, phase: str) -> float:
    """Compute theoretical FLOPs for one forward pass.

    Prefill: batch=1, all tokens in parallel.
    Decode:  batch=1, single new token, attends to full KV cache.
    """
    arch = config.MODEL_ARCH
    if not arch.get("hidden_size"):
        return 0.0

    B, L = 1, arch["n_layers"]
    h, n_h, d_h = arch["hidden_size"], arch["n_heads"], arch["head_dim"]
    ffn = arch["intermediate_size"]

    if phase == "prefill":
        S, S_kv = seq_len, seq_len
    else:
        S, S_kv = 1, seq_len

    per_layer = (
        6 * B * S * h * h
        + 2 * B * n_h * S * S_kv * d_h
        + 2 * B * S * h * h
        + 4 * B * S * h * ffn
    )
    return float(L * per_layer)


def compute_theoretical_bytes(seq_len: int, phase: str) -> float:
    """Compute theoretical memory bytes transferred (weights + activations)."""
    arch = config.MODEL_ARCH
    if not arch.get("hidden_size"):
        return 0.0

    dtype_size = DTYPE_BYTES.get(config.DTYPE, 2)
    L, h, ffn = arch["n_layers"], arch["hidden_size"], arch["intermediate_size"]

    weight_bytes_per_layer = dtype_size * (4 * h * h + 2 * h * ffn)
    total_weight_bytes = L * weight_bytes_per_layer

    S = seq_len if phase == "prefill" else 1
    activation_bytes_per_layer = dtype_size * S * h * 8
    total_activation_bytes = L * activation_bytes_per_layer

    return float(total_weight_bytes + total_activation_bytes)


# ── Roofline metrics ────────────────────────────────────────────────────────

def compute_roofline_metrics(df, seq_len, phase):
    """Add FLOPs, bytes, AI, achieved GFLOPS per kernel (theoretical distribution)."""
    if df.empty:
        for col in ("flops", "bytes_transferred", "arithmetic_intensity",
                     "achieved_gflops", "roofline_bound_gflops", "efficiency"):
            df[col] = pd.Series(dtype="float64")
        return df

    gpu = config.get_gpu_specs()
    peak_gflops = gpu["peak_flops_fp16"] / 1e9
    peak_bw = gpu["peak_bw_bytes"]

    df = df.copy()
    total_time = df["cuda_time_us"].sum()
    total_flops = compute_theoretical_flops(seq_len, phase)
    total_bytes = compute_theoretical_bytes(seq_len, phase)

    time_frac = df["cuda_time_us"] / total_time if total_time > 0 else 0

    # GEMM/attention get FLOPs; others are memory-bound
    is_compute = df["category"].isin(["gemm", "attention"])
    compute_time = df.loc[is_compute, "cuda_time_us"].sum()

    df["flops"] = 0.0
    if compute_time > 0:
        df.loc[is_compute, "flops"] = (
            total_flops * df.loc[is_compute, "cuda_time_us"] / compute_time
        )

    df["bytes_transferred"] = total_bytes * time_frac

    df["arithmetic_intensity"] = np.where(
        df["bytes_transferred"] > 0, df["flops"] / df["bytes_transferred"], 0.0)

    time_s = df["cuda_time_us"] / 1e6
    df["achieved_gflops"] = np.where(time_s > 0, df["flops"] / time_s / 1e9, 0.0)

    df["roofline_bound_gflops"] = np.minimum(
        peak_gflops, peak_bw * df["arithmetic_intensity"] / 1e9)

    df["efficiency"] = np.where(
        df["roofline_bound_gflops"] > 0,
        df["achieved_gflops"] / df["roofline_bound_gflops"], 0.0)

    return df


# ── Phase summary ────────────────────────────────────────────────────────────

def compute_phase_summary(df, phase, token_count, seq_len):
    """Aggregate kernel-level data into a single phase summary."""
    gpu = config.get_gpu_specs()
    total_time_ms = df["cuda_time_ms"].sum() if not df.empty else 0
    total_time_s = total_time_ms / 1000.0

    total_flops = compute_theoretical_flops(seq_len, phase)
    total_bytes = compute_theoretical_bytes(seq_len, phase)

    achieved_tflops = total_flops / total_time_s / 1e12 if total_time_s > 0 else 0
    bw_achieved = total_bytes / total_time_s if total_time_s > 0 else 0

    return {
        "phase": phase,
        "token_count": token_count,
        "seq_len": seq_len,
        "total_latency_ms": total_time_ms,
        "total_flops": total_flops,
        "achieved_tflops": achieved_tflops,
        "compute_utilization": achieved_tflops / (gpu["peak_flops_fp16"] / 1e12),
        "memory_bw_achieved_gbs": bw_achieved / 1e9,
        "memory_bw_utilization": bw_achieved / gpu["peak_bw_bytes"],
        "tokens_per_second": token_count / total_time_s if total_time_s > 0 else 0,
    }
