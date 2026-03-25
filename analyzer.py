"""Parse torch.profiler traces and compute Roofline metrics."""

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

def extract_kernel_events(prof) -> pd.DataFrame:
    """Turn profiler key_averages into a DataFrame of CUDA kernel events."""
    key_avgs = prof.key_averages(group_by_input_shape=True)
    rows = []
    for evt in key_avgs:
        if evt.self_cuda_time_total == 0:
            continue
        rows.append({
            "kernel_name": evt.key,
            "cuda_time_us": evt.self_cuda_time_total,
            "cuda_time_ms": evt.self_cuda_time_total / 1000.0,
            "count": evt.count,
            "avg_time_ms": (evt.self_cuda_time_total / evt.count) / 1000.0,
            "flops": evt.flops if evt.flops else 0,
            "input_shapes": str(evt.input_shapes) if evt.input_shapes else "",
            "category": classify_kernel(evt.key),
        })
    return pd.DataFrame(rows)


# ── Memory bytes estimation ──────────────────────────────────────────────────

def _parse_shapes(shape_str: str):
    """Extract list of tensor shapes from the string repr."""
    # e.g. "[[128, 768], [768, 768], [768]]"
    try:
        import ast
        return ast.literal_eval(shape_str) if shape_str else []
    except Exception:
        return []


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


def estimate_memory_bytes(row) -> float:
    """Heuristic: estimate bytes transferred for a kernel from its shapes."""
    dtype_size = DTYPE_BYTES.get(config.DTYPE, 2)
    shapes = _parse_shapes(row["input_shapes"])
    if not shapes:
        return 0.0

    cat = row["category"]
    total_elements = sum(_numel(s) for s in shapes if isinstance(s, (list, tuple)) and len(s) > 0)

    if cat == "gemm":
        # Read inputs + write output
        return dtype_size * total_elements
    elif cat == "attention":
        # Q, K, V read + O write ≈ 4× the largest tensor
        return dtype_size * total_elements
    else:
        # Elementwise / norm: read + write ≈ 2× elements
        return dtype_size * total_elements * 2


# ── Roofline metrics ────────────────────────────────────────────────────────

def compute_roofline_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add arithmetic intensity, achieved GFLOPS, and roofline bound."""
    gpu = config.get_gpu_specs()
    peak_gflops = gpu["peak_flops_fp16"] / 1e9
    peak_bw = gpu["peak_bw_bytes"]

    df = df.copy()
    df["bytes_transferred"] = df.apply(estimate_memory_bytes, axis=1)

    # Arithmetic intensity (FLOPS / byte)
    df["arithmetic_intensity"] = np.where(
        df["bytes_transferred"] > 0,
        df["flops"] / df["bytes_transferred"],
        0.0,
    )

    # Achieved GFLOPS
    time_s = df["cuda_time_us"] / 1e6
    df["achieved_gflops"] = np.where(time_s > 0, df["flops"] / time_s / 1e9, 0.0)

    # Roofline bound
    df["roofline_bound_gflops"] = np.minimum(
        peak_gflops,
        peak_bw * df["arithmetic_intensity"] / 1e9,
    )

    # Efficiency
    df["efficiency"] = np.where(
        df["roofline_bound_gflops"] > 0,
        df["achieved_gflops"] / df["roofline_bound_gflops"],
        0.0,
    )

    return df


# ── Phase summary ────────────────────────────────────────────────────────────

def compute_phase_summary(df: pd.DataFrame, phase: str, token_count: int) -> dict:
    """Aggregate kernel-level data into a single phase summary."""
    gpu = config.get_gpu_specs()
    total_time_ms = df["cuda_time_ms"].sum()
    total_time_s = total_time_ms / 1000.0
    total_flops = df["flops"].sum()
    total_bytes = df["bytes_transferred"].sum()

    achieved_tflops = total_flops / total_time_s / 1e12 if total_time_s > 0 else 0
    bw_achieved = total_bytes / total_time_s if total_time_s > 0 else 0

    return {
        "phase": phase,
        "token_count": token_count,
        "total_latency_ms": total_time_ms,
        "total_flops": total_flops,
        "achieved_tflops": achieved_tflops,
        "compute_utilization": achieved_tflops / (gpu["peak_flops_fp16"] / 1e12),
        "memory_bw_achieved_gbs": bw_achieved / 1e9,
        "memory_bw_utilization": bw_achieved / gpu["peak_bw_bytes"],
        "tokens_per_second": token_count / total_time_s if total_time_s > 0 else 0,
    }

