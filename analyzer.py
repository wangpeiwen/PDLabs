"""Compute theoretical FLOPs, memory bytes, and Roofline metrics
from model architecture + measured wall-clock time.

No dependency on torch.profiler — uses CUDA event timing from profiler.py.
"""

import numpy as np
import pandas as pd

import config

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


# ── Theoretical FLOPs ───────────────────────────────────────────────────────

def compute_theoretical_flops(seq_len: int, phase: str) -> float:
    """Per-token-generation FLOPs for one forward pass.

    Prefill: batch=1, S tokens in parallel, KV cache length = S.
    Decode:  batch=1, 1 new token, KV cache length = seq_len.
    """
    arch = config.MODEL_ARCH
    h = arch.get("hidden_size") or 0
    if h == 0:
        return 0.0

    L = arch["n_layers"]
    n_h = arch["n_heads"]
    d_h = arch["head_dim"]
    ffn = arch["intermediate_size"]

    S = seq_len if phase == "prefill" else 1
    S_kv = seq_len

    per_layer = (
        6 * S * h * h                  # QKV projections (3 matrices, each 2*S*h*h)
        + 2 * n_h * S * S_kv * d_h     # attention: QK^T + attn@V
        + 2 * S * h * h                # output projection
        + 4 * S * h * ffn              # FFN up + down
    )
    return float(L * per_layer)


# ── Theoretical memory bytes ─────────────────────────────────────────────────

def compute_theoretical_bytes(seq_len: int, phase: str) -> float:
    """Bytes transferred = weight reads + KV cache reads + activation I/O.

    Decode is weight-dominated (read all weights for 1 token).
    Prefill amortizes weight reads across S tokens.
    """
    arch = config.MODEL_ARCH
    h = arch.get("hidden_size") or 0
    if h == 0:
        return 0.0

    dtype_size = DTYPE_BYTES.get(config.DTYPE, 2)
    L = arch["n_layers"]
    n_h = arch["n_heads"]
    d_h = arch["head_dim"]
    ffn = arch["intermediate_size"]

    S = seq_len if phase == "prefill" else 1
    S_kv = seq_len

    # Weight bytes per layer: QKV(3*h*h) + O(h*h) + FFN_up(h*ffn) + FFN_down(ffn*h)
    weight_per_layer = dtype_size * (4 * h * h + 2 * h * ffn)

    # KV cache read per layer: 2 * S_kv * n_h * d_h (K and V)
    kv_cache_per_layer = dtype_size * 2 * S_kv * n_h * d_h

    # Activation I/O per layer (rough): read+write ~6 tensors of [S, h]
    activation_per_layer = dtype_size * S * h * 6

    total = L * (weight_per_layer + kv_cache_per_layer + activation_per_layer)
    return float(total)


# ── Phase summary ────────────────────────────────────────────────────────────

def compute_phase_summary(phase: str, token_count: int, seq_len: int,
                          elapsed_ms: float) -> dict:
    """Build a summary dict for one profiling run."""
    gpu = config.get_gpu_specs()
    elapsed_s = elapsed_ms / 1000.0

    if phase == "prefill":
        # Prefill processes all seq_len tokens in one pass
        total_flops = compute_theoretical_flops(seq_len, "prefill")
        total_bytes = compute_theoretical_bytes(seq_len, "prefill")
    else:
        # Decode: each of the token_count steps does 1-token forward
        # Average KV cache length ≈ seq_len / 2 (grows from short_prompt to seq_len)
        avg_kv = (config.DECODE_SHORT_PROMPT_LEN + seq_len) / 2
        flops_per_step = compute_theoretical_flops(avg_kv, "decode")
        bytes_per_step = compute_theoretical_bytes(avg_kv, "decode")
        total_flops = flops_per_step * token_count
        total_bytes = bytes_per_step * token_count

    achieved_tflops = total_flops / elapsed_s / 1e12 if elapsed_s > 0 else 0
    bw_achieved = total_bytes / elapsed_s if elapsed_s > 0 else 0

    summary = {
        "phase": phase,
        "token_count": token_count,
        "seq_len": seq_len,
        "elapsed_ms": elapsed_ms,
        "total_flops": total_flops,
        "achieved_tflops": achieved_tflops,
        "compute_utilization": achieved_tflops / (gpu["peak_flops_fp16"] / 1e12),
        "memory_bw_achieved_gbs": bw_achieved / 1e9,
        "memory_bw_utilization": bw_achieved / gpu["peak_bw_bytes"],
        "tokens_per_second": token_count / elapsed_s if elapsed_s > 0 else 0,
        # Arithmetic intensity for Roofline
        "arithmetic_intensity": total_flops / total_bytes if total_bytes > 0 else 0,
    }

    print(f"    {phase} seq_len={seq_len}: {elapsed_ms:.1f}ms, "
          f"{achieved_tflops:.2f} TFLOPS, "
          f"compute={summary['compute_utilization']*100:.1f}%, "
          f"BW={summary['memory_bw_achieved_gbs']:.1f} GB/s "
          f"({summary['memory_bw_utilization']*100:.1f}%), "
          f"AI={summary['arithmetic_intensity']:.1f} FLOPS/B")

    return summary


def compute_all_summaries(results: dict) -> list[dict]:
    """Process all profiling results into summaries."""
    summaries = []

    for r in results["prefill"]:
        plen = r["prompt_length"]
        s = compute_phase_summary("prefill", plen, plen, r["elapsed_ms"])
        summaries.append(s)

    for r in results["decode"]:
        dlen = r["decode_length"]
        kv_len = config.DECODE_SHORT_PROMPT_LEN + dlen
        s = compute_phase_summary("decode", dlen, kv_len, r["elapsed_ms"])
        summaries.append(s)

    return summaries
