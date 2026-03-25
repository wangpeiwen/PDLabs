"""Experiment configuration for Prefill vs Decode profiling."""

import os

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/opt-125m"
DTYPE = "float16"
ENFORCE_EAGER = True          # Disable CUDA graphs for accurate timing
GPU_MEMORY_UTILIZATION = 0.85 # Leave headroom for profiler overhead

# ── Profiling parameters ─────────────────────────────────────────────────────
PREFILL_PROMPT_LENGTHS = [32, 64, 128, 256, 512, 1024, 1536]
DECODE_LENGTHS = [32, 64, 128, 256, 512, 1024]
DECODE_SHORT_PROMPT_LEN = 16  # Short prompt used during decode profiling
WARMUP_ITERATIONS = 3
MAX_MODEL_LEN = None  # Auto-detected at runtime from model config

# ── GPU hardware specs (for Roofline model) ──────────────────────────────────
GPU_SPECS = {
    "A100_SXM": {
        "name": "NVIDIA A100 SXM 80GB",
        "peak_flops_fp16": 312e12,   # 312 TFLOPS
        "peak_bw_bytes": 2039e9,     # 2039 GB/s
    },
    "A800_SXM": {
        "name": "NVIDIA A800 SXM 80GB",
        "peak_flops_fp16": 312e12,   # 312 TFLOPS (same as A100)
        "peak_bw_bytes": 2039e9,     # 2039 GB/s HBM2e (same as A100)
    },
    "RTX_4090": {
        "name": "NVIDIA RTX 4090",
        "peak_flops_fp16": 165.2e12, # 165.2 TFLOPS
        "peak_bw_bytes": 1008e9,     # 1008 GB/s
    },
    "RTX_3090": {
        "name": "NVIDIA RTX 3090",
        "peak_flops_fp16": 71e12,    # 71 TFLOPS
        "peak_bw_bytes": 936.2e9,    # 936.2 GB/s
    },
    "H100_SXM": {
        "name": "NVIDIA H100 SXM",
        "peak_flops_fp16": 989.5e12, # 989.5 TFLOPS
        "peak_bw_bytes": 3350e9,     # 3350 GB/s
    },
}

ACTIVE_GPU = "A800_SXM"  # Change to match your hardware

# ── Model architecture (for theoretical FLOPs/bytes calculation) ─────────
# These are auto-detected at runtime from the model config.
# Set to None here; profiler.py will populate them after loading the model.
MODEL_ARCH = {
    "n_layers": None,
    "hidden_size": None,
    "n_heads": None,
    "head_dim": None,
    "intermediate_size": None,  # FFN intermediate dim
    "vocab_size": None,
}

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ── Derived ──────────────────────────────────────────────────────────────────
def get_gpu_specs():
    specs = GPU_SPECS[ACTIVE_GPU]
    specs["ridge_point"] = specs["peak_flops_fp16"] / specs["peak_bw_bytes"]
    return specs
