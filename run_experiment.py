#!/usr/bin/env python3
"""Main entry point: orchestrate profiling, analysis, and visualization."""

import argparse
import os

import pandas as pd

import config
import profiler
import analyzer
import visualize_timeline_roofline as viz_roofline


def parse_args():
    p = argparse.ArgumentParser(description="Profile Prefill vs Decode in vLLM")
    p.add_argument("--model", default=config.MODEL_NAME,
                   help="HuggingFace model name or local path")
    p.add_argument("--gpu", default=config.ACTIVE_GPU,
                   choices=config.GPU_SPECS.keys(),
                   help="GPU profile for Roofline model")
    p.add_argument("--prompt-lengths", type=int, nargs="+",
                   default=config.PREFILL_PROMPT_LENGTHS)
    p.add_argument("--decode-lengths", type=int, nargs="+",
                   default=config.DECODE_LENGTHS)
    p.add_argument("--output-dir", default=config.OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides
    config.MODEL_NAME = args.model
    config.ACTIVE_GPU = args.gpu
    config.PREFILL_PROMPT_LENGTHS = args.prompt_lengths
    config.DECODE_LENGTHS = args.decode_lengths
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    gpu = config.get_gpu_specs()
    print(f"Model : {config.MODEL_NAME}")
    print(f"GPU   : {gpu['name']}  (peak {gpu['peak_flops_fp16']/1e12:.0f} TFLOPS, "
          f"{gpu['peak_bw_bytes']/1e9:.0f} GB/s)")
    print(f"Output: {config.OUTPUT_DIR}\n")

    # ── 1. Initialize engine ─────────────────────────────────────────────────
    print("[1/4] Initializing vLLM engine ...")
    llm = profiler.create_engine()

    # ── 2. Warmup ────────────────────────────────────────────────────────────
    print(f"[2/4] Warming up ({config.WARMUP_ITERATIONS} iterations) ...")
    profiler.warmup(llm)

    # ── 3. Profile ───────────────────────────────────────────────────────────
    print("[3/4] Profiling prefill & decode phases ...")
    results = profiler.run_all_profiles(llm)

    # ── 4. Analyze & Visualize ───────────────────────────────────────────────
    print("[4/4] Computing metrics & generating figures ...")
    summaries = analyzer.compute_all_summaries(results)

    # Save CSV
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(f"{config.OUTPUT_DIR}/phase_summaries.csv", index=False)

    # Generate Roofline figure
    viz_roofline.create_figure(summaries)

    # Print summary
    print("\n" + "=" * 70)
    cols = ["phase", "token_count", "elapsed_ms", "achieved_tflops",
            "compute_utilization", "memory_bw_achieved_gbs",
            "memory_bw_utilization", "tokens_per_second"]
    print(summary_df[cols].to_string(index=False))
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
