#!/usr/bin/env python3
"""Main entry point: orchestrate profiling, analysis, and visualization."""

import argparse
import os
import sys

import pandas as pd

import config
import profiler
import analyzer
import visualize_timeline_roofline as viz_tl
import visualize_comparison as viz_cmp


def parse_args():
    p = argparse.ArgumentParser(description="Profile Prefill vs Decode in vLLM")
    p.add_argument("--model", default=config.MODEL_NAME, help="HuggingFace model name")
    p.add_argument("--gpu", default=config.ACTIVE_GPU, choices=config.GPU_SPECS.keys(),
                   help="GPU profile for Roofline model")
    p.add_argument("--prompt-lengths", type=int, nargs="+",
                   default=config.PREFILL_PROMPT_LENGTHS,
                   help="Prompt lengths for prefill profiling")
    p.add_argument("--decode-lengths", type=int, nargs="+",
                   default=config.DECODE_LENGTHS,
                   help="Token counts for decode profiling")
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
    print("[1/5] Initializing vLLM engine (enforce_eager=True) ...")
    llm = profiler.create_engine()

    # ── 2. Warmup ────────────────────────────────────────────────────────────
    print(f"[2/5] Warming up ({config.WARMUP_ITERATIONS} iterations) ...")
    profiler.warmup(llm)

    # ── 3. Profile ───────────────────────────────────────────────────────────
    print("[3/5] Profiling prefill & decode phases ...")
    results = profiler.run_all_profiles(llm)

    # ── 4. Analyze ───────────────────────────────────────────────────────────
    print("[4/5] Analyzing traces ...")
    all_summaries = []
    prefill_dfs, decode_dfs = [], []

    for r in results["prefill"]:
        plen = r["prompt_length"]
        df = analyzer.extract_kernel_events(r["profiler"])
        df = analyzer.compute_roofline_metrics(df, seq_len=plen, phase="prefill")
        df["phase"] = "prefill"
        df["config_label"] = f"prompt_len={plen}"
        prefill_dfs.append(df)
        summary = analyzer.compute_phase_summary(df, "prefill", plen, seq_len=plen)
        all_summaries.append(summary)

    for r in results["decode"]:
        dlen = r["decode_length"]
        # For decode, seq_len is the KV cache length (short prompt + generated tokens)
        kv_len = config.DECODE_SHORT_PROMPT_LEN + dlen
        df = analyzer.extract_kernel_events(r["profiler"])
        df = analyzer.compute_roofline_metrics(df, seq_len=kv_len, phase="decode")
        df["phase"] = "decode"
        df["config_label"] = f"decode_len={dlen}"
        decode_dfs.append(df)
        summary = analyzer.compute_phase_summary(df, "decode", dlen, seq_len=kv_len)
        all_summaries.append(summary)

    # Merge for visualization (use largest prefill, longest decode as representative)
    pf_repr = prefill_dfs[-1] if prefill_dfs else pd.DataFrame()
    dc_repr = decode_dfs[-1] if decode_dfs else pd.DataFrame()

    # Save raw metrics
    all_kernels = pd.concat(prefill_dfs + decode_dfs, ignore_index=True)
    all_kernels.to_csv(f"{config.OUTPUT_DIR}/kernel_metrics.csv", index=False)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(f"{config.OUTPUT_DIR}/phase_summaries.csv", index=False)
    print(f"  Saved kernel_metrics.csv and phase_summaries.csv")

    # ── 5. Visualize ─────────────────────────────────────────────────────────
    print("[5/5] Generating figures ...")
    viz_tl.create_figure(pf_repr, dc_repr)
    viz_cmp.create_figure(all_summaries)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

