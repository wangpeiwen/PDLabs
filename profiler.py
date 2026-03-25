"""Profiling logic: wrap vLLM inference with torch.profiler to capture
separate traces for the Prefill and Decode phases."""

import gc
import torch
from vllm import LLM, SamplingParams

import config


def create_engine():
    """Initialize vLLM with eager mode so profiler sees individual kernels."""
    return LLM(
        model=config.MODEL_NAME,
        dtype=config.DTYPE,
        enforce_eager=config.ENFORCE_EAGER,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
    )


def _build_prompt(llm, target_len: int) -> str:
    """Build a prompt string that tokenizes to exactly `target_len` tokens."""
    tokenizer = llm.get_tokenizer()
    # Repeat a simple word, then truncate to exact length
    base = "hello " * (target_len * 2)
    token_ids = tokenizer.encode(base)[:target_len]
    return tokenizer.decode(token_ids)


def warmup(llm, n: int = None):
    """Run a few inference passes to warm up CUDA caches and JIT."""
    n = n or config.WARMUP_ITERATIONS
    prompt = _build_prompt(llm, 32)
    params = SamplingParams(temperature=0.0, max_tokens=8)
    for _ in range(n):
        llm.generate([prompt], params)
    torch.cuda.synchronize()


def _profile_inference(llm, prompt: str, max_tokens: int, trace_name: str):
    """Run a single profiled inference and return the profiler + trace path."""
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    trace_path = f"{config.OUTPUT_DIR}/{trace_name}.json"

    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        llm.generate([prompt], params)
        torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    return {
        "profiler": prof,
        "key_averages": prof.key_averages(group_by_input_shape=True),
        "chrome_trace_path": trace_path,
    }


def profile_prefill(llm, prompt_length: int) -> dict:
    """Profile the prefill phase by generating only 1 token from a long prompt."""
    prompt = _build_prompt(llm, prompt_length)
    result = _profile_inference(
        llm, prompt, max_tokens=1,
        trace_name=f"prefill_len{prompt_length}",
    )
    result["phase"] = "prefill"
    result["prompt_length"] = prompt_length
    _cleanup()
    return result


def profile_decode(llm, decode_length: int) -> dict:
    """Profile the decode phase using a short prompt + long generation."""
    prompt = _build_prompt(llm, config.DECODE_SHORT_PROMPT_LEN)
    result = _profile_inference(
        llm, prompt, max_tokens=decode_length,
        trace_name=f"decode_len{decode_length}",
    )
    result["phase"] = "decode"
    result["decode_length"] = decode_length
    _cleanup()
    return result


def _cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def run_all_profiles(llm) -> dict:
    """Run the full profiling sweep and return all results."""
    results = {"prefill": [], "decode": []}

    for plen in config.PREFILL_PROMPT_LENGTHS:
        print(f"  Profiling prefill  prompt_len={plen} ...")
        results["prefill"].append(profile_prefill(llm, plen))

    for dlen in config.DECODE_LENGTHS:
        print(f"  Profiling decode   decode_len={dlen} ...")
        results["decode"].append(profile_decode(llm, dlen))

    return results
