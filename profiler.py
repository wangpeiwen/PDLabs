"""Profiling logic: use torch.cuda.Event for precise timing,
theoretical model for FLOPs/bytes — inspired by test.py's simple approach."""

import gc
import torch
from vllm import LLM, SamplingParams
from transformers import AutoConfig

import config


def create_engine():
    """Initialize vLLM and populate model architecture info."""
    llm = LLM(
        model=config.MODEL_NAME,
        dtype=config.DTYPE,
        enforce_eager=config.ENFORCE_EAGER,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
    )
    _populate_model_arch()
    return llm


def _populate_model_arch():
    """Read model config and fill config.MODEL_ARCH."""
    mc = AutoConfig.from_pretrained(config.MODEL_NAME)
    print(f"  [debug] Model config type: {type(mc).__name__}")

    h = getattr(mc, "hidden_size", None) or getattr(mc, "d_model", None) or 768
    n_heads = getattr(mc, "num_attention_heads", None) or getattr(mc, "n_head", None) or 12
    n_layers = getattr(mc, "num_hidden_layers", None) or getattr(mc, "n_layer", None) or 12
    vocab = getattr(mc, "vocab_size", None) or 50272

    inter = None
    for attr in ("intermediate_size", "ffn_dim", "n_inner"):
        val = getattr(mc, attr, None)
        if val is not None and val > 0:
            inter = val
            break
    if not inter:
        inter = h * 4

    config.MODEL_ARCH.update({
        "n_layers": n_layers,
        "hidden_size": h,
        "n_heads": n_heads,
        "head_dim": h // n_heads,
        "intermediate_size": inter,
        "vocab_size": vocab,
    })
    print(f"  Model arch: {n_layers}L, h={h}, heads={n_heads}, "
          f"head_dim={h // n_heads}, ffn={inter}, vocab={vocab}")


def _build_prompt(llm, target_len: int) -> str:
    """Build a prompt that tokenizes to exactly target_len tokens."""
    tokenizer = llm.get_tokenizer()
    base = "hello " * (target_len * 2)
    token_ids = tokenizer.encode(base)[:target_len]
    return tokenizer.decode(token_ids)


def warmup(llm, n: int = None):
    n = n or config.WARMUP_ITERATIONS
    prompt = _build_prompt(llm, 32)
    params = SamplingParams(temperature=0.0, max_tokens=8)
    for _ in range(n):
        llm.generate([prompt], params)
    torch.cuda.synchronize()


def _timed_generate(llm, prompts, params):
    """Run generate with CUDA event timing, return (outputs, elapsed_ms)."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    outputs = llm.generate(prompts, params)
    end.record()
    torch.cuda.synchronize()
    return outputs, start.elapsed_time(end)


def profile_prefill(llm, prompt_length: int) -> dict:
    """Profile prefill: long prompt, generate only 1 token."""
    prompt = _build_prompt(llm, prompt_length)
    params = SamplingParams(temperature=0.0, max_tokens=1)
    _, elapsed_ms = _timed_generate(llm, [prompt], params)

    _cleanup()
    return {
        "phase": "prefill",
        "prompt_length": prompt_length,
        "elapsed_ms": elapsed_ms,
    }


def profile_decode(llm, decode_length: int) -> dict:
    """Profile decode: short prompt, generate many tokens.

    We run two passes:
      1. short prompt + 1 token  → measures prefill-only time
      2. short prompt + N tokens → total time
    Decode time = total - prefill_only
    """
    short_prompt = _build_prompt(llm, config.DECODE_SHORT_PROMPT_LEN)

    # Measure prefill-only for the short prompt
    params_pf = SamplingParams(temperature=0.0, max_tokens=1)
    _, prefill_ms = _timed_generate(llm, [short_prompt], params_pf)

    # Measure total (prefill + decode)
    params_dc = SamplingParams(temperature=0.0, max_tokens=decode_length)
    _, total_ms = _timed_generate(llm, [short_prompt], params_dc)

    decode_ms = max(total_ms - prefill_ms, 0.01)

    _cleanup()
    return {
        "phase": "decode",
        "decode_length": decode_length,
        "elapsed_ms": decode_ms,
        "total_ms": total_ms,
        "prefill_overhead_ms": prefill_ms,
    }


def _cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def run_all_profiles(llm) -> dict:
    """Run the full profiling sweep."""
    results = {"prefill": [], "decode": []}

    for plen in config.PREFILL_PROMPT_LENGTHS:
        print(f"  Profiling prefill  prompt_len={plen} ...")
        results["prefill"].append(profile_prefill(llm, plen))

    for dlen in config.DECODE_LENGTHS:
        print(f"  Profiling decode   decode_len={dlen} ...")
        results["decode"].append(profile_decode(llm, dlen))

    return results
