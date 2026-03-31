#!/usr/bin/env python3
"""Performance benchmark for 3KingsInference.

Measures:
- Prefill speed (prompt processing)
- Decode speed (token generation)
- Expert load statistics (SSD vs page cache)
- KV cache compression stats
- Memory usage
"""

import argparse
import time

import mlx.core as mx

from core.config import InferenceConfig, ModelConfig
from core.pipeline import ThreeKingsPipeline
from streaming.page_cache import print_cache_report


def benchmark_decode(pipeline: ThreeKingsPipeline, prompt: str, n_tokens: int) -> dict:
    """Benchmark decode speed."""
    print(f"\nGenerating {n_tokens} tokens...")

    tokens = []
    t_start = time.time()

    for token in pipeline.generate(prompt, max_tokens=n_tokens, temperature=0.7, stream=True):
        tokens.append(token)

    t_total = time.time() - t_start
    n_generated = len(tokens)

    return {
        "tokens_generated": n_generated,
        "total_time_s": t_total,
        "tokens_per_second": n_generated / t_total if t_total > 0 else 0,
        "prompt_tokens": pipeline._stats.prompt_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="3Kings Benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--expert-dir", type=str, default="./packed_experts")
    parser.add_argument("--machine-id", type=int, default=0)
    parser.add_argument("--peer", type=str, default=None)
    parser.add_argument("--tokens", type=int, default=100, help="Tokens to generate per test")
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--value-bits", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10, help="Warmup tokens")
    args = parser.parse_args()

    n_machines = 2 if args.peer else 1
    peers = [args.peer] if args.peer else []

    config = InferenceConfig(
        model=ModelConfig.qwen_397b(),
        n_machines=n_machines,
        machine_id=args.machine_id,
        peer_addresses=peers,
        expert_dir=args.expert_dir,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
    )

    pipeline = ThreeKingsPipeline(config)
    pipeline.setup(args.model)

    print("=" * 60)
    print("3KingsInference Benchmark")
    print("=" * 60)

    # Memory info
    mem = config.memory_estimate()
    print(f"\nMemory Estimate:")
    print(f"  Non-expert weights: {mem['non_expert_gb']:.1f} GB")
    print(f"  Expert working set: {mem['expert_working_gb']:.1f} GB")
    print(f"  OS overhead:        {mem['os_overhead_gb']:.1f} GB")
    print(f"  KV at 8K tokens:    {mem['kv_at_8k_gb']:.2f} GB")
    print(f"  KV at 32K tokens:   {mem['kv_at_32k_gb']:.2f} GB")
    print(f"  Available page cache: {mem['available_page_cache_gb']:.0f} GB")

    # Page cache report
    print()
    print_cache_report()

    # Warmup
    if args.warmup > 0:
        print(f"\nWarmup: generating {args.warmup} tokens...")
        for _ in pipeline.generate("Hello", max_tokens=args.warmup, stream=True):
            pass

    # Benchmark
    prompts = [
        "Explain the theory of general relativity in simple terms.",
        "Write a Python function that implements a binary search tree.",
        "What are the key differences between TCP and UDP protocols?",
    ]

    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n--- Test {i+1}: {prompt[:50]}... ---")
        result = benchmark_decode(pipeline, prompt, args.tokens)
        results.append(result)
        print(f"  {result['tokens_generated']} tokens in {result['total_time_s']:.2f}s "
              f"= {result['tokens_per_second']:.1f} tok/s")

    # Summary
    avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
    print(f"\n{'=' * 60}")
    print(f"Average decode speed: {avg_tps:.1f} tok/s")
    print(f"{'=' * 60}")

    # Final stats
    pipeline.print_stats()

    if pipeline.cache_manager:
        pipeline.cache_manager.print_stats()

    pipeline.teardown()


if __name__ == "__main__":
    main()
