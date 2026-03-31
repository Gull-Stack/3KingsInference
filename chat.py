#!/usr/bin/env python3
"""Interactive chat CLI for 3KingsInference.

Usage:
    # Single machine
    python chat.py --model mlx-community/Qwen3.5-397B-A17B-4bit

    # Machine 0 (primary)
    python chat.py --model ./model --machine-id 0 --peer 192.168.1.2

    # Machine 1 (secondary)
    python chat.py --model ./model --machine-id 1 --peer 192.168.1.1
"""

import argparse
import sys

from core.config import InferenceConfig, ModelConfig
from core.pipeline import ThreeKingsPipeline


def main():
    parser = argparse.ArgumentParser(description="3Kings Interactive Chat")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF repo")
    parser.add_argument("--expert-dir", type=str, default="./packed_experts", help="Expert files directory")
    parser.add_argument("--machine-id", type=int, default=0, help="This machine's ID (0 or 1)")
    parser.add_argument("--peer", type=str, default=None, help="Peer machine address")
    parser.add_argument("--port", type=int, default=5555, help="Activation passing port")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--key-bits", type=int, default=4, help="KV cache key bits")
    parser.add_argument("--value-bits", type=int, default=5, help="KV cache value bits")
    parser.add_argument("--no-compression", action="store_true", help="Disable KV compression")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    args = parser.parse_args()

    # Build config
    n_machines = 2 if args.peer else 1
    peers = [args.peer] if args.peer else []

    config = InferenceConfig(
        model=ModelConfig.qwen_397b(),
        n_machines=n_machines,
        machine_id=args.machine_id,
        peer_addresses=peers,
        shard_port=args.port,
        expert_dir=args.expert_dir,
        enable_kv_compression=not args.no_compression,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Setup pipeline
    pipeline = ThreeKingsPipeline(config)
    pipeline.setup(args.model)

    # Chat loop
    print("3Kings Chat — type 'quit' to exit, 'stats' for performance info\n")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "stats":
                pipeline.print_stats()
                continue
            if user_input.lower() == "clear":
                messages = messages[:1] if args.system else []
                if pipeline.cache_manager:
                    pipeline.cache_manager.clear()
                print("[Context cleared]\n")
                continue

            messages.append({"role": "user", "content": user_input})

            # Format with chat template
            prompt_ids = pipeline.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            prompt_text = pipeline.tokenizer.decode(prompt_ids, skip_special_tokens=False)

            # Generate
            print("Assistant: ", end="", flush=True)
            full_response = ""
            for token in pipeline.generate(
                prompt_text,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
            ):
                print(token, end="", flush=True)
                full_response += token

            print("\n")
            messages.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        pipeline.print_stats()
        pipeline.teardown()


if __name__ == "__main__":
    main()
