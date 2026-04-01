#!/usr/bin/env python3
"""Split expert weights from Qwen3.5 safetensors into per-layer binary files.

Creates packed_experts/layer_XX.bin files for mmap streaming.
Each file contains all experts for one layer, contiguous in memory:
  [expert_0 | expert_1 | ... | expert_511]

Each expert's data is: [gate_proj | up_proj | down_proj] packed contiguously.

Usage:
    python scripts/split_experts.py \\
        --model-dir ./model \\
        --output-dir ./packed_experts \\
        --layers 0-59

    # Split only layers for machine A:
    python scripts/split_experts.py \\
        --model-dir ./model \\
        --output-dir ./packed_experts \\
        --layers 0-29
"""

import argparse
import glob
import json
import os
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def get_expert_keys(layer_idx: int) -> dict[str, list[str]]:
    """Get the safetensors key patterns for expert weights in a layer.

    Qwen3.5 MLX quantized format stores each projection as three tensors:
      .weight (uint32 packed), .scales (bfloat16), .biases (bfloat16)
    """
    prefix = f"language_model.model.layers.{layer_idx}.mlp.switch_mlp"
    keys = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        keys[proj] = [
            f"{prefix}.{proj}.weight",
            f"{prefix}.{proj}.scales",
            f"{prefix}.{proj}.biases",
        ]
    return keys


def find_weight_file_for_key(weight_files: list[str], key: str) -> str | None:
    """Find which safetensors file contains a given key.

    Uses the safetensors index if available, otherwise scans files.
    """
    # Check for index file first
    for wf in weight_files:
        index_path = wf.replace(".safetensors", ".safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            if key in index.get("weight_map", {}):
                mapped_file = index["weight_map"][key]
                return str(Path(wf).parent / mapped_file)

    # Fallback: try each file
    for wf in weight_files:
        try:
            weights = mx.load(wf)
            if key in weights:
                return wf
        except Exception:
            continue
    return None


def split_experts(
    model_dir: str,
    output_dir: str,
    layer_start: int = 0,
    layer_end: int = 60,
    num_experts: int = 512,
):
    """Extract expert weights and write per-layer binary files.

    Expert weights in safetensors are stored as:
      switch_mlp.gate_proj.weight: shape (num_experts, moe_intermediate, hidden_dim)
      switch_mlp.up_proj.weight:   shape (num_experts, moe_intermediate, hidden_dim)
      switch_mlp.down_proj.weight: shape (num_experts, hidden_dim, moe_intermediate)

    We repack each expert as a flat binary blob:
      [gate_proj_flat | up_proj_flat | down_proj_flat]
    """
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find safetensors files
    weight_files = sorted(glob.glob(str(model_path / "*.safetensors")))
    if not weight_files:
        print(f"Error: No safetensors files found in {model_dir}")
        sys.exit(1)

    # Check for index file
    index_file = model_path / "model.safetensors.index.json"
    weight_map = {}
    if index_file.exists():
        with open(index_file) as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        print(f"Found weight index with {len(weight_map)} entries")

    print(f"Splitting experts for layers {layer_start}-{layer_end - 1}")
    print(f"Output: {output_dir}/layer_XX.bin")
    print(f"Experts per layer: {num_experts}")
    print()

    total_bytes = 0
    t_start = time.time()

    # Cache of loaded safetensor files to avoid reloading
    loaded_files: dict[str, dict] = {}

    def load_safetensors(filepath: str) -> dict:
        if filepath not in loaded_files:
            # Only keep 2 files cached to limit memory
            if len(loaded_files) >= 2:
                oldest = next(iter(loaded_files))
                del loaded_files[oldest]
            loaded_files[filepath] = mx.load(filepath)
        return loaded_files[filepath]

    for layer_idx in range(layer_start, layer_end):
        keys = get_expert_keys(layer_idx)
        layer_file = output_path / f"layer_{layer_idx:02d}.bin"

        print(f"  Layer {layer_idx:2d}: ", end="", flush=True)

        # Load all tensors for this layer's experts (weight + scales + biases per proj)
        expert_tensors: dict[str, dict[str, mx.array]] = {}
        skip = False

        for proj_name, key_list in keys.items():
            expert_tensors[proj_name] = {}
            for key in key_list:
                # Determine tensor type from key suffix
                tensor_type = key.rsplit(".", 1)[-1]  # "weight", "scales", or "biases"

                # Find the right source file
                source_file = None
                if weight_map and key in weight_map:
                    source_file = str(model_path / weight_map[key])
                else:
                    source_file = find_weight_file_for_key(weight_files, key)

                if source_file is None:
                    print(f"SKIP (key {key} not found)")
                    skip = True
                    break

                weights = load_safetensors(source_file)
                if key not in weights:
                    print(f"SKIP (key {key} missing from {source_file})")
                    skip = True
                    break

                expert_tensors[proj_name][tensor_type] = weights[key]

            if skip:
                break

        if skip or len(expert_tensors) != 3:
            continue

        # Get expert count from first tensor
        actual_experts = expert_tensors["gate_proj"]["weight"].shape[0]
        if actual_experts != num_experts:
            print(f"WARN: expected {num_experts} experts, got {actual_experts}")

        # Write per-layer binary file
        # Each expert: [gate_weight|gate_scales|gate_biases|up_weight|up_scales|up_biases|down_weight|down_scales|down_biases]
        with open(layer_file, "wb") as f:
            for eidx in range(actual_experts):
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    for tensor_type in ("weight", "scales", "biases"):
                        tensor = expert_tensors[proj_name][tensor_type]
                        expert_slice = tensor[eidx].flatten()
                        # Convert to numpy-compatible dtype before writing
                        if expert_slice.dtype == mx.bfloat16:
                            expert_slice = expert_slice.astype(mx.float16)
                        raw = np.array(expert_slice, copy=False).tobytes()
                        f.write(raw)

        file_size = os.path.getsize(layer_file)
        expert_size = file_size // actual_experts
        total_bytes += file_size

        print(f"{file_size / 1024**2:.0f} MB "
              f"({actual_experts} experts × {expert_size / 1024:.0f} KB each)")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total: {total_bytes / 1024**3:.1f} GB across {layer_end - layer_start} layers")

    # Write metadata
    meta_file = output_path / "metadata.json"
    meta = {
        "model": "Qwen3.5-397B-A17B",
        "layer_start": layer_start,
        "layer_end": layer_end,
        "num_experts": num_experts,
        "total_bytes": total_bytes,
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to {meta_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Split Qwen3.5 expert weights into per-layer files for SSD streaming"
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing model safetensors")
    parser.add_argument("--output-dir", type=str, default="./packed_experts",
                        help="Output directory for layer files")
    parser.add_argument("--layers", type=str, default="0-59",
                        help="Layer range to extract (e.g. '0-29' for machine A)")
    parser.add_argument("--num-experts", type=int, default=512,
                        help="Number of experts per layer")
    args = parser.parse_args()

    # Parse layer range
    parts = args.layers.split("-")
    layer_start = int(parts[0])
    layer_end = int(parts[1]) + 1 if len(parts) > 1 else layer_start + 1

    split_experts(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        layer_start=layer_start,
        layer_end=layer_end,
        num_experts=args.num_experts,
    )


if __name__ == "__main__":
    main()
