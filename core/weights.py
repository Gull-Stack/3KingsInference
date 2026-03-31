"""Weight loader for 3KingsInference.

Loads non-expert weights from MLX safetensors into the model.
Expert FFN weights (switch_mlp.{gate,up,down}_proj) are SKIPPED —
they stream from SSD via Mjolnir.

Supports:
- mlx-community quantized models (4-bit, 6-bit, 8-bit)
- Standard safetensors format
- Automatic weight key mapping
"""

import json
import glob
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from core.model import Qwen3_5ForCausalLM, ModelArgs


# Expert weight patterns to SKIP (these stream from SSD)
EXPERT_PATTERNS = [
    ".mlp.switch_mlp.gate_proj.",
    ".mlp.switch_mlp.up_proj.",
    ".mlp.switch_mlp.down_proj.",
]

# Patterns to always skip
SKIP_PATTERNS = [
    "mtp.",          # Multi-token prediction heads (not used in inference)
    "visual",        # Vision tower (text-only inference)
    "vision_tower",
]


def _is_expert_weight(key: str) -> bool:
    """Check if a weight key belongs to expert FFN (should stream from SSD)."""
    return any(pat in key for pat in EXPERT_PATTERNS)


def _is_skip_weight(key: str) -> bool:
    """Check if a weight should be skipped entirely."""
    return any(pat in key for pat in SKIP_PATTERNS)


def _map_weight_key(key: str) -> str:
    """Map safetensors weight key to our model's parameter name.

    MLX safetensors keys follow the same naming as our model
    (both derived from HuggingFace convention), so most keys
    map directly. Handle special cases here.
    """
    # Our model uses the same naming convention as mlx-lm
    # The main difference is our StreamingMoEBlock vs SwitchGLU

    # Map shared expert gate
    if ".mlp.shared_expert_gate.weight" in key:
        return key.replace(".mlp.shared_expert_gate.", ".mlp.shared_expert.shared_expert_gate.")

    # Map router gate
    # mlx-lm: model.layers.X.mlp.gate.weight -> our: model.layers.X.mlp.gate.weight
    # This maps directly since our StreamingMoEBlock also has self.gate

    return key


def _sanitize_conv1d(key: str, value: mx.array) -> mx.array:
    """Fix Conv1d weight shape if needed.

    mlx-lm convention: Conv1d weights may need axis swap.
    """
    if "conv1d.weight" in key and value.ndim == 3 and value.shape[-1] != 1:
        return mx.moveaxis(value, 2, 1)
    return value


def load_model_config(model_path: str) -> dict:
    """Load model config.json from model directory."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path) as f:
        return json.load(f)


def config_to_model_args(config: dict) -> ModelArgs:
    """Convert HuggingFace config.json to ModelArgs."""
    text_config = config.get("text_config", config)

    args = ModelArgs(
        hidden_size=text_config.get("hidden_size", 4096),
        num_hidden_layers=text_config.get("num_hidden_layers", 60),
        num_attention_heads=text_config.get("num_attention_heads", 64),
        num_key_value_heads=text_config.get("num_key_value_heads", 8),
        head_dim=text_config.get("head_dim", 128),
        intermediate_size=text_config.get("intermediate_size", 14336),
        vocab_size=text_config.get("vocab_size", 152064),
        rms_norm_eps=text_config.get("rms_norm_eps", 1e-6),
        rope_theta=text_config.get("rope_theta", 1000000.0),
        partial_rotary_factor=text_config.get("partial_rotary_factor", 0.25),
        max_position_embeddings=text_config.get("max_position_embeddings", 131072),
        full_attention_interval=text_config.get("full_attention_interval", 4),
        num_experts=text_config.get("num_experts", 0),
        num_experts_per_tok=text_config.get("num_experts_per_tok", 0),
        moe_intermediate_size=text_config.get("moe_intermediate_size", 0),
        shared_expert_intermediate_size=text_config.get("shared_expert_intermediate_size", 0),
        norm_topk_prob=text_config.get("norm_topk_prob", True),
        linear_num_value_heads=text_config.get("linear_num_value_heads", 64),
        linear_num_key_heads=text_config.get("linear_num_key_heads", 16),
        linear_key_head_dim=text_config.get("linear_key_head_dim", 192),
        linear_value_head_dim=text_config.get("linear_value_head_dim", 128),
        linear_conv_kernel_dim=text_config.get("linear_conv_kernel_dim", 4),
    )
    return args


def load_weights(
    model: Qwen3_5ForCausalLM,
    model_path: str,
    lazy: bool = True,
) -> tuple[int, int, int]:
    """Load non-expert weights from safetensors into the model.

    Expert FFN weights are skipped — they stream from SSD via Mjolnir.

    Args:
        model: The model to load weights into
        model_path: Path to model directory with safetensors files
        lazy: If True, use mx.load with lazy evaluation

    Returns:
        Tuple of (loaded_count, skipped_expert_count, skipped_other_count)
    """
    model_dir = Path(model_path)

    # Find safetensors files
    weight_files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    loaded = 0
    skipped_expert = 0
    skipped_other = 0

    # Collect all non-expert weights
    weights_to_load = {}

    for wf in weight_files:
        file_weights = mx.load(wf)

        for key, value in file_weights.items():
            # Skip expert weights (they stream from SSD)
            if _is_expert_weight(key):
                skipped_expert += 1
                continue

            # Skip irrelevant weights
            if _is_skip_weight(key):
                skipped_other += 1
                continue

            # Map key name
            mapped_key = _map_weight_key(key)

            # Sanitize conv1d weights
            value = _sanitize_conv1d(mapped_key, value)

            weights_to_load[mapped_key] = value
            loaded += 1

    # Load into model
    model.load_weights(list(weights_to_load.items()), strict=False)

    return loaded, skipped_expert, skipped_other


def load_model_from_path(
    model_path: str,
    layer_range: Optional[tuple[int, int]] = None,
) -> tuple[Qwen3_5ForCausalLM, ModelArgs]:
    """Load model architecture + non-expert weights from a model directory.

    Args:
        model_path: Path to HuggingFace model directory
        layer_range: Optional (start, end) to only load specific layers
                     (for sharded inference)

    Returns:
        (model, args) tuple
    """
    # Load config
    config = load_model_config(model_path)
    args = config_to_model_args(config)

    print(f"Model: {args.num_hidden_layers} layers, {args.hidden_size}d, "
          f"{args.num_experts} experts")

    # Build model
    model = Qwen3_5ForCausalLM(args)

    # Load non-expert weights
    print(f"Loading non-expert weights from {model_path}...")
    loaded, skipped_expert, skipped_other = load_weights(model, model_path)
    print(f"  Loaded: {loaded} tensors")
    print(f"  Skipped (expert FFN → SSD): {skipped_expert} tensors")
    print(f"  Skipped (other): {skipped_other} tensors")

    # Evaluate loaded weights
    mx.eval(model.parameters())

    return model, args
