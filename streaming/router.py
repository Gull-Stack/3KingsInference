"""Top-K expert routing for MoE layers.

Given hidden states and a routing weight matrix, selects
the top-K experts per token and computes gating weights.
"""

import mlx.core as mx


class TopKRouter:
    """Routes tokens to top-K experts with softmax gating.

    The Qwen3.5 MoE uses K=4 active experts out of 512 per layer,
    plus one shared expert that always runs.

    Args:
        n_experts: Total number of experts (e.g. 512)
        k_active: Number of experts to activate (e.g. 4)
        has_shared_expert: Whether the model has a shared expert
    """

    def __init__(
        self,
        n_experts: int = 512,
        k_active: int = 4,
        has_shared_expert: bool = True,
    ):
        self.n_experts = n_experts
        self.k_active = k_active
        self.has_shared_expert = has_shared_expert

    def route(
        self,
        hidden_states: mx.array,
        router_weights: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute expert routing for input tokens.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) or (hidden_dim,)
            router_weights: (hidden_dim, n_experts) routing projection

        Returns:
            expert_indices: (batch, seq_len, k_active) int32 — which experts
            gate_weights: (batch, seq_len, k_active) float32 — softmax weights
        """
        # Compute routing logits
        logits = hidden_states @ router_weights  # (..., n_experts)

        # Top-K selection
        # MLX doesn't have a direct top-k, so we use argpartition + gather
        top_k_indices = mx.argpartition(-logits, kth=self.k_active - 1, axis=-1)
        top_k_indices = top_k_indices[..., :self.k_active]

        # Gather the logits for selected experts
        # Use advanced indexing
        if logits.ndim == 1:
            top_k_logits = logits[top_k_indices]
        else:
            # Batch gather
            batch_shape = logits.shape[:-1]
            top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)

        # Softmax over selected experts only (not all 512)
        gate_weights = mx.softmax(top_k_logits, axis=-1)

        return top_k_indices.astype(mx.int32), gate_weights

    def route_and_combine(
        self,
        hidden_states: mx.array,
        router_weights: mx.array,
        expert_outputs: list[mx.array],
        shared_expert_output: mx.array = None,
    ) -> mx.array:
        """Route, weight, and combine expert outputs.

        Args:
            hidden_states: Input for routing
            router_weights: Routing projection matrix
            expert_outputs: List of K expert outputs
            shared_expert_output: Output from shared expert (if any)

        Returns:
            combined: Weighted sum of expert outputs + shared
        """
        _, gate_weights = self.route(hidden_states, router_weights)

        # Weighted combination of K expert outputs
        combined = mx.zeros_like(expert_outputs[0])
        for i, expert_out in enumerate(expert_outputs):
            weight = gate_weights[..., i:i+1]
            combined = combined + weight * expert_out

        # Add shared expert (unweighted / separately gated)
        if shared_expert_output is not None and self.has_shared_expert:
            combined = combined + shared_expert_output

        return combined
