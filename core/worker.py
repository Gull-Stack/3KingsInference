"""Worker loop for Machine B (non-primary) in sharded inference.

Receives activations from Machine A, runs local layers (30-59),
sends results back. Repeats for each forward pass (prompt + each
decode step).
"""

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def run_worker_loop(pipeline):
    """Run the worker processing loop.

    Each iteration handles one forward pass:
    1. Receive hidden_states from Machine A
    2. Run through local layers (30-59)
    3. Apply final norm
    4. Send result back to Machine A
    """
    model = pipeline.model
    channel = pipeline.channel
    inner = model.language_model.model  # Qwen3_5TextModel

    # Build caches for local layers only
    cache = model.language_model.make_cache()

    print(f"Worker ready: layers {inner.start_idx}-{inner.end_idx - 1}, "
          f"{len(cache)} cache slots")

    step = 0
    while True:
        try:
            # Receive activations from Machine A
            hidden_states = channel.recv_activation()
            step += 1

            # Compute masks
            first_cache = cache[0]
            fa_mask = create_attention_mask(hidden_states, first_cache)
            ssm_mask = create_ssm_mask(hidden_states, first_cache)

            # Run through local layers
            for i in range(inner.num_layers):
                layer = inner.layers[inner.start_idx + i]
                mask = ssm_mask if layer.is_linear else fa_mask
                hidden_states = layer(hidden_states, mask=mask, cache=cache[i])

            # Apply final norm and send back
            hidden_states = inner.norm(hidden_states)
            mx.eval(hidden_states)
            channel.send_activation(hidden_states)

            if step % 50 == 0:
                print(f"  Worker: {step} forward passes completed")

        except ConnectionError:
            print("Worker: peer disconnected")
            break
