# 3KingsInference Architecture

## Layer Pattern

Qwen3.5-397B uses `full_attention_interval = 4`:
- Layers 0, 1, 2: GatedDeltaNet (linear attention, no KV cache)
- Layer 3: Full softmax attention (KV cache, compressed by Vision)
- Layers 4, 5, 6: GatedDeltaNet
- Layer 7: Full attention
- ... repeating for 60 layers

This gives 15 full-attention layers and 45 DeltaNet layers.

## Per-Token Flow (2-machine distributed)

```
Machine A                                Machine B
─────────                                ─────────
1. Embed token
2. Layers 0-29:
   - GatedDeltaNet: conv + recurrence
   - Full Attn: Q/K/V + compressed cache
   - MoE: route → stream 4 experts
          from SSD → SwiGLU → combine
3. Send activations ──────────────────►  4. Receive activations
                                         5. Layers 30-59 (same as 2)
                                         6. Final norm + lm_head → logits
7. Receive logits  ◄──────────────────   7. Send logits
8. Sample next token
```

## Memory Layout (per 64GB machine)

| Component | RAM | Source |
|-----------|-----|--------|
| Non-expert weights | ~2.75 GB | Loaded at startup |
| Expert working set | ~6 GB | Streamed from SSD via mmap |
| KV cache (TurboQuant) | ~0.6-1.2 GB | 4-bit keys, 5-bit values |
| OS + overhead | ~6 GB | macOS |
| **Page cache** | **~54 GB** | OS manages, caches hot experts |

## Key Design Decisions

1. **Expert weights on SSD, not RAM**: Flash-MoE proved OS page cache LRU beats custom caching
2. **Asymmetric KV compression**: Keys tolerate more compression (directional), values need precision (magnitude)
3. **No TurboQuant_prod**: Algorithm 2's (b-1)-bit resolution loss is amplified by softmax
4. **Proper 3-bit packing**: 10 values per uint32, not wasteful 4-bit containers
5. **Raw TCP for activations**: 8KB per token, ~1.6µs on Thunderbolt — not the bottleneck
