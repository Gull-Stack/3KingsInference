# 3KingsInference 👑👑👑

**Three kings. One throne. Run the largest open MoE models on two Mac Minis.**

The first integrated inference stack that combines distributed sharding, SSD expert streaming, and KV cache compression to run 400B-class models on consumer Apple Silicon hardware.

## The Three Kings

| King | Role | What It Does |
|------|------|-------------|
| **Odin** (Sharding) | Distributes layers across machines | Splits 60 transformer layers across 2× M4 Pro via network activation passing |
| **Mjolnir** (Streaming) | Streams expert weights from SSD | Only 4 of 512 experts active per token — stream on demand, trust the OS page cache |
| **Vision** (Compression) | Compresses KV cache 5x | TurboQuant MSE-optimal quantization on attention layers — 5x smaller cache, same quality |

## Why All Three

No single approach can run a 397B model well on 128GB:

| Approach | Problem |
|----------|---------|
| Sharding alone | 397B at 4-bit = 209GB. Won't fit in 128GB combined RAM. |
| SSD streaming alone | Works on one machine but 4.4 tok/s. No multi-machine. |
| KV compression alone | Doesn't help with model weight memory at all. |
| **All three together** | Model weights stream from SSD (~6GB working set). Layers split across machines. KV cache compressed 5x. **~54GB free page cache per machine → near-100% expert hit rate.** |

## Target Performance

**Qwen3.5-397B-A17B on 2× M4 Pro 64GB Mac Minis:**

| Metric | Single Machine (Flash-MoE) | 3KingsInference |
|--------|---------------------------|-----------------|
| Decode tok/s | 4.4 | 8-12 (est.) |
| Max context | ~8K | 32K+ |
| RAM used | 48GB (tight) | ~10GB per machine |
| Page cache | ~35GB, 71% hit | ~54GB, ~95%+ hit |
| Quality | Excellent | Excellent + long context |

## Architecture

```
Machine A (64GB)                    Machine B (64GB)
┌────────────────────┐              ┌────────────────────┐
│ Layers 0-29        │              │ Layers 30-59       │
│                    │              │                    │
│ ┌────────────────┐ │   network    │ ┌────────────────┐ │
│ │ Odin Shard     │◄├─────────────►┤ │ Odin Shard     │ │
│ │ (30 layers)    │ │  activations │ │ (30 layers)    │ │
│ └───────┬────────┘ │              │ └───────┬────────┘ │
│         │          │              │         │          │
│ ┌───────▼────────┐ │              │ ┌───────▼────────┐ │
│ │ Mjolnir Stream │ │              │ │ Mjolnir Stream │ │
│ │ (local SSD)    │ │              │ │ (local SSD)    │ │
│ │ 512 experts/   │ │              │ │ 512 experts/   │ │
│ │ layer, K=4     │ │              │ │ layer, K=4     │ │
│ └───────┬────────┘ │              │ └───────┬────────┘ │
│         │          │              │         │          │
│ ┌───────▼────────┐ │              │ ┌───────▼────────┐ │
│ │ Vision Cache   │ │              │ │ Vision Cache   │ │
│ │ (TurboQuant)   │ │              │ │ (TurboQuant)   │ │
│ │ 5x compressed  │ │              │ │ 5x compressed  │ │
│ └────────────────┘ │              │ └────────────────┘ │
└────────────────────┘              └────────────────────┘
     ~54GB page cache                   ~54GB page cache
```

## Project Structure

```
3KingsInference/
├── core/
│   ├── __init__.py
│   ├── config.py          # Model config (layer count, expert count, dims)
│   ├── pipeline.py        # Main inference pipeline orchestrator
│   ├── tokenizer.py       # BPE tokenizer
│   └── model.py           # Model definition (attention, MoE, delta-net)
├── sharding/              # Odin — distributed layer sharding
│   ├── __init__.py
│   ├── shard.py           # Layer assignment + activation passing
│   ├── network.py         # Inter-machine communication
│   └── coordinator.py     # Orchestrates multi-machine inference
├── streaming/             # Mjolnir — SSD expert streaming
│   ├── __init__.py
│   ├── expert_loader.py   # mmap + pread expert weight loading
│   ├── router.py          # Top-K expert routing
│   └── page_cache.py      # Page cache monitoring + stats
├── compression/           # Vision — TurboQuant KV cache compression
│   ├── __init__.py
│   ├── codebook.py        # Lloyd-Max codebooks (from TurboQuant-Thor)
│   ├── rotation.py        # WHT + random signs (from TurboQuant-Thor)
│   ├── quantizer.py       # TurboQuantMSE (from TurboQuant-Thor)
│   ├── packing.py         # Bit packing (from TurboQuant-Thor)
│   ├── kv_cache.py        # Compressed KV cache manager
│   └── sparse_v.py        # Adaptive sparse V (from TurboQuant-Thor)
├── tests/
│   ├── test_streaming.py
│   ├── test_sharding.py
│   ├── test_compression.py
│   └── test_pipeline.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── SETUP.md
├── server.py              # OpenAI-compatible API server
├── chat.py                # Interactive chat CLI
└── benchmark.py           # Performance benchmarking
```

## Build Order

1. **Mjolnir (streaming)** — mmap expert loading in MLX, test on Qwen3.5-35B-A3B single machine
2. **Vision (compression)** — port TurboQuant-Thor core into compression/, integrate with KV cache
3. **Odin (sharding)** — layer splitting + network activation passing between two machines
4. **Pipeline** — wire all three together, add API server + chat CLI
5. **Benchmark** — measure everything, optimize bottlenecks

## Requirements

- 2× Apple Silicon Macs (M4 Pro 64GB recommended)
- Python 3.10+
- MLX >= 0.22
- 1TB+ SSD on each machine (for model expert files)
- Network connection between machines (Thunderbolt bridge or 10GbE ideal)

## References

- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026) — KV cache compression math
- [Flash-MoE](https://github.com/danveloper/flash-moe) — SSD expert streaming inspiration
- [exo](https://github.com/exo-explore/exo) — distributed inference on consumer hardware
- [Apple "LLM in a Flash"](https://arxiv.org/abs/2312.11514) — original SSD streaming concept

## License

MIT

---

Built by [Gull-Stack](https://gullstack.com) — shipping what others only theorize about.
