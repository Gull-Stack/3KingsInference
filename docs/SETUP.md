# Setup Guide

## Hardware Requirements

- 2x Apple Silicon Macs with 64GB+ unified memory (M4 Pro recommended)
- 1TB+ NVMe SSD on each machine
- Network: Thunderbolt 5 bridge (best) or 10GbE ethernet

## Software Requirements

- macOS 26.2+ (for RDMA support)
- Python 3.10+
- Xcode + Metal Toolchain

## Installation

```bash
git clone https://github.com/Gull-Stack/3KingsInference.git
cd 3KingsInference
pip install -e ".[dev]"
```

## Prepare Model Files

### 1. Download the model

```bash
# Using huggingface-cli
huggingface-cli download mlx-community/Qwen3.5-397B-A17B-4bit --local-dir ./model
```

### 2. Split expert weights for SSD streaming

```bash
# TODO: expert splitting script
# This will create packed_experts/layer_XX.bin files
# Each file contains all 512 experts for one layer
python scripts/split_experts.py --model-dir ./model --output-dir ./packed_experts
```

### 3. Copy expert files to each machine

Machine A gets layers 0-29:
```bash
scp packed_experts/layer_{00..29}.bin machine-a:~/packed_experts/
```

Machine B gets layers 30-59:
```bash
scp packed_experts/layer_{30..59}.bin machine-b:~/packed_experts/
```

## Running

### Single machine (for testing with smaller models)

```bash
python chat.py --model ./model --expert-dir ./packed_experts
```

### Two machines

On Machine A (primary):
```bash
python chat.py --model ./model --expert-dir ./packed_experts \
    --machine-id 0 --peer <machine-b-ip>
```

On Machine B (secondary):
```bash
python chat.py --model ./model --expert-dir ./packed_experts \
    --machine-id 1 --peer <machine-a-ip>
```

### API Server

```bash
python server.py --model ./model --port 8080
```

### Benchmark

```bash
python benchmark.py --model ./model --tokens 100
```
