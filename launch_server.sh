#!/bin/bash
cd ~/3KingsInference
source .venv/bin/activate
exec python3 server.py \
  --model ~/models/Qwen3.5-397B-A17B-4bit \
  --expert-dir ./packed_experts \
  --machine-id 0 \
  --peer 192.168.100.3 \
  --shard-port 5555 \
  --port 8080
