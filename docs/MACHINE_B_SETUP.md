# Machine B Setup — Second Mac Mini M4 Pro 64GB

Run these steps in order on the second Mac Mini.

## 1. Install Xcode + Metal Toolchain

```bash
# Install Xcode from App Store first, then:
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain

# Verify metal compiler works
xcrun --find metal
```

## 2. Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
```

## 3. Clone and build Exo with RDMA-patched MLX

```bash
cd ~
git clone https://github.com/exo-explore/exo.git
cd exo
uv sync
```

This builds MLX from source with the RDMA GPU lock fix branch. Takes ~5-10 minutes.

Verify it worked:

```bash
source .venv/bin/activate
python -c "import mlx.core as mx; print('MLX:', mx.__version__); print('Metal:', mx.device_info()['device_name'])"
```

## 4. Clone 3KingsInference

```bash
cd ~
git clone https://github.com/Gull-Stack/3KingsInference.git
cd 3KingsInference
```

## 5. Install 3Kings dependencies

```bash
cd ~/3KingsInference
# Use the exo venv which already has MLX built
source ~/exo/.venv/bin/activate
pip install -e ".[dev]"
pip install transformers
```

## 6. Download the model (224 GB)

```bash
source ~/exo/.venv/bin/activate
python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3.5-397B-A17B-4bit (224 GB)...')
path = snapshot_download(
    'mlx-community/Qwen3.5-397B-A17B-4bit',
    local_dir='/Users/$(whoami)/models/Qwen3.5-397B-A17B-4bit',
)
print(f'Done: {path}')
"
```

Or if Machine A already has it downloaded, copy via network:

```bash
# From Machine A (faster if on same network):
rsync -avP /Users/pointbreak/models/Qwen3.5-397B-A17B-4bit/ \
    machineB:/Users/<user>/models/Qwen3.5-397B-A17B-4bit/
```

## 7. Keep the machine awake

```bash
caffeinate -d -i -s &
```

## 8. Connect Thunderbolt 5 cable to Machine A

Plug a Thunderbolt 5 cable between the two Mac Minis. macOS will auto-configure a network bridge.

Check the Thunderbolt bridge IP:

```bash
# Find the Thunderbolt bridge interface
ifconfig | grep -A5 "bridge"
# Or check System Settings > Network > Thunderbolt Bridge
```

Note the IP address — Machine A will need it.

## 9. Verify connectivity

From Machine B, ping Machine A:

```bash
ping <machine-a-thunderbolt-ip>
```

From Machine A, ping Machine B:

```bash
ping <machine-b-thunderbolt-ip>
```

## 10. Test 3Kings on Machine B (single machine, verify stack works)

```bash
cd ~/3KingsInference
source ~/exo/.venv/bin/activate

python -c "
from core.config import InferenceConfig, ModelConfig
from core.pipeline import ThreeKingsPipeline

config = InferenceConfig(
    model=ModelConfig(
        name='Qwen3.5-35B-A3B', n_layers=36, n_full_attention_layers=9,
        n_delta_net_layers=27, hidden_dim=3584, n_heads=28, n_kv_heads=4,
        head_dim=128, n_experts=128, k_active=4, has_shared_expert=True,
        expert_intermediate=2560, vocab_size=248320, max_context=131072,
    ),
    n_machines=1, enable_kv_compression=True, key_bits=4, value_bits=5,
)

pipeline = ThreeKingsPipeline(config)
pipeline.setup('/Users/\$(whoami)/models/Qwen3.5-35B-A3B-4bit')

for text in pipeline.generate('Hello from Machine B!', max_tokens=30):
    print(text)
pipeline.print_stats()
"
```

If that generates text, Machine B is ready.

## 11. Run distributed mode (after both machines are set up)

On Machine A (primary, starts first):

```bash
cd ~/3KingsInference
source ~/exo/.venv/bin/activate  # or turboquant-thor venv
python chat.py \
    --model ~/models/Qwen3.5-397B-A17B-4bit \
    --machine-id 0 \
    --peer <machine-b-thunderbolt-ip> \
    --port 5555
```

On Machine B (secondary, starts after A):

```bash
cd ~/3KingsInference
source ~/exo/.venv/bin/activate
python chat.py \
    --model ~/models/Qwen3.5-397B-A17B-4bit \
    --machine-id 1 \
    --peer <machine-a-thunderbolt-ip> \
    --port 5555
```

## Checklist

- [ ] Xcode + Metal Toolchain installed
- [ ] Exo built with RDMA MLX (`uv sync` completed)
- [ ] 3KingsInference cloned
- [ ] Dependencies installed
- [ ] Model downloaded (224 GB)
- [ ] Thunderbolt 5 cable connected
- [ ] Machines can ping each other
- [ ] Single-machine test passed
- [ ] `caffeinate` running
