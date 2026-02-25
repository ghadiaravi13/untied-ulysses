# FPDT (Fully Pipelined Distributed Transformers) Experiments

Scripts and configurations for running FPDT experiments using Megatron-DeepSpeed with long context sequence parallelism.

## Installation

```bash
# From repo root
uv venv fpdt_env && source fpdt_env/bin/activate

# Install PyTorch (must match your system CUDA from `nvcc --version`: cu124, cu121, cu118, etc.)
uv pip install torch==2.8.0 torchaudio==2.8.0 torchvision --index-url https://download.pytorch.org/whl/test/cu129

git clone https://github.com/NVIDIA/apex.git
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install -v --no-build-isolation ./
cd ..
uv pip install transformers six pybind11 psutil
uv pip install flash-attn==2.8.2 --no-build-isolation
uv pip install deepspeed==0.17.4
```

**Optional: FlashAttention 3 (Hopper GPUs)**

Required to reproduce exact tokens/second numbers from the paper.

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper
uv pip install -U setuptools wheel packaging ninja
MAX_JOBS=128 uv pip install . --no-build-isolation
cd ../..
```

## Usage

### Scripts

Scripts are in `FPDT/Megatron-DeepSpeed/examples_deepspeed/sequence_parallel/`.

| Model | Script | Description |
|-------|--------|-------------|
| LLaMA 8B | `run_single_8B.sh` | Single experiment |
| LLaMA 8B | `run_all_8B.sh` | Full benchmark (128K-5M) |
| Qwen3 32B | `run_single_32B.sh` | Single experiment |
| Qwen3 32B | `run_all_32B.sh` | Full benchmark |
| Qwen3 32B | `run_single_multinode_32B.sh` | Single experiment (multinode) |
| Qwen3 32B | `run_all_multinode_32B.sh` | Full benchmark (multinode) |

### Quick Start

```bash
cd FPDT/Megatron-DeepSpeed/examples_deepspeed/sequence_parallel

# Run single experiment (optionally pass sequence length)
./run_single_8B.sh
./run_single_8B.sh 262144  # 256K context

# Run all experiments
./run_all_8B.sh
```

### Configuration

These flags are pre-configured in the scripts ([example](FPDT/Megatron-DeepSpeed/examples_deepspeed/sequence_parallel/run_single_8B.sh#L173-L176)):

| Flag | Description |
|------|-------------|
| `--ds-sequence-parallel-fpdt` | Enables FPDT mode |
| `--ds-sequence-parallel-size N` | Sequence parallel degree (default: 8) |
| `--ds-sequence-parallel-fpdt-chunk-size N` | Chunk size (default: 65536) |
| `--ds-sequence-parallel-fpdt-offloading` | Activation offloading to CPU |
| `--use-flash-attn-v3` | Flash Attention v3 (default) |
| `--use-flash-attn-v2` | Flash Attention v2 |

## Profiling with PyTorch Profiler

To generate Chrome trace files for performance analysis, use the following flags:

| Flag | Description |
|------|-------------|
| `--torch-profiler-enable` | Enable PyTorch profiler for generating chrome traces |
| `--torch-profiler-trace-dir PATH` | Directory to save trace files (default: `./profiler_traces`) |

### Example Usage

Add these flags to your training script's `megatron_options`:

```bash
megatron_options="${megatron_options} \
    --torch-profiler-enable \
    --torch-profiler-trace-dir ${output_home}/traces"
```

### Output Structure

Traces are saved as:
```
<trace-dir>/
  iteration_1/
    rank0_trace.json
    rank1_trace.json
    ...
  iteration_2/
    ...
```

### Viewing Traces

Open the `.json` trace files in:
- Chrome: Navigate to `chrome://tracing` and load the file
- Perfetto: https://ui.perfetto.dev/

## Output Structure

Logs and checkpoints are saved to:
- **LLaMA 8B model**: `output_8B_FA3/` (or `output_8B/` for FA2)
- **Qwen3 32B model (multi-node)**: `output_32B_FA3_multinode/`

Each run creates:
- `log/` - Training logs with timestamped filenames
- `tensorboard/` - TensorBoard event files
- `checkpoint/` - Model checkpoints (if saving is enabled)