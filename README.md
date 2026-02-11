# FlashInfer-Evolve Pipeline

End-to-end pipeline for the [FlashInfer AI Kernel Generation Contest](https://bench.flashinfer.ai/).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LLM Backend (pick one)                                         │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │ Ollama on A5000 machine │  │ OpenAI API (Atlas credits)   │  │
│  │ qwen2.5-coder:32b       │  │ gpt-4o / o3-mini             │  │
│  │ deepseek-coder-v2:16b   │  │                              │  │
│  └───────────┬─────────────┘  └──────────────┬───────────────┘  │
│              │  OpenAI-compatible /v1/ API   │                  │
│              └───────────────┬───────────────┘                  │
└──────────────────────────────┼──────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  OpenEvolve (evolutionary loop)                                  │
│  - Reads seed kernel.py                                          │
│  - Generates mutations via LLM (diff-based)                      │
│  - Calls evaluator.py for each candidate                         │
│  - Maintains population of 200 programs across 3 islands         │
│  - 200 iterations, early stopping after 40 stale iterations      │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  workspace/evaluator.py (profiler-in-the-loop)                   │
│  1. Receive evolved kernel.py path                               │
│  2. Pack into FlashInfer Solution                                │
│  3. Run flashinfer-bench (19 workloads, correctness + latency)   │
│  4. If all pass + NCU enabled: run NCU profiling                 │
│  5. Return combined_score + per-workload metrics + NCU feedback   │
│     → LLM sees ALL metrics in its next prompt                    │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  GPU (RTX 4090 or B200 via Modal)                                │
│  - sm_89 (4090) or sm_100 (B200) — both support FP8             │
│  - flashinfer-bench handles all benchmark orchestration          │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Linux with CUDA GPU (sm_89+ for FP8)
- Python 3.10+
- `pip install flashinfer-bench`
- OpenEvolve: `cd openevolve && pip install -e .`
- FlashInfer trace dataset downloaded

### 1. Set environment variables

```bash
export FIB_DATASET_PATH=/path/to/flashinfer-trace

# For profiler-in-the-loop (optional, needs NCU installed):
export OE_ENABLE_NCU=1
```

### 2. Configure LLM backend

Edit `config.yaml`:

- **Ollama**: set `api_base` to your A5000 machine URL
- **OpenAI**: set `api_base: "https://api.openai.com/v1/"` and `api_key` to your key

### 3. Run the evolution

```bash
cd openevolve
python openevolve-run.py \
    ../flashinfer_bench/solution/triton/kernel.py \
    ../workspace/evaluator.py \
    --config ../config.yaml \
    --iterations 200
```

### 4. Benchmark & compare results

```bash
# Test the current seed kernel
python scripts/bench.py

# Compare multiple evolved kernels
python scripts/bench.py --kernel gen_42/kernel.py gen_100/kernel.py

# Quick mode (faster, less accurate)
python scripts/bench.py --kernel kernel.py --quick
```

### 5. Submit to leaderboard

```bash
# Pack your best kernel into a solution.json
cd flashinfer_bench
python scripts/pack_solution.py

# Run on Modal B200 for official scores
python scripts/run_modal.py
```

## Key Files

| File                                           | Purpose                                        |
| ---------------------------------------------- | ---------------------------------------------- |
| `flashinfer_bench/solution/triton/kernel.py` | Seed kernel (evolves)                          |
| `workspace/evaluator.py (not included)`      | OpenEvolve evaluator with profiler-in-the-loop |
| `config.yaml (not included)`                 | OpenEvolve config (has API keys)               |
| `scripts/bench.py`                           | Benchmark & ranking tool                       |
| `flashinfer_bench/config.toml`               | FlashInfer submission metadata                 |

## Profiler-in-the-Loop

When `OE_ENABLE_NCU=1`, the evaluator runs NVIDIA Nsight Compute on correct kernels
and feeds metrics back to the LLM:

- **SM Throughput** — is the kernel compute-bound?
- **DRAM Throughput** — is it memory-bound?
- **Achieved Occupancy** — are we launching enough warps?
- **Actionable advice** — e.g. "MEMORY-BOUND: Improve data reuse, tiling, or fuse dequant into GEMMs"

This gives the LLM concrete performance data to guide its next mutation.

## Track Details

**Definition**: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

- FP8 block-scale quantized weights (block_size=128)
- DeepSeek-V3 no-aux routing (sigmoid → group top-2 → top-4 groups → global top-8)
- 32 local experts, top-8 selected per token
- GEMM1: 7168 → 4096 (gate + up), SwiGLU, GEMM2: 2048 → 7168 (down)
- 19 workloads varying `seq_len`
- Scoring: geometric mean of speedup vs reference across all workloads

## Our Kernel vs the Baseline

### Baseline: `flashinfer_moe` (the reference to beat)

FlashInfer's built-in `trtllm_fp8_block_scale_moe`

```
Entry:        main.py::run (returns output tensor)
Language:     python (thin wrapper → C++/CUDA under the hood)
DPS:          false
Key detail:   tile_tokens_dim dynamically computed per seq_len:
              next_power_of_2(seq_len * top_k / num_experts), clamped [8, 64]
```

### OpenEvolve seed kernel

Ccorrect yet naive PyTorch implementation

```
Entry:        kernel.py::kernel (writes into pre-allocated output)
Language:     triton (pure Triton/PyTorch, no flashinfer dependency)
DPS:          true
Status:       Correct, slow (~0.1-0.3x vs baseline)
```

### Win Con

Key optimizations in order of impact:

| # | Optimization                                                | Justification                                                                                          |
| - | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 1 | Replace per-expert Python loop with parallel Triton kernels | Eliminates serial bottleneck (32 sequential expert calls → 1 fused launch)                            |
| 2 | Fuse FP8 dequant into GEMMs                                 | Avoids materializing full fp32 weight tensors in global memory                                         |
| 3 | Use `tl.dot` with fp8 operands                            | Leverages FP8 tensor cores                                                                             |
| 4 | Tile GEMMs for shared-memory reuse                          | Matches baseline's `tile_tokens_dim` strategy — tile token dim per expert, power-of-2 sizes [8, 64] |
| 5 | Vectorize routing / top-k selection                         | Routing is cheap but serial; Triton-native top-k avoids Python overhead                                |
| 6 | Minimize global memory traffic                              | Fuse SwiGLU between GEMM1 and GEMM2 so intermediate stays in SRAM                                      |

The profiler-in-the-loop feeds NCU metrics (SM throughput, DRAM throughput, occupancy)
back to the LLM after each correct kernel, so it can identify which of these bottlenecks
to attack next.
