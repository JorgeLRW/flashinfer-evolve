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
│              │  OpenAI-compatible /v1/ API    │                  │
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

| File                                           | Purpose                                        | Git tracked?              |
| ---------------------------------------------- | ---------------------------------------------- | ------------------------- |
| `flashinfer_bench/solution/triton/kernel.py` | Seed kernel (evolves)                          | Yes                       |
| `workspace/evaluator.py`                     | OpenEvolve evaluator with profiler-in-the-loop | **No** (gitignored) |
| `config.yaml`                                | OpenEvolve config (has API keys)               | **No** (gitignored) |
| `scripts/bench.py`                           | Benchmark & ranking tool                       | Yes                       |
| `flashinfer_bench/config.toml`               | FlashInfer submission metadata                 | Yes                       |
| `profiler_loop/ranker.py`                    | Legacy stub (use bench.py instead)             | Yes                       |

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
