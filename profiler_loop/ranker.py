import os
import subprocess
import pandas as pd
from pathlib import Path

"""
Ranks the kernels on latency and TFLOPS -- basic implementation (for humans)
"""

# 1. Config: Point to the TraceSet you downloaded
TRACE_DATA_PATH = os.getenv("FIB_DATASET_PATH")
BASELINE_DIR = Path("flashinfer_bench/solutions/baseline")

def run_benchmark(solution_path):
    """Runs the FlashInfer benchmark for a specific kernel JSON."""
    # We use the starter kit's local runner
    cmd = [
        "python", "flashinfer_bench/scripts/run_local.py",
        "--solution", str(solution_path),
        "--dataset", TRACE_DATA_PATH
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_results(result.stdout)

def parse_results(output):
    """Logic to extract Latency/TFLOPs from the console output."""
    # TODO: Regex to grab the 'Mean Latency' from FlashInfer output (nonexistent yet for custom kernels)
    return {"latency": 0.0, "tflops": 0.0}

if __name__ == "__main__":
    print(f"--- Ranking Kernels in {BASELINE_DIR} ---")
    # Loop through the GPT-5 / Claude baseline solutions
    for solution in BASELINE_DIR.glob("**/*.json"):
        stats = run_benchmark(solution)
        print(f"Kernel: {solution.name} | Latency: {stats['latency']}ms")