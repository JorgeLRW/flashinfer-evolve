#!/usr/bin/env python3
"""
bench.py — Benchmark & Rank Kernel Solutions
=============================================

Test one or more kernel implementations against the FlashInfer workloads and
produce a side-by-side comparison table.

Usage:
  # Benchmark the current starter-kit solution (from config.toml)
  python scripts/bench.py

  # Benchmark a specific kernel file
  python scripts/bench.py --kernel path/to/kernel.py

  # Compare multiple kernels
  python scripts/bench.py --kernel v1/kernel.py v2/kernel.py v3/kernel.py

  # Benchmark a pre-packed solution.json
  python scripts/bench.py --solution solution_a.json solution_b.json

  # Save results to JSON
  python scripts/bench.py --kernel kernel.py --output results.json

  # Quick mode (fewer iterations, useful during dev)
  python scripts/bench.py --kernel kernel.py --quick

Requires:
  - CUDA-capable GPU (sm_89+ for FP8)
  - FIB_DATASET_PATH environment variable pointing to flashinfer-trace dataset
  - pip install flashinfer-bench
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIB_ROOT = PROJECT_ROOT / "flashinfer_bench"

sys.path.insert(0, str(FIB_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFINITION_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
DEFAULT_AUTHOR = "bench-test"
DEFAULT_LANGUAGE = "triton"
DEFAULT_ENTRY_POINT = "kernel"


def get_dataset_path() -> str:
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        print("ERROR: FIB_DATASET_PATH environment variable is not set.")
        print("  Set it to the path of your flashinfer-trace dataset, e.g.:")
        print("  export FIB_DATASET_PATH=/path/to/flashinfer-trace")
        sys.exit(1)
    return path


def pack_kernel_file(kernel_path: str, name: str = "bench-kernel") -> "Solution":
    """Pack a standalone kernel.py into a FlashInfer Solution object."""
    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    tmpdir = tempfile.mkdtemp(prefix="bench_")
    try:
        # Copy kernel file into temp dir as kernel.py
        dst = Path(tmpdir) / "kernel.py"
        shutil.copy2(kernel_path, dst)

        spec = BuildSpec(
            language=DEFAULT_LANGUAGE,
            target_hardware=["cuda"],
            entry_point=DEFAULT_ENTRY_POINT,
        )
        solution = pack_solution_from_files(
            path=str(tmpdir),
            spec=spec,
            name=name,
            definition=DEFINITION_NAME,
            author=DEFAULT_AUTHOR,
        )
        return solution
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def pack_from_config() -> "Solution":
    """Pack using the starter-kit's config.toml (default path)."""
    sys.path.insert(0, str(FIB_ROOT))
    from scripts.pack_solution import pack_solution

    solution_path = pack_solution()
    from flashinfer_bench import Solution
    return Solution.model_validate_json(solution_path.read_text())


def run_benchmark(
    solution: "Solution",
    dataset_path: str,
    quick: bool = False,
) -> Dict[str, Any]:
    """Run flashinfer-bench and return per-workload results."""
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

    if quick:
        config = BenchmarkConfig(warmup_runs=1, iterations=20, num_trials=2)
    else:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(dataset_path)
    def_name = solution.definition

    if def_name not in trace_set.definitions:
        return {"error": f"Definition '{def_name}' not found in dataset"}

    definition = trace_set.definitions[def_name]
    workloads = trace_set.workloads.get(def_name, [])

    if not workloads:
        return {"error": f"No workloads for '{def_name}'"}

    bench_ts = TraceSet(
        root=trace_set.root,
        definitions={def_name: definition},
        solutions={def_name: [solution]},
        workloads={def_name: workloads},
        traces={def_name: []},
    )

    benchmark = Benchmark(bench_ts, config)
    result_ts = benchmark.run_all()

    traces = result_ts.traces.get(def_name, [])
    results = []

    for trace in traces:
        if not trace.evaluation:
            results.append({
                "workload_uuid": getattr(trace.workload, "uuid", "?"),
                "seq_len": getattr(trace.workload, "axes", {}).get("seq_len", "?"),
                "status": "no_evaluation",
            })
            continue

        status = trace.evaluation.status
        status_val = status.value if hasattr(status, "value") else str(status)
        entry = {
            "workload_uuid": trace.workload.uuid if trace.workload else "?",
            "seq_len": trace.workload.axes.get("seq_len", "?") if trace.workload else "?",
            "status": status_val,
        }
        if trace.evaluation.performance:
            entry["latency_ms"] = trace.evaluation.performance.latency_ms
            entry["ref_latency_ms"] = trace.evaluation.performance.reference_latency_ms
            entry["speedup"] = trace.evaluation.performance.speedup_factor
        if trace.evaluation.correctness:
            entry["max_abs_err"] = trace.evaluation.correctness.max_absolute_error
            entry["max_rel_err"] = trace.evaluation.correctness.max_relative_error

        results.append(entry)

    return {"workloads": results}


def compute_summary(results: Dict) -> Dict[str, Any]:
    """Compute aggregate stats from benchmark results."""
    if "error" in results:
        return {"error": results["error"]}

    workloads = results.get("workloads", [])
    total = len(workloads)
    passed = sum(1 for w in workloads if w.get("status") == "passed")
    failed = total - passed

    latencies = [w["latency_ms"] for w in workloads if "latency_ms" in w]
    speedups = [w["speedup"] for w in workloads if "speedup" in w]
    ref_latencies = [w["ref_latency_ms"] for w in workloads if "ref_latency_ms" in w]

    return {
        "passed": passed,
        "failed": failed,
        "total": total,
        "pass_rate": f"{passed}/{total}",
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "avg_ref_latency_ms": sum(ref_latencies) / len(ref_latencies) if ref_latencies else None,
        "avg_speedup": sum(speedups) / len(speedups) if speedups else None,
        "min_speedup": min(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
    }


def print_workload_table(name: str, results: Dict):
    """Print a per-workload results table."""
    if "error" in results:
        print(f"\n  {name}: ERROR — {results['error']}")
        return

    workloads = results.get("workloads", [])
    # Sort by seq_len
    workloads_sorted = sorted(workloads, key=lambda w: w.get("seq_len", 0))

    print(f"\n{'='*80}")
    print(f"  Solution: {name}")
    print(f"{'='*80}")
    print(f"  {'seq_len':>8}  {'status':>10}  {'latency':>10}  {'ref_lat':>10}  {'speedup':>8}  {'abs_err':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")

    for w in workloads_sorted:
        seq = str(w.get("seq_len", "?"))
        status = w.get("status", "?")
        lat = f"{w['latency_ms']:.3f}" if "latency_ms" in w else "—"
        ref = f"{w['ref_latency_ms']:.3f}" if "ref_latency_ms" in w else "—"
        spd = f"{w['speedup']:.2f}x" if "speedup" in w else "—"
        err = f"{w['max_abs_err']:.2e}" if "max_abs_err" in w else "—"

        status_icon = "✓" if status == "passed" else "✗"
        print(f"  {seq:>8}  {status_icon} {status:>8}  {lat:>10}  {ref:>10}  {spd:>8}  {err:>10}")


def print_comparison_table(all_results: Dict[str, Dict]):
    """Print a comparison summary across all solutions."""
    if len(all_results) < 2:
        return

    print(f"\n{'='*80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Solution':<30} {'Pass':>6} {'Avg Lat':>10} {'Avg Spdup':>10} {'Min Spd':>8} {'Max Spd':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    for name, results in all_results.items():
        summary = compute_summary(results)
        if "error" in summary:
            print(f"  {name:<30} ERROR: {summary['error']}")
            continue

        pr = summary["pass_rate"]
        al = f"{summary['avg_latency_ms']:.3f}" if summary["avg_latency_ms"] else "—"
        asp = f"{summary['avg_speedup']:.3f}x" if summary["avg_speedup"] else "—"
        mnsp = f"{summary['min_speedup']:.2f}x" if summary["min_speedup"] else "—"
        mxsp = f"{summary['max_speedup']:.2f}x" if summary["max_speedup"] else "—"
        print(f"  {name:<30} {pr:>6} {al:>10} {asp:>10} {mnsp:>8} {mxsp:>8}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark & rank FlashInfer kernel solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--kernel", "-k",
        nargs="+",
        help="Path(s) to kernel.py file(s) to benchmark",
    )
    parser.add_argument(
        "--solution", "-s",
        nargs="+",
        help="Path(s) to pre-packed solution.json file(s)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: fewer iterations for faster dev feedback",
    )
    args = parser.parse_args()

    dataset_path = get_dataset_path()

    # Collect solutions to benchmark
    solutions: List[tuple] = []  # [(name, Solution)]

    if args.kernel:
        for kp in args.kernel:
            kpath = Path(kp)
            if not kpath.exists():
                print(f"ERROR: Kernel file not found: {kp}")
                sys.exit(1)
            name = kpath.stem if len(args.kernel) == 1 else kpath.parent.name + "/" + kpath.name
            print(f"Packing kernel: {kp} ...")
            sol = pack_kernel_file(str(kpath), name=name)
            solutions.append((name, sol))

    if args.solution:
        from flashinfer_bench import Solution
        for sp in args.solution:
            spath = Path(sp)
            if not spath.exists():
                print(f"ERROR: Solution file not found: {sp}")
                sys.exit(1)
            sol = Solution.model_validate_json(spath.read_text())
            solutions.append((sol.name, sol))

    # Default: pack from config.toml
    if not solutions:
        print("No --kernel or --solution specified. Packing from config.toml ...")
        sol = pack_from_config()
        solutions.append((sol.name, sol))

    # Run benchmarks
    all_results = {}
    for name, sol in solutions:
        print(f"\nBenchmarking: {name} ...")
        t0 = time.time()
        results = run_benchmark(sol, dataset_path, quick=args.quick)
        elapsed = time.time() - t0
        all_results[name] = results

        print_workload_table(name, results)
        summary = compute_summary(results)
        if "error" not in summary:
            print(f"\n  Summary: {summary['pass_rate']} passed | "
                  f"avg speedup: {summary['avg_speedup']:.3f}x | "
                  f"avg latency: {summary['avg_latency_ms']:.3f} ms | "
                  f"bench time: {elapsed:.1f}s")

    # Comparison table (if multiple)
    print_comparison_table(all_results)

    # Save to JSON
    if args.output:
        output = {}
        for name, results in all_results.items():
            output[name] = {
                "workloads": results.get("workloads", []),
                "summary": compute_summary(results),
            }
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
