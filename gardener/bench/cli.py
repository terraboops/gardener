"""CLI for the Gardener bench harness.

Usage:
    python -m gardener.bench.cli --model ID --phases smoke,coherence --n 1 --out report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .harness import BenchConfig, BenchReport, run_bench


def main(argv=None):
    p = argparse.ArgumentParser(prog="gardener.bench")
    p.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="HuggingFace model ID (default: Qwen2.5-0.5B-Instruct-4bit)",
    )
    p.add_argument(
        "--phases",
        default="smoke,coherence,niah,decode_speed,prefill_speed,memory_profile",
        help="comma-separated phase names",
    )
    p.add_argument("--niah-context", type=int, default=4096,
                   help="NIAH context length in tokens (default: 4096)")
    p.add_argument("--decode-tokens", type=int, default=30,
                   help="number of tokens to generate for decode_speed phase")
    p.add_argument("--prefill-tokens", type=int, default=4096,
                   help="prompt length in tokens for prefill_speed phase")
    p.add_argument("--n", type=int, default=1,
                   help="number of stratified runs (default: 1)")
    p.add_argument("--out", default="bench_report.json",
                   help="output JSON path (default: bench_report.json)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-run summary lines")
    args = p.parse_args(argv)

    cfg = BenchConfig(
        model_id=args.model,
        phases=[s.strip() for s in args.phases.split(",") if s.strip()],
        niah_context_tokens=args.niah_context,
        decode_tokens=args.decode_tokens,
        prefill_tokens=args.prefill_tokens,
    )

    reports: list[BenchReport] = [run_bench(cfg) for _ in range(args.n)]

    out = {
        "runs": [r._to_dict() for r in reports],
        "summary": _stratify(reports),
    }
    Path(args.out).write_text(json.dumps(out, default=str, indent=2))

    if not args.quiet:
        for i, r in enumerate(reports):
            ok = "PASS" if r.passed() else "FAIL"
            print(
                f"run {i + 1}/{args.n}  {ok}  swap_mode={r.swap_mode}  "
                f"metal_peak={r.metal_peak_gb:.1f} GB  "
                f"gates_met={r.gates_met}  "
                f"gates_failed={r.gates_failed}"
            )

    return 0 if all(r.passed() for r in reports) else 1


def _stratify(reports: list[BenchReport]) -> dict:
    by_mode: dict[str, list[BenchReport]] = {}
    for r in reports:
        by_mode.setdefault(r.swap_mode, []).append(r)
    summary: dict = {}
    for mode, group in by_mode.items():
        if not group:
            continue
        pass_rate = sum(1 for r in group if r.passed()) / len(group)
        summary[mode] = {
            "n": len(group),
            "pass_rate": pass_rate,
            "metal_peak_gb_mean": sum(r.metal_peak_gb for r in group) / len(group),
        }
    return summary


if __name__ == "__main__":
    sys.exit(main())
