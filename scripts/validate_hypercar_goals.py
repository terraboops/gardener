#!/usr/bin/env python3
"""Validate Gardener against hypercar's 6 goals end-to-end.

Wraps `gardener.bench.cli` to run two passes against a chosen model and emit
a Markdown report at `docs/validation-<date>.md`:

  Pass 1 — fast gates @ 16K context, stratified N=3 (for swap-mode honesty):
           smoke, coherence, decode_speed, prefill_speed, memory_profile,
           humaneval_lite, mmlu_pro, ruler, livecodebench
  Pass 2 — long-context NIAH @ 524288 (Goal 1), single run (~hours wall-time)

The 6 hypercar goals are mapped to the gardener phases that measure them:

  G1: 1M context             → niah @ 524288  (we validate 512K, the hypercar
                                                Qwen3.6 record)
  G2: Quality (4 evals)      → humaneval_lite + mmlu_pro + ruler + livecodebench
  G3: Decode ≥ 50 tok/s      → decode_speed
  G4: Prefill ≥ 500 tok/s    → prefill_speed
  G5: Swap p90 < 100 MB/s    → memory_profile (swap_delta_gb, swap_mode)
  G6: 48 GB M4 Pro fit       → memory_profile (metal_peak_gb < 48)

Usage:
  python scripts/validate_hypercar_goals.py \\
      --model mlx-community/Qwen3.6-35B-A3B-4bit

Quick run (skip the multi-hour 512K NIAH, target a smaller context):
  python scripts/validate_hypercar_goals.py --quick

Use a model already cached:
  python scripts/validate_hypercar_goals.py --no-download

Reports land at: docs/validation-YYYY-MM-DD.md  + raw JSON in same dir.

NB: a full run on Qwen3.6-35B is several hours wall-time. The fast gates
(Pass 1) are ~30-60 min total at 35B. The 512K NIAH (Pass 2) is ~2 hr alone.
Run overnight unless you're patient.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
DEFAULT_OUT_DIR = REPO_ROOT / "docs"

# Hypercar's 6 goals as specified in omlx-mamba3/CLAUDE.md.
HYPERCAR_TARGETS = {
    "G1_context_tokens": 524288,         # 512K validated (1M is the open frontier)
    "G2_humaneval_min": 0.35,            # hypercar gate; CLAUDE.md achieves 0.95
    "G2_mmlu_pro_min": 0.35,             # hypercar gate; CLAUDE.md achieves 0.62
    "G2_ruler_mk_min": 0.80,
    "G2_ruler_vt_min": 0.70,
    "G2_lcb_min": 0.30,                  # hypercar gate; CLAUDE.md achieves 0.40
    "G3_decode_tok_s_min": 50.0,
    "G4_prefill_tok_s_min": 500.0,
    "G5_swap_mode_max": "fast",          # need swap_delta_gb < 5 (fast bucket)
    "G6_metal_peak_gb_max": 48.0,
}


# --------------------------------------------------------------------------
# Pass orchestration
# --------------------------------------------------------------------------

def ensure_model(model_id: str, *, do_download: bool) -> None:
    """Force a one-time download via mlx_lm.load if requested."""
    if not do_download:
        return
    print(f"[validate] ensuring model is cached: {model_id}")
    code = (
        "from mlx_lm import load; "
        f"load({model_id!r}); print('cached:', {model_id!r})"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, cwd=str(REPO_ROOT),
    )


def run_bench(
    *,
    model: str,
    phases: list[str],
    out_path: Path,
    n_runs: int,
    niah_context: int,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Invoke `gardener.bench.cli` and return the parsed JSON report."""
    cli_args = [
        sys.executable, "-m", "gardener.bench.cli",
        "--model", model,
        "--phases", ",".join(phases),
        "--n", str(n_runs),
        "--niah-context", str(niah_context),
        "--out", str(out_path),
        "--quiet",
    ]
    if extra_args:
        cli_args.extend(extra_args)
    print(f"[validate] running: {' '.join(cli_args)}")
    subprocess.run(cli_args, check=False, cwd=str(REPO_ROOT))
    if not out_path.exists():
        raise RuntimeError(f"bench produced no output at {out_path}")
    return json.loads(out_path.read_text())


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------

def _phase_metric(runs: list[dict], phase_name: str, metric: str) -> list[float]:
    """Collect a phase metric across runs, stratified later if needed."""
    out: list[float] = []
    for r in runs:
        for p in r.get("phases", []):
            if p.get("name") == phase_name and p.get("status") == "passed":
                m = p.get("metrics", {}).get(metric)
                if isinstance(m, (int, float)):
                    out.append(float(m))
    return out


def _phase_status_counts(runs: list[dict], phase_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in runs:
        for p in r.get("phases", []):
            if p.get("name") == phase_name:
                s = p.get("status", "unknown")
                counts[s] = counts.get(s, 0) + 1
    return counts


def _stratify_by_swap_mode(runs: list[dict]) -> dict[str, list[dict]]:
    """Group runs by swap_mode (hypercar's bimodal discriminator)."""
    grouped: dict[str, list[dict]] = {}
    for r in runs:
        mode = r.get("swap_mode", "neutral")
        grouped.setdefault(mode, []).append(r)
    return grouped


def _p50_p95(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    s = sorted(values)
    p50 = statistics.median(s)
    idx95 = max(0, int(0.95 * (len(s) - 1) + 0.5))
    p95 = s[idx95]
    return p50, p95


def assemble_report(
    *,
    model: str,
    pass1: dict[str, Any],
    pass2: dict[str, Any] | None,
    quick: bool,
) -> str:
    """Build a per-goal Markdown report from the bench JSON outputs."""
    runs1 = pass1.get("runs", [])
    runs2 = (pass2 or {}).get("runs", [])
    now = dt.datetime.now().isoformat(timespec="seconds")

    md: list[str] = []
    md.append(f"# Gardener — Hypercar-goal validation report\n")
    md.append(f"**Model**: `{model}`  ")
    md.append(f"**Generated**: {now}  ")
    md.append(f"**Mode**: {'quick (skipped 512K NIAH)' if quick else 'full'}  ")
    md.append(f"**Fast-gate runs (N)**: {len(runs1)}  ")
    if pass2 is not None:
        md.append(f"**Long-context run**: 1 (512K NIAH)  ")
    md.append("")

    # --- Swap-mode stratification (the bimodal honesty caveat) ----------
    strat = _stratify_by_swap_mode(runs1)
    md.append("## Swap-mode stratification (Pass 1)\n")
    md.append("Hypercar's bimodal finding: same-mode runs are within 3% of each "
              "other; cross-mode (fast vs slow) is +44% wall-time. The discriminator "
              "is `swap_delta < 5 GB → fast; > 8 GB → slow; else neutral`. "
              "Numbers below are reported *per swap mode* so fast/slow runs are "
              "not silently averaged.\n")
    md.append("| swap_mode | runs | mean Metal peak (GB) |")
    md.append("|---|---:|---:|")
    for mode in ("fast", "neutral", "slow"):
        group = strat.get(mode, [])
        if not group:
            continue
        mean_metal = (
            statistics.mean(r["metal_peak_gb"] for r in group)
            if all("metal_peak_gb" in r for r in group) else None
        )
        md.append(
            f"| {mode} | {len(group)} | "
            f"{mean_metal:.1f}" if mean_metal is not None else f"| {mode} | {len(group)} | —"
        )
    md.append("")

    # --- Per-goal table ------------------------------------------------
    md.append("## Per-goal results\n")
    md.append("| Goal | Target | Measured | Status |")
    md.append("|---|---|---|---|")

    rows: list[tuple[str, str, str, str]] = []

    # G1 — context
    if pass2 is not None:
        niah_runs = pass2.get("runs", [])
        passed = any(_phase_status_counts(niah_runs, "niah").get("passed", 0) > 0
                     for _ in [None])
        passed = sum(_phase_status_counts(niah_runs, "niah").get("passed", 0)
                     for _ in [None]) > 0
        rows.append((
            "**G1** Context window",
            f"≥ {HYPERCAR_TARGETS['G1_context_tokens']:,} tokens (NIAH PASS)",
            f"NIAH @ 524,288 → {'PASS' if passed else 'FAIL'}",
            "✅" if passed else "❌",
        ))
    else:
        rows.append((
            "**G1** Context window",
            f"≥ {HYPERCAR_TARGETS['G1_context_tokens']:,} tokens (NIAH PASS)",
            "skipped (--quick)",
            "⏭️",
        ))

    # G2 — 4 quality evals
    def _quality_row(phase: str, label: str, threshold_key: str, metric: str):
        vals = _phase_metric(runs1, phase, metric)
        target = HYPERCAR_TARGETS[threshold_key]
        if not vals:
            rows.append((f"**G2** {label}",
                         f"≥ {target * 100:.0f}%",
                         "not run",
                         "⚠️"))
            return
        mean = statistics.mean(vals)
        ok = mean >= target
        rows.append((f"**G2** {label}",
                     f"≥ {target * 100:.0f}%",
                     f"{mean * 100:.1f}% (N={len(vals)})",
                     "✅" if ok else "❌"))

    _quality_row("humaneval_lite", "HumanEval-Lite", "G2_humaneval_min", "pass_rate")
    _quality_row("mmlu_pro", "MMLU-Pro", "G2_mmlu_pro_min", "accuracy")
    # RULER has two thresholds — handled separately below.
    ruler_mk_vals: list[float] = []
    ruler_vt_vals: list[float] = []
    for r in runs1:
        for p in r.get("phases", []):
            if p.get("name") == "ruler" and p.get("status") == "passed":
                m = p.get("metrics", {})
                if "mk_16k_accuracy" in m:
                    ruler_mk_vals.append(float(m["mk_16k_accuracy"]))
                if "vt_4k_accuracy" in m:
                    ruler_vt_vals.append(float(m["vt_4k_accuracy"]))
    if ruler_mk_vals:
        mean = statistics.mean(ruler_mk_vals)
        ok = mean >= HYPERCAR_TARGETS["G2_ruler_mk_min"]
        rows.append(("**G2** RULER multi-key @ 16K",
                     f"≥ {HYPERCAR_TARGETS['G2_ruler_mk_min'] * 100:.0f}%",
                     f"{mean * 100:.1f}% (N={len(ruler_mk_vals)})",
                     "✅" if ok else "❌"))
    if ruler_vt_vals:
        mean = statistics.mean(ruler_vt_vals)
        ok = mean >= HYPERCAR_TARGETS["G2_ruler_vt_min"]
        rows.append(("**G2** RULER variable-tracking @ 4K",
                     f"≥ {HYPERCAR_TARGETS['G2_ruler_vt_min'] * 100:.0f}%",
                     f"{mean * 100:.1f}% (N={len(ruler_vt_vals)})",
                     "✅" if ok else "❌"))
    _quality_row("livecodebench", "LiveCodeBench", "G2_lcb_min", "pass_rate")

    # G3 — decode speed
    decode_vals = _phase_metric(runs1, "decode_speed", "tok_per_sec")
    if decode_vals:
        p50, p95 = _p50_p95(decode_vals)
        ok = p50 >= HYPERCAR_TARGETS["G3_decode_tok_s_min"]
        rows.append(("**G3** Decode speed",
                     f"≥ {HYPERCAR_TARGETS['G3_decode_tok_s_min']:.0f} tok/s",
                     f"p50={p50:.1f}, p95={p95:.1f} (N={len(decode_vals)})",
                     "✅" if ok else "❌"))
    else:
        rows.append(("**G3** Decode speed",
                     f"≥ {HYPERCAR_TARGETS['G3_decode_tok_s_min']:.0f} tok/s",
                     "not run", "⚠️"))

    # G4 — prefill speed
    prefill_vals = _phase_metric(runs1, "prefill_speed", "tok_per_sec")
    if prefill_vals:
        p50, p95 = _p50_p95(prefill_vals)
        ok = p50 >= HYPERCAR_TARGETS["G4_prefill_tok_s_min"]
        rows.append(("**G4** Prefill speed",
                     f"≥ {HYPERCAR_TARGETS['G4_prefill_tok_s_min']:.0f} tok/s",
                     f"p50={p50:.1f}, p95={p95:.1f} (N={len(prefill_vals)})",
                     "✅" if ok else "❌"))
    else:
        rows.append(("**G4** Prefill speed",
                     f"≥ {HYPERCAR_TARGETS['G4_prefill_tok_s_min']:.0f} tok/s",
                     "not run", "⚠️"))

    # G5 — swap pressure
    swap_deltas = [r.get("swap_delta_gb") for r in runs1
                   if isinstance(r.get("swap_delta_gb"), (int, float))]
    fast_count = sum(1 for r in runs1 if r.get("swap_mode") == "fast")
    if swap_deltas:
        p50, p95 = _p50_p95([float(s) for s in swap_deltas])
        # G5's target is "p90 swap I/O < 100 MB/s" (a rate, not a delta).
        # We approximate by checking the fast-mode discriminator: ≥80% of
        # runs in fast bucket (swap < 5 GB) is a strong proxy.
        ok = (fast_count / max(len(runs1), 1)) >= 0.8
        rows.append(("**G5** Swap pressure",
                     "p90 swap I/O < 100 MB/s (proxy: ≥80% runs in fast bucket)",
                     f"swap_delta p50={p50:.1f} GB, p95={p95:.1f} GB; "
                     f"{fast_count}/{len(runs1)} fast",
                     "✅" if ok else "❌"))
    else:
        rows.append(("**G5** Swap pressure",
                     "p90 swap I/O < 100 MB/s",
                     "not measured", "⚠️"))

    # G6 — memory fit
    metal_peaks = [r.get("metal_peak_gb") for r in runs1
                   if isinstance(r.get("metal_peak_gb"), (int, float))]
    if metal_peaks:
        max_peak = max(float(p) for p in metal_peaks)
        ok = max_peak < HYPERCAR_TARGETS["G6_metal_peak_gb_max"]
        rows.append(("**G6** 48 GB M4 Pro fit",
                     f"Metal peak < {HYPERCAR_TARGETS['G6_metal_peak_gb_max']:.0f} GB",
                     f"max peak {max_peak:.1f} GB (N={len(metal_peaks)})",
                     "✅" if ok else "❌"))
    else:
        rows.append(("**G6** 48 GB M4 Pro fit",
                     f"Metal peak < {HYPERCAR_TARGETS['G6_metal_peak_gb_max']:.0f} GB",
                     "not measured", "⚠️"))

    for goal, target, measured, status in rows:
        md.append(f"| {goal} | {target} | {measured} | {status} |")

    md.append("")
    md.append("## Honest caveats\n")
    md.append("- **G2 numbers depend on the *model*, not on Gardener.** "
              "Gardener's role is to make the *intelligence ceiling* reachable "
              "on your hardware — the HX ports enable larger models on the "
              "same memory budget. The model you pick determines the "
              "score; the platform determines whether you can run it at all.")
    md.append("- **G3 plateau is expected at 16K+** without speculative decoding. "
              "Run with `--draft-model` set on the `AgentProfile` for the "
              "predicted ~3× decode speedup at α ≥ 0.5; measurement of α on "
              "your specific workload is in `scripts/probe_speculative_decoding.py` "
              "(if present) or via repeated bench runs.")
    md.append("- **G5 is a proxy.** Hypercar's exact metric is *sustained swap "
              "I/O rate* over N≥8 runs; we approximate with the swap-mode "
              "discriminator (fast/slow buckets) which is faster to compute "
              "and surfaces the bimodal pattern. For a rigorous G5 measurement, "
              "run `--n 8` and inspect the per-run swap_delta_gb distribution.")
    md.append("- **MInference (HX6) is NOT enabled** in this validation. "
              "The shipped calibration table is a synthetic placeholder; using "
              "it without a real calibration likely regresses NIAH/RULER quality. "
              "Run `scripts/minference_calibrate.py --model <id>` first, then "
              "re-validate with sparse prefill enabled.")
    md.append("- **TQ3+model integration is xfailed** in the test suite due to "
              "an upstream mlx/mlx_lm `quantized_matmul` signature drift "
              "(see `tests/test_turboquant_kv.py`). DuoKVCache sidesteps it; "
              "switch to DuoKV if you need quantized KV at long context today.")
    md.append("")

    md.append("---\n")
    md.append("*Raw JSON for both passes is alongside this report in the same "
              "directory. Re-run with `--n 8` for a tighter G5 measurement, "
              "or `--quick` to skip the multi-hour 512K NIAH.*\n")

    return "\n".join(md) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="validate_hypercar_goals",
        description="Validate Gardener against hypercar's 6 goals.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model id (default: {DEFAULT_MODEL})")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), type=Path,
                   help="Directory for the report + raw JSON (default: docs/)")
    p.add_argument("--n", type=int, default=3,
                   help="Stratified runs for fast gates (default: 3; use 8 for tight G5)")
    p.add_argument("--quick", action="store_true",
                   help="Skip the 512K NIAH pass (G1 partial); validates G2-G6 only")
    p.add_argument("--niah-context", type=int, default=524288,
                   help="NIAH context length for G1 (default: 524288 = 512K)")
    p.add_argument("--fast-niah-context", type=int, default=16384,
                   help="NIAH context for fast Pass 1 (default: 16384)")
    p.add_argument("--no-download", action="store_true",
                   help="Skip the upfront model-cache warm-up")
    args = p.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    date = dt.date.today().isoformat()
    safe_model = args.model.replace("/", "_")
    pass1_json = out_dir / f"validation-{date}-{safe_model}-pass1.json"
    pass2_json = out_dir / f"validation-{date}-{safe_model}-pass2.json"
    report_md = out_dir / f"validation-{date}-{safe_model}.md"

    ensure_model(args.model, do_download=not args.no_download)

    # Pass 1: fast gates at 16K context, stratified.
    pass1_phases = [
        "smoke", "coherence",
        "decode_speed", "prefill_speed", "memory_profile",
        "humaneval_lite", "mmlu_pro", "ruler", "livecodebench",
    ]
    pass1 = run_bench(
        model=args.model,
        phases=pass1_phases,
        out_path=pass1_json,
        n_runs=args.n,
        niah_context=args.fast_niah_context,
    )

    # Pass 2: long-context NIAH (G1). Single run because each takes hours.
    pass2: dict[str, Any] | None = None
    if not args.quick:
        pass2 = run_bench(
            model=args.model,
            phases=["niah"],
            out_path=pass2_json,
            n_runs=1,
            niah_context=args.niah_context,
        )

    report = assemble_report(
        model=args.model, pass1=pass1, pass2=pass2, quick=args.quick,
    )
    report_md.write_text(report)
    print(f"\n[validate] report → {report_md}")
    print(f"[validate] pass-1 raw → {pass1_json}")
    if pass2 is not None:
        print(f"[validate] pass-2 raw → {pass2_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
