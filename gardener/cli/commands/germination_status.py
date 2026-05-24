"""gardener germination-status — cross-agent T3 fail-rate aggregator.

Walks the registry, reads each agent's germination.yaml, summarizes
status counts + per-drafter breakdown + failure detail. Output is
human-readable table by default, JSON with --format json.

Feeds the T3 input to `gardener calibrate-wizard` when the latter is
given --t3-from-germination-status PATH (slice F3+ integration).

Usage:
    gardener germination-status
    gardener germination-status --format json --output germination.json
    gardener germination-status --fail-on-t3-exceed 0.05
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from gardener.wizard.germination import aggregate

from ..registry import load_registry


def cmd_germination_status(args) -> int:
    registry = load_registry()
    agg = aggregate(registry)

    if args.format == "json":
        out = json.dumps(agg.to_dict(), indent=2, sort_keys=True, default=_json_default)
        if args.output:
            Path(args.output).write_text(out + "\n")
        else:
            print(out, flush=True)
    else:
        _print_table(agg)
        if args.output:
            Path(args.output).write_text(
                json.dumps(agg.to_dict(), indent=2, sort_keys=True, default=_json_default)
                + "\n"
            )
            print(f"\n(also wrote JSON to {args.output})", flush=True)

    if args.fail_on_t3_exceed is not None:
        rate = agg.t3_germination_fail_rate
        if rate is None:
            # No decided agents yet — can't fail something undefined.
            print(
                "germination-status: no decided agents yet; "
                "--fail-on-t3-exceed cannot evaluate",
                file=sys.stderr,
            )
            return 0
        if rate > args.fail_on_t3_exceed:
            print(
                f"germination-status: T3 fail rate {rate:.1%} exceeds "
                f"threshold {args.fail_on_t3_exceed:.1%}",
                file=sys.stderr,
            )
            return 1
    return 0


def _json_default(o):
    # GerminationAggregate.to_dict already produces plain types, but
    # registry may have Path values stashed somewhere; keep this safe.
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"unserializable: {type(o).__name__}")


def _print_table(agg) -> None:
    print(f"Germination status across {agg.total_agents} registered agent(s):", flush=True)
    print(flush=True)
    if agg.total_agents == 0:
        print("  (no agents registered — try `gardener wizard` to plant one)", flush=True)
        return

    print("  Status breakdown:", flush=True)
    for status in ("pending", "passed", "failed"):
        n = agg.per_status.get(status, 0)
        marker = {"pending": "…", "passed": "✓", "failed": "✗"}[status]
        print(f"    {marker} {status:8s} {n:4d}", flush=True)

    if agg.t3_germination_fail_rate is not None:
        print(flush=True)
        print(
            f"  T3 germination fail rate: {agg.t3_germination_fail_rate:.1%} "
            f"({agg.per_status['failed']} / {agg.decided_count} decided)",
            flush=True,
        )
    else:
        print(flush=True)
        print(
            "  T3 germination fail rate: undefined (no decided agents yet)",
            flush=True,
        )

    if agg.per_drafter:
        print(flush=True)
        print("  Per-drafter breakdown:", flush=True)
        for drafter, counts in sorted(agg.per_drafter.items()):
            decided = counts["passed"] + counts["failed"]
            rate = (counts["failed"] / decided) if decided else None
            rate_s = f"{rate:.1%}" if rate is not None else "—"
            print(
                f"    {drafter:48s}  pending={counts['pending']:3d} "
                f"passed={counts['passed']:3d} failed={counts['failed']:3d} "
                f"(T3 fail: {rate_s})",
                flush=True,
            )

    if agg.failures:
        print(flush=True)
        print(f"  Failed agents ({len(agg.failures)}):", flush=True)
        for f in agg.failures:
            print(
                f"    ✗ {f['agent']} (drafted_by={f['drafted_by']}, "
                f"on_fail={f['on_fail']}, planted_at={f['planted_at']})",
                flush=True,
            )
            for e in f["errors"]:
                print(f"        - {e}", flush=True)

    if agg.unknown:
        print(flush=True)
        print(f"  ⚠ Unknown/corrupt germination state ({len(agg.unknown)}):", flush=True)
        for u in agg.unknown:
            print(f"    - {u}", flush=True)
