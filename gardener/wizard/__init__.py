"""Q2.1: LLM-assisted wizard.

Drafts a system prompt + 0-3 seed knowledge facts from a plain-language
seed-purpose, using a small local LLM (the "drafter"). The cultivation
loop grows whatever the wizard plants; this package's job is to plant
something coherent enough to germinate.

Submodules:
    config       — thresholds, drafter selection, corpus policy
    parser       — sectioned-markdown parsing (T1 metric source)
    drafter      — Drafter Protocol + MockDrafter for CI
    drafter_mlx  — real MLX-backed drafter (local-only, gated by `model` marker)
    adversarial  — safety classifier for the adversarial corpus stratum
    thresholds   — baseline evaluation + threshold-comparison verdict logic

Locked thresholds (terra-locked 2026-05-23, T1 tightened 2026-05-23 (V)):
    T1 parse fail max:         5%  (tightened from 10% in F5 after
                                    confirming auto-retry hides single
                                    failures from the user)
    T2 acceptance fail max:   20% (logged only — NOT a switch trigger)
    T3 germination fail max:   5%

Adversarial stratum: observability-only. Pass criterion is "fail safely",
never folded into the T1/T3 threshold calculation.
"""
