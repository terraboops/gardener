"""Model-quality eval gates (HX12 tier).

Ported subset of hypercar's Phase 3b/3c/3d/4 quality gates (RULER, MMLU-Pro,
LiveCodeBench, HumanEval-Lite). These are model-quality benchmarks, not
agent-platform gates — they're how you answer 'does Gardener with this model
pass HumanEval?' style questions.
"""
from .humaneval import HUMANEVAL_LITE, run_humaneval_lite, HumanEvalResult

__all__ = ["HUMANEVAL_LITE", "run_humaneval_lite", "HumanEvalResult"]
