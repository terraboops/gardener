"""Model-quality eval gates (HX12 tier).

Ported subset of hypercar's Phase 3b/3c/3d/4 quality gates (RULER, MMLU-Pro,
LiveCodeBench, HumanEval-Lite). These are model-quality benchmarks, not
agent-platform gates — they're how you answer 'does Gardener with this model
pass HumanEval?' style questions.
"""
from .humaneval import HUMANEVAL_LITE, run_humaneval_lite, HumanEvalResult
from .mmlu_pro import MMLUProResult, extract_answer, format_prompt, load_mmlu_pro, run_mmlu_pro
from .ruler import (
    RulerTaskResult, RulerSuiteResult,
    RULER_QUICK_SUITE, RULER_FULL_SUITE,
    generate_multi_key_niah, generate_variable_tracking, generate_frequent_word,
    run_ruler_task, run_ruler_suite,
)
from .livecodebench import (
    LCBProblem, LCBProblemResult, LCBResult,
    LCB_RETRIES, LCB_RETRY_TEMP, LCB_TIMEOUT_S, LCB_MEMORY_MB,
    load_livecodebench, format_prompt as lcb_format_prompt,
    extract_last_code_block, execute_code, run_livecodebench,
)

__all__ = [
    "HUMANEVAL_LITE", "HumanEvalResult", "run_humaneval_lite",
    "MMLUProResult", "extract_answer", "format_prompt",
    "load_mmlu_pro", "run_mmlu_pro",
    "RulerTaskResult", "RulerSuiteResult",
    "RULER_QUICK_SUITE", "RULER_FULL_SUITE",
    "generate_multi_key_niah", "generate_variable_tracking", "generate_frequent_word",
    "run_ruler_task", "run_ruler_suite",
    "LCBProblem", "LCBProblemResult", "LCBResult",
    "LCB_RETRIES", "LCB_RETRY_TEMP", "LCB_TIMEOUT_S", "LCB_MEMORY_MB",
    "load_livecodebench", "lcb_format_prompt",
    "extract_last_code_block", "execute_code", "run_livecodebench",
]
