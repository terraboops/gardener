"""Interactive accept/edit/regenerate/skip flow for the wizard's drafter.

Used by `gardener wizard` when a drafter is configured. Atomic per draft
(per terra 2026-05-23): users accept/edit/regenerate the whole draft, not
individual sections. Edit drops to $EDITOR with the full sectioned-markdown.

Flow:
    1. draft = drafter.draft(purpose)
    2. result = parse(draft)
    3. if not result.ok and retries_left > 0: retry once (T1 auto-recovery)
    4. if still not ok: surface error, ask skip-or-quit
    5. display result; prompt [a]ccept / [e]dit / [r]egenerate / [s]kip / [q]uit
    6. branch:
         a → return (system_prompt, knowledge_facts)
         e → run external editor, re-parse, loop
         r → re-call drafter, loop
         s → return None (caller falls back to template)
         q → raise WizardCancelled
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from .drafter import Drafter
from .parser import ParseResult, parse


class WizardCancelled(Exception):
    """Raised when the user types [q]uit during the flow."""


@dataclass(frozen=True)
class DraftedSeed:
    system_prompt: str
    knowledge_facts: tuple[str, ...]


# Type aliases for testability (subprocess-piped stdin is the actual use).
Prompter = Callable[[str], str]
EditorFn = Callable[[str], str]


def run_draft_flow(
    *,
    purpose: str,
    agent_name: str,
    drafter: Drafter,
    out_stream=None,
    prompter: Optional[Prompter] = None,
    editor_fn: Optional[EditorFn] = None,
    max_t1_retries: int = 1,
) -> Optional[DraftedSeed]:
    """Run the interactive accept/edit/regen/skip flow.

    Returns DraftedSeed on accept, None on skip (caller uses template).
    Raises WizardCancelled if the user quits.
    """
    out = out_stream or sys.stdout
    prompt = prompter or _stdin_prompter
    edit = editor_fn or _editor_fn

    while True:
        _print(out, f"\n[drafting via {drafter.name}...]")
        raw = drafter.draft(purpose, agent_name=agent_name)
        result = parse(raw)

        # T1 auto-recovery: one retry on parse failure.
        retries_left = max_t1_retries
        while not result.ok and retries_left > 0:
            _print(
                out,
                f"[draft did not parse ({len(result.errors)} error(s)); retrying...]",
            )
            for e in result.errors:
                _print(out, f"  - {e}")
            raw = drafter.draft(purpose, agent_name=agent_name)
            result = parse(raw)
            retries_left -= 1

        if not result.ok:
            _print(out, "\nDrafter failed to produce a parseable seed after retries.")
            for e in result.errors:
                _print(out, f"  - {e}")
            choice = prompt(
                "[s]kip drafter (use template), [r]etry, [q]uit? "
            ).strip().lower()
            if choice == "s":
                return None
            if choice == "q":
                raise WizardCancelled("user quit at draft failure")
            # retry → loop top
            continue

        _display(out, result, drafter_name=drafter.name)

        choice = prompt(
            "\n[a]ccept  [e]dit  [r]egenerate  [s]kip drafter  [q]uit: "
        ).strip().lower()

        if choice in ("a", "accept", ""):
            return DraftedSeed(
                system_prompt=result.system_prompt or "",
                knowledge_facts=result.knowledge_facts,
            )
        if choice in ("e", "edit"):
            edited = edit(raw)
            edited_result = parse(edited)
            if not edited_result.ok:
                _print(out, "Edited draft does not parse:")
                for e in edited_result.errors:
                    _print(out, f"  - {e}")
                # Loop back to display + prompt with the (still failing) edit
                # so the user can decide.
                result = edited_result
                raw = edited
                continue
            # Use the edited version directly without re-prompting; the
            # user already approved by editing-and-saving.
            return DraftedSeed(
                system_prompt=edited_result.system_prompt or "",
                knowledge_facts=edited_result.knowledge_facts,
            )
        if choice in ("r", "regenerate"):
            continue  # loop top, re-draft
        if choice in ("s", "skip"):
            return None
        if choice in ("q", "quit"):
            raise WizardCancelled("user quit at draft flow")
        _print(out, f"unrecognized choice {choice!r}; try again")


# ---------------------------------------------------------------------------
# Display + defaults
# ---------------------------------------------------------------------------


def _display(out, result: ParseResult, *, drafter_name: str) -> None:
    _print(out, "")
    _print(out, "— DRAFTED SEED " + "─" * 50)
    if result.system_prompt:
        _print(out, "System Prompt:")
        for line in result.system_prompt.splitlines():
            _print(out, f"  {line}")
    if result.knowledge_facts:
        _print(out, "")
        _print(out, f"Knowledge ({len(result.knowledge_facts)} facts):")
        for i, fact in enumerate(result.knowledge_facts, 1):
            _print(out, f"  {i}. {fact}")
    if result.decline_reason:
        _print(out, "")
        _print(out, "Decline:")
        _print(out, f"  {result.decline_reason}")
    if result.clarify_question:
        _print(out, "")
        _print(out, "Clarify:")
        _print(out, f"  {result.clarify_question}")
    if result.warnings:
        _print(out, "")
        _print(out, "(warnings)")
        for w in result.warnings:
            _print(out, f"  ⚠ {w}")
    _print(out, "─" * 64)
    _print(out, f"  drafted by: {drafter_name}")


def _print(stream, msg: str) -> None:
    stream.write(msg + "\n")
    stream.flush()


def _stdin_prompter(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    line = sys.stdin.readline()
    return line.rstrip("\n") if line else ""


def _editor_fn(initial_text: str) -> str:
    """Drop to $EDITOR (or VISUAL, or vi) with the initial text.

    Returns the saved text. Used for [e]dit in the draft flow.
    """
    editor = (
        os.environ.get("GARDENER_EDITOR")
        or os.environ.get("VISUAL")
        or os.environ.get("EDITOR")
        or "vi"
    )
    with tempfile.NamedTemporaryFile(
        mode="w+",
        suffix=".md",
        delete=False,
        encoding="utf-8",
    ) as tf:
        tf.write(initial_text)
        path = tf.name
    try:
        subprocess.run([*editor.split(), path], check=False)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
