"""Sectioned-markdown parser for drafter output.

Recognized sections (case-insensitive, h2 or h3):
    System Prompt   — drafted system prompt body
    Knowledge       — bullet list of 0-3 seed facts (`-` or `*`)
    Decline         — drafter refused (with reason)
    Clarify         — drafter wants more info (with question)

A parse is OK if at least ONE recognized section is present and well-formed.
The parser is *neutral* about whether the section mix is appropriate for
the input — that's the adversarial classifier's call.

This module is the T1 (Parse) metric source. It MUST NOT raise on any
input string; failure is reported via ParseResult.errors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# Section header regex: ## or ### followed by a known section name.
# Captures the name in canonical lowercase form.
_KNOWN_SECTIONS = ("system prompt", "knowledge", "decline", "clarify")
_HEADER_RE = re.compile(
    r"^[ \t]*#{2,3}[ \t]+(.+?)[ \t]*$",
    re.MULTILINE,
)
_BULLET_RE = re.compile(r"^[ \t]*[-*][ \t]+(.+?)[ \t]*$")
_MAX_KNOWLEDGE_FACTS = 3


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    system_prompt: Optional[str] = None
    knowledge_facts: tuple[str, ...] = ()
    decline_reason: Optional[str] = None
    clarify_question: Optional[str] = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def parse(text: str) -> ParseResult:
    """Parse drafter output into recognized sections.

    Never raises. Failure is reported via ParseResult.ok=False + errors.
    """
    errors: list[str] = []
    warnings: list[str] = []

    sections = _split_sections(text)
    if not sections:
        return ParseResult(
            ok=False,
            errors=("no recognized sections found in input",),
        )

    # Check for duplicate sections of the same canonical name.
    seen: dict[str, int] = {}
    for name, _ in sections:
        seen[name] = seen.get(name, 0) + 1
    for name, count in seen.items():
        if count > 1:
            errors.append(f"duplicate '{name}' section ({count} occurrences)")

    body_by_name = {name: body for name, body in sections}

    system_prompt = body_by_name.get("system prompt")
    if system_prompt is not None:
        # .strip() removes leading/trailing whitespace from the whole body
        # while preserving internal line breaks + indentation (needed for
        # code fences inside the prompt).
        system_prompt = system_prompt.strip()
        # Empty after strip → T1 failure
        if not system_prompt:
            errors.append("system prompt section is empty")
            system_prompt = None

    knowledge_facts: tuple[str, ...] = ()
    if "knowledge" in body_by_name:
        bullets = _extract_bullets(body_by_name["knowledge"])
        if len(bullets) > _MAX_KNOWLEDGE_FACTS:
            warnings.append(
                f"knowledge has {len(bullets)} facts; truncated to "
                f"{_MAX_KNOWLEDGE_FACTS}"
            )
            bullets = bullets[:_MAX_KNOWLEDGE_FACTS]
        knowledge_facts = tuple(bullets)

    decline_reason = body_by_name.get("decline")
    if decline_reason is not None:
        decline_reason = decline_reason.strip()

    clarify_question = body_by_name.get("clarify")
    if clarify_question is not None:
        clarify_question = clarify_question.strip()

    # Sanity warning: seeding AND clarifying in the same response is unusual.
    if system_prompt and clarify_question:
        warnings.append(
            "drafter produced both seed and clarify; intent ambiguous"
        )

    # OK if at least one of system_prompt / decline / clarify is present
    # AND no fatal errors were collected.
    has_payload = bool(system_prompt or decline_reason or clarify_question)
    ok = has_payload and not errors

    return ParseResult(
        ok=ok,
        system_prompt=system_prompt,
        knowledge_facts=knowledge_facts,
        decline_reason=decline_reason,
        clarify_question=clarify_question,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Walk the text; emit (canonical_name, body) for each recognized header.

    Bodies retain internal blank lines but are stripped of the trailing
    newline before the next header. Unknown headers are ignored (text under
    them is appended to no section).
    """
    if not text:
        return []
    out: list[tuple[str, str]] = []
    current_name: Optional[str] = None
    current_body: list[str] = []
    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            header = m.group(1).strip().lower()
            if header in _KNOWN_SECTIONS:
                if current_name is not None:
                    out.append((current_name, "\n".join(current_body)))
                current_name = header
                current_body = []
                continue
            # Unknown header → close the current section and skip.
            if current_name is not None:
                out.append((current_name, "\n".join(current_body)))
                current_name = None
                current_body = []
            continue
        if current_name is not None:
            current_body.append(line)
    if current_name is not None:
        out.append((current_name, "\n".join(current_body)))
    return out


def _extract_bullets(body: str) -> list[str]:
    """Pull `-` and `*` bulleted lines from a section body, in order."""
    bullets: list[str] = []
    for line in body.splitlines():
        m = _BULLET_RE.match(line)
        if m:
            bullets.append(m.group(1).strip())
    return bullets
