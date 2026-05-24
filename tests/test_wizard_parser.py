"""Tests for gardener/wizard/parser.py — sectioned-markdown parsing.

T1 metric source. The parser is neutral about what kind of input produced
the output: it just extracts recognized sections and flags structural
problems. Consumers (adversarial classifier, wizard CLI) interpret the
section mix.

Recognized sections:
    System Prompt   — the drafted system prompt (a plantable seed)
    Knowledge       — bullet list of 0-3 seed knowledge facts
    Decline         — drafter refused to produce a seed (with reason)
    Clarify         — drafter wants more information (with question)

A parse is OK if AT LEAST ONE recognized section is present and well-formed.
"""
from __future__ import annotations

import textwrap

import pytest

from gardener.wizard.parser import ParseResult, parse


# ---------------------------------------------------------------------------
# Normal seed parses
# ---------------------------------------------------------------------------


def test_minimal_valid_just_system_prompt():
    text = textwrap.dedent(
        """
        ### System Prompt
        You are a helpful coding agent.
        """
    )
    r = parse(text)
    assert r.ok
    assert r.system_prompt == "You are a helpful coding agent."
    assert r.knowledge_facts == ()
    assert r.errors == ()


def test_system_prompt_plus_three_knowledge_facts():
    text = textwrap.dedent(
        """
        ### System Prompt
        You are a Python expert. Be concise.

        ### Knowledge
        - Python uses indentation for blocks
        - PEP 8 is the style guide
        - The standard library is large
        """
    )
    r = parse(text)
    assert r.ok
    assert "Python expert" in r.system_prompt
    assert r.knowledge_facts == (
        "Python uses indentation for blocks",
        "PEP 8 is the style guide",
        "The standard library is large",
    )


def test_more_than_three_facts_truncates_with_warning():
    text = textwrap.dedent(
        """
        ### System Prompt
        Helper.

        ### Knowledge
        - one
        - two
        - three
        - four
        - five
        """
    )
    r = parse(text)
    assert r.ok
    assert len(r.knowledge_facts) == 3
    assert r.knowledge_facts == ("one", "two", "three")
    assert any("truncated" in w.lower() for w in r.warnings)


def test_asterisk_bullets_also_work():
    text = textwrap.dedent(
        """
        ### System Prompt
        Helper.

        ### Knowledge
        * alpha
        * beta
        """
    )
    r = parse(text)
    assert r.ok
    assert r.knowledge_facts == ("alpha", "beta")


def test_h2_heading_level_also_accepted():
    text = textwrap.dedent(
        """
        ## System Prompt
        Helper.

        ## Knowledge
        - one fact
        """
    )
    r = parse(text)
    assert r.ok
    assert r.system_prompt == "Helper."
    assert r.knowledge_facts == ("one fact",)


def test_leading_and_trailing_whitespace_tolerated():
    text = "\n\n\n   ### System Prompt   \n\n   Helper agent.   \n\n\n"
    r = parse(text)
    assert r.ok
    assert r.system_prompt == "Helper agent."


def test_multiline_system_prompt_preserved():
    text = textwrap.dedent(
        """
        ### System Prompt
        You are a helpful agent.

        When the user asks a question, answer it.
        When you don't know, say so.
        """
    )
    r = parse(text)
    assert r.ok
    # Internal blank lines + line breaks preserved.
    assert "answer it" in r.system_prompt
    assert "say so" in r.system_prompt
    assert r.system_prompt.count("\n") >= 2


# ---------------------------------------------------------------------------
# Decline / Clarify (adversarial-stratum-relevant)
# ---------------------------------------------------------------------------


def test_decline_only_is_valid():
    text = textwrap.dedent(
        """
        ### Decline
        The input was empty; I can't seed an agent from no information.
        """
    )
    r = parse(text)
    assert r.ok
    assert r.system_prompt is None
    assert "empty" in r.decline_reason


def test_clarify_only_is_valid():
    text = textwrap.dedent(
        """
        ### Clarify
        Which language should this agent work in — Python, TypeScript, or both?
        """
    )
    r = parse(text)
    assert r.ok
    assert r.system_prompt is None
    assert "Python" in r.clarify_question


# ---------------------------------------------------------------------------
# Failure modes (T1 fail = these)
# ---------------------------------------------------------------------------


def test_empty_input_is_t1_failure():
    r = parse("")
    assert not r.ok
    assert any("no recognized section" in e.lower() for e in r.errors)


def test_no_recognized_sections_is_t1_failure():
    text = "This is just some prose with no headings.\nNothing useful here."
    r = parse(text)
    assert not r.ok


def test_empty_system_prompt_is_t1_failure():
    text = textwrap.dedent(
        """
        ### System Prompt

        ### Knowledge
        - one
        """
    )
    r = parse(text)
    assert not r.ok
    assert any("system prompt" in e.lower() and "empty" in e.lower() for e in r.errors)


def test_duplicate_system_prompt_section_is_t1_failure():
    text = textwrap.dedent(
        """
        ### System Prompt
        First version.

        ### System Prompt
        Second version.
        """
    )
    r = parse(text)
    assert not r.ok
    assert any("duplicate" in e.lower() for e in r.errors)


def test_unparseable_returns_structured_result_not_exception():
    # The parser MUST NOT raise — T1 failure is a measurement, not a crash.
    r = parse("```\nnot markdown\n```\n## ## ## broken")
    assert isinstance(r, ParseResult)
    # Result may be ok or not depending on what's found; the contract is
    # just that we get a ParseResult, not an exception.


# ---------------------------------------------------------------------------
# Combinations
# ---------------------------------------------------------------------------


def test_seed_plus_clarify_is_valid_but_unusual_warning():
    # The drafter probably shouldn't both seed AND ask for clarification;
    # flag as warning, but T1 still passes (structurally well-formed).
    text = textwrap.dedent(
        """
        ### System Prompt
        Helper.

        ### Clarify
        Did you mean Python or TypeScript?
        """
    )
    r = parse(text)
    assert r.ok
    assert r.system_prompt == "Helper."
    assert "Python" in r.clarify_question
    assert any("both seed and clarify" in w.lower() for w in r.warnings)


def test_code_fence_inside_section_preserved():
    text = textwrap.dedent(
        """
        ### System Prompt
        You are a Python expert. Example:

        ```python
        def hello():
            print("hi")
        ```

        That's the style.
        """
    )
    r = parse(text)
    assert r.ok
    assert "```python" in r.system_prompt
    assert "def hello" in r.system_prompt
