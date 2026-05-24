"""gardener wizard — interactive agent creator.

Two modes:
  --inline KEY=VALUE...   non-interactive scripted path (for tests + scripts)
  (no flag)               interactive prompt-driven path

When `--drafter MODEL_ID` is passed (or --mock-drafter-fixtures for tests),
the interactive path enters Q2.1's draft flow:
    drafter generates → parser validates → user accept/edit/regenerate/skip
Falls back to the template-based prompt if drafter not configured or skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml

from ..agent_dir import DEFAULT_MODEL, AgentConfig, scaffold_agent, validate_name
from ..registry import register


MODEL_SUGGESTIONS = [
    ("mlx-community/Qwen2.5-0.5B-Instruct-4bit", "tiny, fast, for testing"),
    ("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", "coding-focused, ~4 GB"),
    ("mlx-community/Qwen3.6-35B-A3B-4bit", "hypercar-class, ~20 GB; needs M4 Pro 48 GB"),
]

DEFAULT_NAME = "my-agent"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256


def _ask(prompt: str, default: str = "", *, stream=None) -> str:
    stream = stream or sys.stdin
    suffix = f" [{default}]" if default else ""
    sys.stdout.write(f"{prompt}{suffix}: ")
    sys.stdout.flush()
    line = stream.readline()
    if not line:
        return default
    val = line.rstrip("\n").strip()
    return val if val else default


def _parse_inline_kwargs(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--inline arg must be key=value: {item!r}")
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip()
    return out


def _draft_system_prompt(name: str, purpose: str) -> str:
    purpose = purpose or "a general-purpose helpful assistant"
    return (
        f"You are {name}, an agent whose purpose is: {purpose}\n"
        "\n"
        "Respond clearly and concisely. If you don't know something, say so\n"
        "rather than guess. When asked to take an action you can't safely\n"
        "complete, explain why.\n"
    )


# ---------------------------------------------------------------------------
# Drafter integration (Q2.1)
# ---------------------------------------------------------------------------


def _maybe_build_drafter(args):
    """Return a Drafter instance if --drafter or --mock-drafter-fixtures set."""
    mock_fixtures = getattr(args, "mock_drafter_fixtures", None)
    drafter_id = getattr(args, "drafter", None)
    if mock_fixtures:
        from gardener.wizard.drafter import MockDrafter
        return MockDrafter.from_yaml(
            mock_fixtures, name=f"mock:{Path(mock_fixtures).name}"
        )
    if drafter_id:
        # Lazy MLX import so CI without MLX can still load this module.
        from gardener.wizard.drafter_mlx import MlxDrafter
        return MlxDrafter(drafter_id)
    return None


def _try_drafter_flow(*, drafter, purpose: str, agent_name: str):
    """Run the accept/edit/regen flow. Returns (system_prompt, knowledge_facts)
    or None if the user skipped to template fallback.

    Lets WizardCancelled propagate so the wizard exits non-zero on quit.
    """
    from gardener.wizard.cli_flow import run_draft_flow

    seeded = run_draft_flow(
        purpose=purpose, agent_name=agent_name, drafter=drafter
    )
    if seeded is None:
        return None
    return seeded.system_prompt, seeded.knowledge_facts


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def cmd_wizard(args) -> int:
    # Inline (non-interactive) path — primarily for tests + scripts.
    if args.inline is not None:
        kw = _parse_inline_kwargs(args.inline)
        name = kw.get("name", DEFAULT_NAME)
        description = kw.get("description", "")
        purpose = kw.get("purpose", "")
        model = kw.get("model", DEFAULT_MODEL)
        temperature = float(kw.get("temperature", DEFAULT_TEMPERATURE))
        max_tokens = int(kw.get("max_tokens", DEFAULT_MAX_TOKENS))
        path_str = kw.get("path", f"./{name}")
        seed_knowledge = kw.get("seed_knowledge", "")
        force = bool(int(kw.get("force", "0")))
        drafted_knowledge: tuple[str, ...] = ()
        drafted_prompt: Optional[str] = None
    else:
        # Interactive prompts.
        print("Gardener wizard — let's create a new agent.\n", flush=True)
        name = _ask("name", DEFAULT_NAME)
        try:
            validate_name(name)
        except ValueError as e:
            print(f"gardener wizard: {e}", file=sys.stderr)
            return 1
        description = _ask("description (one-liner)", "")
        purpose = _ask(
            "purpose (plain language: what does this agent do?)",
            "a general-purpose helpful assistant",
        )

        # --- Drafter flow (Q2.1) ---
        drafted_prompt = None
        drafted_knowledge = ()
        drafter = _maybe_build_drafter(args)
        if drafter is not None:
            try:
                draft_result = _try_drafter_flow(
                    drafter=drafter, purpose=purpose, agent_name=name
                )
            except Exception as e:
                # WizardCancelled or drafter load error
                from gardener.wizard.cli_flow import WizardCancelled
                if isinstance(e, WizardCancelled):
                    print("gardener wizard: cancelled.", file=sys.stderr)
                    return 1
                print(
                    f"gardener wizard: drafter failed: {e}; falling back to template",
                    file=sys.stderr,
                )
                draft_result = None
            if draft_result is not None:
                drafted_prompt, drafted_knowledge = draft_result

        print("\nModel suggestions:", flush=True)
        for m, note in MODEL_SUGGESTIONS:
            print(f"  - {m}  ({note})", flush=True)
        model = _ask("model (HF id)", DEFAULT_MODEL)
        temperature = float(
            _ask(
                "temperature (0.0 deterministic, 0.7 creative)",
                str(DEFAULT_TEMPERATURE),
            )
        )
        max_tokens = int(
            _ask("max_tokens per response", str(DEFAULT_MAX_TOKENS))
        )
        path_str = _ask("path", f"./{name}")
        # If drafter seeded knowledge, skip the manual seed_knowledge prompt.
        if drafted_knowledge:
            seed_knowledge = ""
        else:
            seed_knowledge = _ask(
                "seed knowledge (optional one-line fact; empty to skip)", ""
            )
        force = False

    # Validate name for inline path too (validate_name raises ValueError)
    try:
        validate_name(name)
    except ValueError as e:
        print(f"gardener wizard: {e}", file=sys.stderr)
        return 1

    path = Path(path_str).expanduser().resolve()

    # Confirmation block — always print, even in inline mode (visible in logs).
    print("\nReady to create:", flush=True)
    print(f"  name:         {name}", flush=True)
    print(f"  description:  {description!r}", flush=True)
    print(f"  model:        {model}", flush=True)
    print(f"  path:         {path}", flush=True)
    print(f"  temperature:  {temperature}", flush=True)
    print(f"  max_tokens:   {max_tokens}", flush=True)
    if drafted_prompt is not None:
        n_facts = len(drafted_knowledge)
        print(
            f"  prompt:       drafted ({len(drafted_prompt)} chars) "
            f"+ {n_facts} seed knowledge fact(s)",
            flush=True,
        )
    elif seed_knowledge:
        print(
            f"  seed knowledge: 1 item: {seed_knowledge}",
            flush=True,
        )
    else:
        print("  seed knowledge: (none)", flush=True)

    if args.inline is None and not args.yes:
        confirm = _ask("\nCreate?", "Y")
        if confirm.lower() not in ("y", "yes", ""):
            print("aborted.", flush=True)
            return 1

    try:
        cfg = scaffold_agent(
            path, name=name, model=model, description=description, force=force
        )
    except (ValueError, FileExistsError) as e:
        print(f"gardener wizard: {e}", file=sys.stderr)
        return 1

    # System prompt: drafted > template
    prompt_text = drafted_prompt if drafted_prompt is not None else _draft_system_prompt(name, purpose)
    (path / "prompt.md").write_text(prompt_text)

    cfg.temperature = temperature
    cfg.max_tokens = max_tokens
    (path / "agent.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    )
    register(name, path)

    # Knowledge facts: drafted bullets > manual single line
    facts_to_seed: list[tuple[str, str]] = []
    if drafted_knowledge:
        for fact in drafted_knowledge:
            facts_to_seed.append(("wizard-drafted", fact))
    elif seed_knowledge:
        facts_to_seed.append(("wizard-seeded", seed_knowledge))

    if facts_to_seed:
        from gardener.knowledge import KnowledgeObject, KnowledgeStore
        store = KnowledgeStore(path / "knowledge", agent=name)
        for predicate_tag, fact in facts_to_seed:
            # Make each fact's predicate triple unique by putting the
            # (truncated) fact text in the object slot — otherwise
            # multiple drafted facts collide on semantic_hash and overwrite
            # each other in the store.
            obj_slot = fact[:64] or "empty"
            kid = store.write(
                KnowledgeObject(
                    predicates=[[predicate_tag, "states", obj_slot]],
                    insight=fact,
                    justification=(
                        "drafted by wizard drafter at germination"
                        if predicate_tag == "wizard-drafted"
                        else "seeded interactively by a human at wizard time"
                    ),
                    source_agent=name,
                )
            )
            store.mark_tested(kid, helped=True)
        print(
            f"\nSeeded {len(facts_to_seed)} knowledge object(s).",
            flush=True,
        )

    print("\nCreated.", flush=True)
    print(f"Try: gardener query {name} 'hello, who are you?'", flush=True)
    return 0
