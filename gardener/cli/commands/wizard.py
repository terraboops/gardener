"""gardener wizard — interactive agent creator."""
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
    print(
        f"  seed knowledge: "
        f"{('1 item: ' + seed_knowledge) if seed_knowledge else '(none)'}",
        flush=True,
    )

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

    # Customize beyond Q1's defaults: drafted system prompt + temperature + max_tokens.
    (path / "prompt.md").write_text(_draft_system_prompt(name, purpose))
    cfg.temperature = temperature
    cfg.max_tokens = max_tokens
    (path / "agent.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    )
    register(name, path)

    if seed_knowledge:
        from gardener.knowledge import KnowledgeObject, KnowledgeStore

        store = KnowledgeStore(path / "knowledge", agent=name)
        kid = store.write(
            KnowledgeObject(
                predicates=[["wizard-seeded", "fact", "true"]],
                insight=seed_knowledge,
                justification="seeded interactively by a human at wizard time",
                source_agent=name,
            )
        )
        store.mark_tested(kid, helped=True)
        print(f"\nSeeded 1 knowledge object: {kid}", flush=True)

    print("\nCreated.", flush=True)
    print(f"Try: gardener query {name} 'hello, who are you?'", flush=True)
    return 0
