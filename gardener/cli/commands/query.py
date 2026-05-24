"""gardener query -- one-shot ask an agent."""
from __future__ import annotations

import datetime as dt
import sys

from ..agent_dir import load_agent, load_prompt
from ..registry import resolve


def cmd_query(args) -> int:
    try:
        agent_path = resolve(args.agent)
    except KeyError as e:
        print(f"gardener query: {e}", file=sys.stderr)
        return 1

    try:
        cfg = load_agent(agent_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"gardener query: {e}", file=sys.stderr)
        return 1

    try:
        system_prompt = load_prompt(agent_path, cfg)
    except FileNotFoundError:
        system_prompt = ""

    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.max_tokens
    temperature = args.temperature if args.temperature is not None else cfg.temperature

    # Lazy import to avoid pulling mlx_lm on every CLI call.
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:
        print(f"gardener query: mlx_lm not available: {e}", file=sys.stderr)
        return 1

    model, tok = load(cfg.model)
    sampler = make_sampler(temp=temperature)

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.question},
    ]
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = (
            f"<|system|>\n{system_prompt}\n<|user|>\n{args.question}\n<|assistant|>\n"
        )

    response = generate(
        model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler,
        verbose=False,
    )
    print(response)

    # Journal the query.
    from ...journal import EventJournal

    journal = EventJournal(agent_path / "journal")
    journal.append(
        "query",
        {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "question": args.question,
            "response": response,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    # Q2.1-I: germination tracking — feed the first 3 calls into the state
    # machine. No-op past the observation window.
    from gardener.wizard.germination import (
        is_response_error,
        load as load_germination,
        record_call,
        save as save_germination,
    )

    state = load_germination(agent_path)
    if state.is_observing():
        failed = is_response_error(response)
        new_state, events = record_call(
            state,
            succeeded=not failed,
            error_reason=("empty response" if failed else None),
        )
        save_germination(agent_path, new_state)
        for evt in events:
            journal.append("germination", evt)
    return 0
