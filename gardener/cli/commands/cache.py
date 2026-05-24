"""gardener cache — warm/list/rm pre-loaded subagent contexts.

`subagents` is imported lazily inside the warm/use functions because it
transitively pulls in `mlx_lm`, which is Mac-only. Lazy import lets the
CLI (and its no-MLX subcommands, e.g. wizard/list/observe/calibrate) load
in a Linux CI environment without MLX installed.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from ..agent_dir import load_agent, load_prompt
from ..registry import resolve


def _resolve_warmup_text(args, agent_path: Path, cfg) -> str:
    specified = sum(
        bool(x)
        for x in (
            args.text,
            args.file,
            getattr(args, "use_prompt_md", False),
        )
    )
    if specified > 1:
        raise ValueError(
            "specify at most one of --text / --file / --use-prompt-md"
        )
    if args.text is not None:
        return args.text
    if args.file is not None:
        return Path(args.file).read_text()
    # Default: agent's prompt.md (whether --use-prompt-md was set or not).
    return load_prompt(agent_path, cfg)


def _derived_cache_name(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"warm-{h}"


def cmd_cache_warm(args) -> int:
    try:
        agent_path = resolve(args.agent)
    except KeyError as e:
        print(f"gardener cache warm: {e}", file=sys.stderr)
        return 1

    try:
        cfg = load_agent(agent_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"gardener cache warm: {e}", file=sys.stderr)
        return 1

    try:
        warmup_text = _resolve_warmup_text(args, agent_path, cfg)
    except (ValueError, FileNotFoundError) as e:
        print(f"gardener cache warm: {e}", file=sys.stderr)
        return 1

    cache_name = args.name or _derived_cache_name(warmup_text)

    print(f"[cache] loading model {cfg.model} ...", file=sys.stderr, flush=True)
    try:
        from mlx_lm import load
        from ...subagents import AgentProfile, SubagentRegistry
    except ImportError as e:
        print(f"gardener cache warm: mlx_lm not available: {e}", file=sys.stderr)
        return 1

    model, tok = load(cfg.model)
    reg = SubagentRegistry(model, tok, root=agent_path / "cache")
    profile = AgentProfile(
        name=cache_name,
        system_prompt=load_prompt(agent_path, cfg),
        warmup_text=warmup_text,
        params={"max_tokens": cfg.max_tokens, "temperature": cfg.temperature},
    )
    path = reg.register(profile)
    size = path.stat().st_size if path.exists() else 0
    print(f"[cache] warmed: {cache_name}", flush=True)
    print(f"        path:   {path}", flush=True)
    print(f"        size:   {size:,} bytes", flush=True)
    print(f"        chars:  {len(warmup_text):,} (warmup_text)", flush=True)
    return 0


def cmd_cache_list(args) -> int:
    try:
        agent_path = resolve(args.agent)
    except KeyError as e:
        print(f"gardener cache list: {e}", file=sys.stderr)
        return 1

    cache_dir = agent_path / "cache"
    if not cache_dir.exists():
        print(f"(no cache directory at {cache_dir})", flush=True)
        return 0
    entries = sorted(cache_dir.glob("*.safetensors"))
    if not entries:
        print("(no warm caches yet)", flush=True)
        return 0
    print(f"{'NAME':<32s}  {'BYTES':>12s}  MODIFIED", flush=True)
    print(f"{'-' * 32}  {'-' * 12}  {'-' * 20}", flush=True)
    for p in entries:
        name = p.stem
        stat = p.stat()
        import datetime
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(
            timespec="seconds"
        )
        print(f"{name:<32s}  {stat.st_size:>12,d}  {mtime}", flush=True)
    return 0


def cmd_cache_rm(args) -> int:
    try:
        agent_path = resolve(args.agent)
    except KeyError as e:
        print(f"gardener cache rm: {e}", file=sys.stderr)
        return 1

    cache_dir = agent_path / "cache"
    target = cache_dir / f"{args.cache_name}.safetensors"
    meta = cache_dir / f"{args.cache_name}.meta"
    if not target.exists():
        print(f"no such cache: {args.cache_name}", file=sys.stderr, flush=True)
        return 1
    target.unlink()
    if meta.exists():
        meta.unlink()
    print(f"[cache] removed: {args.cache_name}", flush=True)
    return 0


def cmd_cache_use(args) -> int:
    try:
        agent_path = resolve(args.agent)
    except KeyError as e:
        print(f"gardener cache use: {e}", file=sys.stderr)
        return 1

    try:
        cfg = load_agent(agent_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"gardener cache use: {e}", file=sys.stderr)
        return 1

    cache_dir = agent_path / "cache"
    target = cache_dir / f"{args.cache_name}.safetensors"
    if not target.exists():
        print(f"no such cache: {args.cache_name}", file=sys.stderr, flush=True)
        return 1

    print(f"[cache] loading model {cfg.model} ...", file=sys.stderr, flush=True)
    try:
        from mlx_lm import load
        from ...subagents import AgentProfile, SubagentRegistry
    except ImportError as e:
        print(f"gardener cache use: mlx_lm not available: {e}", file=sys.stderr)
        return 1

    model, tok = load(cfg.model)
    reg = SubagentRegistry(model, tok, root=cache_dir)

    # Re-register from the .meta sidecar so the in-memory profile matches the
    # on-disk warm cache. register() is idempotent: it skips regeneration when
    # warmup_text + cache file are unchanged.
    meta_path = cache_dir / f"{args.cache_name}.meta"
    warmup_text = (
        meta_path.read_text()
        if meta_path.exists()
        else load_prompt(agent_path, cfg)
    )
    reg.register(
        AgentProfile(
            name=args.cache_name,
            system_prompt=load_prompt(agent_path, cfg),
            warmup_text=warmup_text,
            params={"max_tokens": cfg.max_tokens, "temperature": cfg.temperature},
        )
    )

    runner = reg.runner(args.cache_name)
    from ...pipeline.ir import Node

    node = Node(
        id="q",
        kind="agent",
        agent=args.cache_name,
        params={"max_tokens": cfg.max_tokens, "temperature": cfg.temperature},
    )
    response = runner(node, {"question": args.question})
    print(response, flush=True)
    return 0
