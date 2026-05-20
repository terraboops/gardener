"""Registry of registered agents at ~/.gardener/registry.json."""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_REGISTRY = Path.home() / ".gardener" / "registry.json"


def registry_path() -> Path:
    env = os.environ.get("GARDENER_REGISTRY")
    return Path(env) if env else DEFAULT_REGISTRY


def load_registry() -> dict[str, str]:
    p = registry_path()
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save_registry(reg: dict[str, str]) -> None:
    p = registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2, sort_keys=True))


def register(name: str, path: Path) -> None:
    reg = load_registry()
    reg[name] = str(path.resolve())
    save_registry(reg)


def unregister(name: str) -> None:
    reg = load_registry()
    reg.pop(name, None)
    save_registry(reg)


def resolve(name_or_path: str) -> Path:
    """Resolve an arg to an agent directory: short-name -> registry lookup;
    else treat as a path."""
    reg = load_registry()
    if name_or_path in reg:
        return Path(reg[name_or_path])
    p = Path(name_or_path)
    if p.exists() and (p / "agent.yaml").exists():
        return p
    raise KeyError(
        f"unknown agent {name_or_path!r}: not registered and not an agent "
        f"directory (no agent.yaml). Run `gardener list` to see registered agents."
    )
