"""Agent directory layout + agent.yaml load/validate."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_PROMPT = (
    "You are a helpful, concise assistant.\n"
    "Answer questions accurately and admit when you don't know.\n"
)


@dataclass
class AgentConfig:
    name: str
    model: str = DEFAULT_MODEL
    description: str = ""
    prompt_path: str = "prompt.md"
    tools: list = field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 0.0
    draft_model: Optional[str] = None
    num_draft_tokens: int = 8
    created_at: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "AgentConfig":
        if "name" not in d:
            raise ValueError("agent.yaml missing required field: name")
        if "model" not in d:
            raise ValueError("agent.yaml missing required field: model")
        if not SLUG_RE.match(str(d["name"])):
            raise ValueError(
                f"invalid agent name {d['name']!r}: must match {SLUG_RE.pattern}"
            )
        # Only pass known fields to avoid unexpected keyword errors
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "description": self.description,
            "prompt_path": self.prompt_path,
            "tools": list(self.tools),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "draft_model": self.draft_model,
            "num_draft_tokens": self.num_draft_tokens,
            "created_at": self.created_at,
        }


def validate_name(name: str) -> None:
    if not SLUG_RE.match(name):
        raise ValueError(
            f"invalid agent name {name!r}: must be lowercase, alphanumeric, "
            "may contain - or _, and start with a letter or digit"
        )


def scaffold_agent(
    path: Path,
    *,
    name: str,
    model: str = DEFAULT_MODEL,
    description: str = "",
    force: bool = False,
) -> AgentConfig:
    validate_name(name)
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(
            f"refusing to scaffold over non-empty directory {path}: use --force"
        )
    path.mkdir(parents=True, exist_ok=True)
    for sub in ("knowledge", "journal", "cache", "pipelines"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    (path / "prompt.md").write_text(DEFAULT_PROMPT)
    cfg = AgentConfig(
        name=name,
        model=model,
        description=description,
        created_at=dt.datetime.now().isoformat(timespec="seconds"),
    )
    (path / "agent.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return cfg


def load_agent(path: Path) -> AgentConfig:
    cfg_path = path / "agent.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"not an agent directory (no agent.yaml): {path}"
        )
    return AgentConfig.from_dict(yaml.safe_load(cfg_path.read_text()))


def load_prompt(path: Path, cfg: AgentConfig) -> str:
    return (path / cfg.prompt_path).read_text()
