"""YAML → Pipeline IR. Loud validation; no silent-{} degradation
(trellis defect addressed in spec Phase 3)."""
from __future__ import annotations

import yaml

from .ir import Edge, Node, Pipeline


def load_pipeline_yaml(text: str) -> Pipeline:
    """Load and validate a pipeline from YAML text.

    Raises ValueError loudly for:
    - YAML parse errors
    - Missing required fields (name, deadline, nodes)
    - Invalid node/edge structure
    - Validation failures (cycles, missing node refs, etc.)
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"pipeline YAML parse error: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"pipeline YAML must be a mapping, got {type(data).__name__}")

    for required in ("name", "deadline", "nodes"):
        if required not in data:
            raise ValueError(f"pipeline YAML missing required field: {required}")

    nodes = [_node_from_yaml(n) for n in data.get("nodes") or []]
    edges = [Edge(**e) for e in data.get("edges") or []]

    pipe = Pipeline(
        name=str(data["name"]),
        deadline=float(data["deadline"]),
        nodes=nodes,
        edges=edges
    )
    pipe.validate()
    return pipe


def _node_from_yaml(d: dict) -> Node:
    """Convert YAML node dict to Node IR."""
    if not isinstance(d, dict) or "id" not in d or "kind" not in d:
        raise ValueError(f"node entry missing id/kind: {d!r}")

    kw = {k: v for k, v in d.items() if k != "body"}
    body = None

    if d.get("kind") == "composite" and isinstance(d.get("body"), dict):
        body = Pipeline(
            name=d["body"]["name"],
            deadline=float(d["body"]["deadline"]),
            nodes=[_node_from_yaml(x) for x in d["body"].get("nodes") or []],
            edges=[Edge(**e) for e in d["body"].get("edges") or []],
        )

    return Node(body=body, **kw)
