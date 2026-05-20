"""Prose DSL → Pipeline IR. Reuses lark + indentation-token technique;
new grammar (not trellis's tiny one) expressive enough for the full DAG IR.
Loud validation; no silent {} degradation."""
from __future__ import annotations

import re
from typing import Any

from lark import Lark, Transformer, v_args

from .ir import Edge, Node, Pipeline


# ---------------------------------------------------------------------------
# Indentation preprocessor: emits <INDENT>/<DEDENT>/<NEWLINE> tokens.
# ---------------------------------------------------------------------------
def _preprocess(text: str) -> str:
    lines: list[str] = []
    indent_stack = [0]
    for raw in text.splitlines():
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(stripped)
        while indent < indent_stack[-1]:
            indent_stack.pop()
            lines.append("<DEDENT>")
        if indent > indent_stack[-1]:
            indent_stack.append(indent)
            lines.append("<INDENT>")
        lines.append(stripped)
        lines.append("<NEWLINE>")
    while len(indent_stack) > 1:
        indent_stack.pop()
        lines.append("<DEDENT>")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------
# Notes on LALR compatibility:
#   - input_field uses explicit two alternatives to avoid shift/reduce on "if"
#   - All _NL/_INDENT/_DEDENT are silent (underscore prefix)
#   - body_field nests a full pipeline recursively; lark LALR handles this fine
GRAMMAR = r"""
start: pipeline

pipeline: "pipeline" NAME ":" _NL _INDENT deadline_field node+ _DEDENT
deadline_field: "deadline" NUMBER _NL

?node: agent_node | human_node | composite_node

agent_node: "agent" NAME ":" _NL _INDENT agent_field+ _DEDENT
?agent_field: prompt_field | max_tokens_field | temperature_field
            | retry_field | timeout_field | input_field

human_node: "human" NAME ":" _NL _INDENT human_field+ _DEDENT
?human_field: ask_field | timeout_field | on_timeout_field | retry_field | input_field

composite_node: "composite" NAME ":" _NL _INDENT composite_field+ _DEDENT
?composite_field: retry_field | timeout_field | input_field | body_field

body_field: "body" ":" _NL _INDENT pipeline _DEDENT

prompt_field:       "prompt" ":" QUOTED _NL
max_tokens_field:   "max_tokens" ":" INT _NL
temperature_field:  "temperature" ":" NUMBER _NL
ask_field:          "ask" ":" QUOTED _NL
retry_field:        "retry" ":" INT _NL
timeout_field:      "timeout" ":" NUMBER _NL
on_timeout_field:   "on_timeout" ":" ON_TIMEOUT _NL

// Two explicit alternatives avoids shift/reduce conflict on optional "if"
input_field: "input" NAME "from" NAME "if" QUOTED _NL   -> input_cond
           | "input" NAME "from" NAME _NL               -> input_plain

ON_TIMEOUT: "dead-letter" | "escalate" | "fallback-agent"
NAME: /[a-zA-Z_][a-zA-Z0-9_-]*/
NUMBER: /[0-9]+(\.[0-9]+)?/
INT: /[0-9]+/
QUOTED: /"(?:[^"\\]|\\.)*"/

_NL: "<NEWLINE>"
_INDENT: "<INDENT>"
_DEDENT: "<DEDENT>"

%import common.WS_INLINE
%ignore WS_INLINE
%ignore /\n+/
"""

_UNESCAPE = re.compile(r"\\(.)", re.DOTALL)


def _unq(s: str) -> str:
    return _UNESCAPE.sub(r"\1", s[1:-1])


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
@v_args(inline=False)
class _T(Transformer):
    def NAME(self, t):      return str(t)
    def NUMBER(self, t):    return float(t)
    def INT(self, t):       return int(t)
    def QUOTED(self, t):    return _unq(str(t))
    def ON_TIMEOUT(self, t): return str(t)

    def deadline_field(self, c): return c[0]   # just the float

    def prompt_field(self, c):       return ("prompt",      c[0])
    def max_tokens_field(self, c):   return ("max_tokens",  c[0])
    def temperature_field(self, c):  return ("temperature", c[0])
    def ask_field(self, c):          return ("ask",         c[0])
    def retry_field(self, c):        return ("retry",       c[0])
    def timeout_field(self, c):      return ("timeout",     c[0])
    def on_timeout_field(self, c):   return ("on_timeout",  c[0])
    def body_field(self, c):         return ("body",        c[0])

    def input_plain(self, c):
        iname, src = c[0], c[1]
        return ("input", (iname, src, None))

    def input_cond(self, c):
        iname, src, cond = c[0], c[1], c[2]
        return ("input", (iname, src, cond))

    # Alias so both rule names are handled
    def input_field(self, c):
        # Should not be reached given the two aliases above, but keep as safety
        if len(c) == 2:
            return ("input", (c[0], c[1], None))
        return ("input", (c[0], c[1], c[2]))

    def agent_node(self, c):     return ("agent",     c[0], c[1:])
    def human_node(self, c):     return ("human",     c[0], c[1:])
    def composite_node(self, c): return ("composite", c[0], c[1:])

    def pipeline(self, c):
        name = c[0]
        deadline = c[1]          # float from deadline_field
        node_tuples = c[2:]
        nodes: list[Node] = []
        edges: list[Edge] = []
        for kind, nid, fields in node_tuples:
            params: dict = {}
            retry: dict = {"max": 1, "on": "fail"}
            timeout = None
            on_timeout = None
            ask = None
            body = None
            for k, v in fields:
                if k == "input":
                    iname, src, cond = v
                    edges.append(Edge(from_id=src, to_id=nid,
                                      input_name=iname, condition=cond))
                elif k == "retry":      retry = {"max": v, "on": "fail"}
                elif k == "timeout":    timeout = v
                elif k == "on_timeout": on_timeout = v
                elif k == "ask":        ask = v
                elif k == "body":       body = v
                elif k == "prompt":     params["system_prompt"] = v
                else:                   params[k] = v
            node = Node(
                id=nid,
                kind=kind,
                params=params,
                retry=retry,
                timeout=timeout,
                on_timeout=on_timeout,
                ask=ask,
                body=body,
                agent=nid if kind == "agent" else None,
            )
            nodes.append(node)
        return Pipeline(name=name, deadline=deadline, nodes=nodes, edges=edges)

    def start(self, c):
        return c[0]


# ---------------------------------------------------------------------------
# Parser instance (module-level; LALR compilation happens once at import)
# ---------------------------------------------------------------------------
try:
    _PARSER = Lark(GRAMMAR, parser="lalr", maybe_placeholders=False)
except Exception:
    # Fallback to Earley if LALR conflicts are irresolvable at install time
    _PARSER = Lark(GRAMMAR, parser="earley", maybe_placeholders=False)


def parse_pipeline_prose(text: str) -> Pipeline:
    """Parse a prose DSL string into a validated Pipeline IR.

    Raises ValueError loudly on any parse or validation failure.
    """
    preprocessed = _preprocess(text)
    try:
        tree = _PARSER.parse(preprocessed)
    except Exception as e:
        raise ValueError(f"prose parse error: {e}") from e
    pipe = _T().transform(tree)
    pipe.validate()
    return pipe
