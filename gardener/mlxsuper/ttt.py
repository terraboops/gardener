"""Minimal test-time-training engine on stock mlx_lm + mlx_lm.tuner LoRA.

Reference map (semantics only, no omlx import): omlx/ttt.py:144 __init__,
:222 generate_candidates, :250 feedback, :272 feedback_from_execution,
:297 train_step. OPLoRA projection via gardener.mlxsuper.oplora."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .oplora import compute_svd_cache, project_lora_grads

logger = logging.getLogger("gardener.mlxsuper.ttt")


@dataclass
class Candidate:
    id: str
    prompt: str
    completion: str
    tokens: list[int]
    reward: float = 0.0
    signal: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class TrainStats:
    loss: float = 0.0
    num_positive: int = 0
    num_negative: int = 0
    elapsed_s: float = 0.0
    adapter_norm: float = 0.0


class TTTEngine:
    def __init__(self, model: nn.Module, tokenizer: Any, *, rank: int = 8,
                 lr: float = 1e-4, num_lora_layers: int = 4,
                 lora_scale: float = 20.0):
        from mlx_lm.tuner.utils import linear_to_lora_layers

        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.candidates: dict[str, Candidate] = {}
        self.history: list[TrainStats] = []
        model.freeze()
        linear_to_lora_layers(
            model, num_lora_layers,
            {"rank": rank, "scale": lora_scale, "dropout": 0.0},
        )
        self._svd_cache: dict[str, dict[str, mx.array]] = {}

    # --- candidate lifecycle -------------------------------------------------
    def generate_candidates(self, prompt: str, n: int = 4,
                            max_tokens: int = 64,
                            temperature: float = 0.8) -> list[Candidate]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature)
        out: list[Candidate] = []
        for _ in range(n):
            text = generate(self.model, self.tokenizer, prompt=prompt,
                             max_tokens=max_tokens, sampler=sampler,
                             verbose=False)
            cid = uuid.uuid4().hex[:8]
            c = Candidate(id=cid, prompt=prompt, completion=text,
                          tokens=self.tokenizer.encode(text))
            out.append(c)
            self.candidates[cid] = c
        return out

    def feedback(self, candidate_id: str, reward: float,
                 signal: str = "tool_call",
                 metadata: dict | None = None) -> None:
        c = self.candidates.get(candidate_id)
        if c is None:
            logger.warning("unknown candidate: %s", candidate_id)
            return
        c.reward = reward
        c.signal = signal
        if metadata:
            c.metadata.update(metadata)

    # --- training ------------------------------------------------------------
    def train_step(self, *, use_oplora: bool = False) -> TrainStats:
        """Cross-entropy on positive candidates; manual SGD on LoRA params.

        With use_oplora=True, LoRA grads are projected onto the safe
        subspace of each adapted base weight before the SGD step."""
        t0 = time.perf_counter()
        positive = [c for c in self.candidates.values() if c.reward > 0]
        negative = [c for c in self.candidates.values() if c.reward <= 0]
        if not positive:
            stats = TrainStats(num_negative=len(negative))
            self.history.append(stats)
            self.candidates.clear()
            return stats

        def loss_fn():
            total = mx.array(0.0)
            for c in positive:
                full = c.prompt + c.completion
                ids = mx.array([self.tokenizer.encode(full)])
                plen = len(self.tokenizer.encode(c.prompt))
                if plen >= ids.shape[1] - 1:
                    continue
                logits = self.model(ids)
                sl = logits[:, plen - 1:-1, :].reshape(-1, logits.shape[-1])
                lb = ids[:, plen:].reshape(-1)
                total = total + nn.losses.cross_entropy(sl, lb,
                                                        reduction="mean")
            return total / max(len(positive), 1)

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        loss_val, grads = loss_and_grad()

        if use_oplora:
            grads = self._project_grads(grads)

        # Manual SGD on trainable (LoRA) params.
        params = self.model.trainable_parameters()
        updated = _tree_sgd(params, grads, self.lr)
        self.model.update(updated)
        mx.eval(self.model.parameters(), loss_val)

        norm = _tree_l2(self.model.trainable_parameters())
        stats = TrainStats(
            loss=float(loss_val.item()),
            num_positive=len(positive),
            num_negative=len(negative),
            elapsed_s=time.perf_counter() - t0,
            adapter_norm=norm,
        )
        self.history.append(stats)
        self.candidates.clear()
        return stats

    def _project_grads(self, grads):
        """OPLoRA-project lora_a/lora_b grads per adapted LoRALinear."""
        from mlx_lm.tuner.lora import LoRALinear

        def walk(prefix, gnode, mnode):
            if isinstance(mnode, LoRALinear):
                W = mnode.linear.weight  # (out, in)
                key = prefix
                if key not in self._svd_cache:
                    self._svd_cache.update(
                        compute_svd_cache({key: W}, k=8))
                svd = self._svd_cache[key]
                # LoRALinear: lora_a (in, r), lora_b (r, out).
                # oplora expects A:(r,n=in), B:(m=out,r) → transpose views.
                dA = gnode["lora_a"].T
                dB = gnode["lora_b"].T
                dA_s, dB_s = project_lora_grads(
                    W, mnode.lora_a.T, mnode.lora_b.T, dA, dB,
                    {"U_k": svd["U_k"], "V_k": svd["V_k"]})
                gnode["lora_a"] = dA_s.T
                gnode["lora_b"] = dB_s.T
                return
            if isinstance(gnode, dict):
                for kk, gv in gnode.items():
                    mv = mnode[kk] if isinstance(mnode, (list, dict)) \
                        else getattr(mnode, kk, None)
                    if mv is not None:
                        walk(f"{prefix}.{kk}", gv, mv)
            elif isinstance(gnode, list):
                for i, gv in enumerate(gnode):
                    walk(f"{prefix}.{i}", gv, mnode[i])

        walk("model", grads, self.model)
        return grads


def _tree_sgd(params, grads, lr):
    if isinstance(params, dict):
        return {k: _tree_sgd(params[k], grads[k], lr) for k in grads}
    if isinstance(params, list):
        return [_tree_sgd(p, g, lr) for p, g in zip(params, grads)]
    return params - lr * grads


def _tree_l2(tree) -> float:
    if isinstance(tree, dict):
        return sum(_tree_l2(v) for v in tree.values())
    if isinstance(tree, list):
        return sum(_tree_l2(v) for v in tree)
    return float(mx.sum(tree.astype(mx.float32) ** 2).item())
