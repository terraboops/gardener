# The Gardener

A single-operator, local platform for cultivating long-running persistent
agents on **hypercar** (oMLX). Pi as the thin worker shell, Hermes Agent as
design inspiration, the Software Garden as doctrine.

> **Carve-out note:** this directory is intentionally self-contained. It lives
> inside the `omlx-mamba3` repo for now and will be extracted into its own
> repository later. Keep all Gardener code, docs, and tests under `gardener/`.
> The only outward dependency is hypercar's HTTP API surface
> (`/v1/chat/completions`, `/v1/sessions/*`) and, where reused unmodified, the
> `omlx.*` Python packages — both consumed as a service/library boundary, never
> by reaching back into the parent repo's internals.

## Status

Design complete. See [`docs/design.md`](docs/design.md). Next step:
implementation plan (superpowers `writing-plans`).

## What this IS / IS NOT

**IS:** persistent portfolio engineer · self-improving from experience ·
parallel subagent swarm · long-horizon autonomous tasks · an orchestration
layer for agentic pipelines · **bidirectional human↔agent work** (agents
delegate to humans as a first-class peer node, not just a review gate).

**IS NOT (v1):** not a cloud-model wrapper · not multi-user SaaS · not a
synchronous IDE copilot · not a from-scratch agent framework. Architecture
leaves distributed teams, multi-user, sync mode, and first-class human
co-editing open as future seams.

## Layout (planned)

```
gardener/
  docs/design.md        # the approved design spec
  README.md             # this file
  # (implementation dirs added during the plan: scheduler/ knowledge/
  #  journal/ harness/ learning/ daemon/ tests/)
```
