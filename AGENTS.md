# Agent Skills

## Smoke test before closing

Closing evidence for any ticket is `make smoke` plus the unit suite: the smoke test starts the real app, drives real HTTP and the real SQLite store, and fails on any break that fixtures mask (unwired sources, broken imports, dead real-system paths). A ticket whose only evidence is pytest is not done.

## Issue tracker

Issues and specs for this repo live as GitHub issues at `github.com:xaltmiles/thor-monitor`, accessed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

## Triage labels

Five canonical triage labels mapped directly to their role names: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

## Domain docs

Single-context layout: one `CONTEXT.md` at the repo root, ADRs in `docs/adr/`. See `docs/agents/domain.md`.
