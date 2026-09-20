# Issue #9 Implementation Notes

**Parent Spec**: #1 (thor-monitor v1)

**What to build**:
1. Catalog view with comparison table (per-model, standard runs only)
2. Timeline view (runs + sessions chronologically)

**Acceptance criteria**:
- [ ] Comparison table: one row per model (name, quant, context), best/avg tok/s per workload, memory footprint
- [ ] Table filters to standard runs by default; custom runs excluded
- [ ] Timeline view shows runs and Sessions chronologically with type distinction
- [ ] Views render server-side with progressive enhancement; no client build step
- [ ] Tests assert table contents and timeline ordering from seeded store data

**Blocked by**: T4 (Passive tok/s and Sessions), T6 (Complete suite)

**Key domain terms**:
- **Catalog**: Accumulated history of Benchmark Runs and Sessions per model, stored in SQLite
- **Benchmark Run**: Tool-initiated workload with standard/custom tag
- **Session**: Period of real usage observed passively, recorded with observed tok/s stats
