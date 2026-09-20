# Issue #9 - Remaining Issues to Fix

## From Reviewer Feedback:

### BLOCKER (spec criterion 5):
Tests do not meaningfully assert table contents or timeline ordering:
1. `test_catalog_comparison_no_custom_runs` ends in `pass` and asserts nothing
2. `test_catalog_timeline_chronological_order` asserts only `"20" in response_text` — no ordering check
3. `test_catalog_comparison_filters_standard_runs_only` never asserts the custom-run values are absent

### MAJOR (standards):
N+1 async DB call in `get_timeline_data()` — `get_all_models()` is awaited inside the per-session loop; hoist above the loop.

### MINOR:
1. `catalog_comparison.html` hardcodes short/long-context/burst workload metric keys; other standard workload types are aggregated in catalog.py but silently unrendered. Consider iterating `workload_metrics` keys.
2. Falsy-zero filtering — `if r.get('gen_tok_s')` drops a legitimate 0.0
3. Template timestamp slicing `event.timestamp[:10]` assumes ISO format
4. Duplicated `f"{name}_{quant}"` key construction (twice in `get_comparison_data`); a model name containing '_' could collide and merge rows

## Action Plan:

1. Fix tests with real assertions
2. Hoist `get_all_models()` out of session loop
3. Fix falsy-zero filtering issue
4. Make template more dynamic for workload types
5. Fix timestamp handling to not assume ISO format
6. Fix model key collision issue
