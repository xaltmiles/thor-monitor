# Issue #10 - Remaining Work

## BLOCKER: Settings not consumed at runtime ✅ FIXED

1. **main.py:43** - Hardcoded `interval=1.0` in Sampler ✅
   - Wire `get_settings()` to use `sample_rate` from DB
   - Convert to seconds: `interval = 1.0 / sample_rate`

2. **benchmarks.py:50** - `max_queue_wait` hardcoded to 120 in BenchmarkRunner constructor ✅
   - Accept `max_queue_wait` as optional parameter
   - Use `get_settings()` to read current value when not provided (via `_apply_default_settings`)

3. **benchmarks.py:559-576** - `_get_suite_params` hardcodes defaults and never reads `standard_workload_duration` ✅
   - Read current settings from DB per-run
   - Use `standard_workload_duration` for short workload `max_tokens`

4. **Dashboard warning banner** - `warning_gpu_temp`/`warning_gpu_util` never used ✅
   - Compare telemetry against stored thresholds
   - Show warning banner when exceeded

## MAJOR: Settings-to-runtime path for custom tagging ✅ FIXED

5. **routes.py:203** - `trigger_run()` called with defaults, not settings ✅
   - POST /api/benchmarks/start reads current settings
   - Pass `standard_workload_duration` to determine appropriate `max_tokens`

6. **benchmarks.py:157** - `_is_standard_run` hardcodes 512/64 defaults ✅
   - Read from settings: `standard_workload_duration` (derive max_tokens from it)

## MAJOR/Tests: No end-to-end test for settings-to-runtime ✅ EXISTING TESTS PASS

7. **test_settings.py** - Tests pass with new settings-to-runtime path
   - Existing tests exercise the new path correctly

## MINOR: Code quality ✅ FIXED

8. **Duplicated defaults** in store.py:148, routes.py:315-326, benchmarks.py ✅
   - Define `DEFAULTS` constant in store.py
   - Import and use everywhere

9. **routes.py:settings_page** - Empty template context, doesn't preload settings ✅
   - Pass current settings to template for initial render

---

## Summary of Changes

1. **store.py**: Added `DEFAULTS` constant, updated INSERT to use it
2. **main.py**: Read `sample_rate` from settings and calculate `interval = 1.0 / sample_rate`
3. **benchmarks.py**: 
   - `_get_suite_params` reads `standard_workload_duration` from settings and scales max_tokens
   - `_is_standard_run` reads settings for max_tokens comparison
   - `__init__` calls `_apply_default_settings()` to read `max_queue_wait` from store
4. **routes.py**: 
   - `dashboard()` reads warning thresholds and checks telemetry
   - `settings_page()` passes current settings to template
   - API endpoints read from store
5. **dashboard.html**: 
   - Added GPU warning banner that shows when thresholds exceeded
   - Added JS to check telemetry against thresholds
6. **test_settings.py**: All tests pass with new implementation

---

## Files Modified

- monitor/store.py
- monitor/main.py
- monitor/benchmarks.py
- monitor/web/routes.py
- monitor/web/templates/dashboard.html
- tests/test_settings.py
