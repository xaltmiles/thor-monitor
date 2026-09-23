#!/usr/bin/env node
/**
 * Behavioral tests for the plots page (issue #30).
 *
 * Loads the HTML actually served by /plots into jsdom, maps the CDN script
 * tags to the exact library versions pinned in package.json (Chart.js 4.5.1,
 * chartjs-plugin-zoom 2.0.1, hammerjs 2.0.8, luxon 3.7.2, luxon adapter),
 * executes the template's real inline script, and drives it with real DOM
 * events: wheel zoom, shift+drag pan, drag zoom, button/dblclick reset,
 * simulated 30s refreshes with advancing data.
 *
 * Usage: node tests/js/run_plots_tests.mjs <served-plots.html> <scenario>
 * Exits 0 when the scenario passes; prints FAIL lines and exits 1 otherwise.
 * Run with scenario "all" to execute every scenario.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM, ResourceLoader, VirtualConsole } from 'jsdom';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const NODE_MODULES = path.join(repoRoot, 'node_modules');

const htmlFile = process.argv[2];
const only = process.argv[3] || 'all';
if (!htmlFile) {
  console.error('usage: run_plots_tests.mjs <served-plots.html> [scenario|all]');
  process.exit(2);
}
const html = readFileSync(htmlFile, 'utf8');

// Map the template's CDN URLs to the locally installed, version-pinned copies.
const CDN_TO_LOCAL = [
  [/chartjs-adapter-luxon@/, path.join(NODE_MODULES, 'chartjs-adapter-luxon/dist/chartjs-adapter-luxon.umd.min.js')],
  [/chartjs-plugin-zoom@/, path.join(NODE_MODULES, 'chartjs-plugin-zoom/dist/chartjs-plugin-zoom.min.js')],
  [/chart\.js@/, path.join(NODE_MODULES, 'chart.js/dist/chart.umd.js')],
  [/hammerjs@/, path.join(NODE_MODULES, 'hammerjs/hammer.min.js')],
  [/luxon@/, path.join(NODE_MODULES, 'luxon/build/global/luxon.min.js')],
  [/htmx\.org@/, path.join(NODE_MODULES, 'htmx.org/dist/htmx.min.js')],
];

class LocalScripts extends ResourceLoader {
  fetch(url) {
    for (const [pattern, file] of CDN_TO_LOCAL) {
      if (pattern.test(String(url))) {
        return Promise.resolve(readFileSync(file, 'utf8'));
      }
    }
    return Promise.resolve(''); // htmx (and anything else): not needed by the chart scenarios
  }
}

// ---------------------------------------------------------------------------
// Fake telemetry API
// ---------------------------------------------------------------------------

const T0 = Date.UTC(2024, 5, 1, 10, 0, 0); // 2024-06-01T10:00:00Z
const STEP_MS = 30_000; // app samples every 30s

function makeRow(t) {
  return {
    timestamp: new Date(t).toISOString(),
    gpu_util: 40 + (Math.floor(t / STEP_MS) % 5),
    gpu_temp: 65,
    gpu_power: 120,
    memory_used: 8e9,
    memory_total: 32e9,
    llama_stats: JSON.stringify({ generated_tokens_rate: 30, prompt_tokens_rate: 900 }),
    ollama_stats: JSON.stringify({ ollama_generated_tokens_rate: 25, ollama_prompt_tokens_rate: 800 }),
  };
}

/** Oldest-first dataset: `count` samples every STEP_MS starting at startMs. */
function makeDataset(startMs, count) {
  const rows = [];
  for (let i = 0; i < count; i++) rows.push(makeRow(startMs + i * STEP_MS));
  return rows;
}

const initialDataset = makeDataset(T0, 40); // 10:00:00 .. 10:19:30

/**
 * The stub mimics /api/plots/history: newest-first payload, optional
 * `since` cursor (>= semantics, cursor sample included).
 */
function makeServer() {
  return {
    calls: [],
    datasetsByLimit: {}, // limit -> oldest-first rows (default: initialDataset)
    dataset: initialDataset,
    respond(url) {
      const u = new URL(url, 'http://localhost');
      const limit = parseInt(u.searchParams.get('limit') ?? '3600', 10);
      this.calls.push({ url: url, limit: limit, since: u.searchParams.get('since') });
      const full = this.datasetsByLimit[limit] ?? this.dataset;
      const since = u.searchParams.get('since');
      const rows = since ? full.filter((r) => r.timestamp >= since) : full;
      return rows.slice().reverse(); // API returns newest-first
    },
  };
}

// ---------------------------------------------------------------------------
// jsdom page boot
// ---------------------------------------------------------------------------

async function bootPage(server, scriptErrors) {
  const virtualConsole = new VirtualConsole();
  virtualConsole.on('jsdomError', (e) => scriptErrors.push(String(e.detail?.stack || e.stack || e)));
  virtualConsole.on('error', (...a) => scriptErrors.push(a.map(String).join(' ')));

  const dom = new JSDOM(html, {
    url: 'http://localhost/plots',
    runScripts: 'dangerously',
    resources: new LocalScripts(),
    pretendToBeVisual: true,
    virtualConsole,
  });
  const { window } = dom;

  // jsdom has no layout engine: give charts a real working area so
  // Chart.js builds a non-degenerate chartArea and the zoom plugin's
  // focal-point math behaves like a browser.
  Object.defineProperty(window.Element.prototype, 'clientWidth', { get: () => 800, configurable: true });
  Object.defineProperty(window.Element.prototype, 'clientHeight', { get: () => 400, configurable: true });
  window.Element.prototype.getBoundingClientRect = function () {
    return { left: 0, top: 0, right: 800, bottom: 400, width: 800, height: 400, x: 0, y: 0 };
  };
  window.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  if (typeof window.devicePixelRatio === 'undefined') window.devicePixelRatio = 1;
  // jsdom's MouseEvent.which is 0, but hammer.js's MouseInput converts any
  // move with `ev.which !== 1` into an end-of-gesture, killing every pan.
  // Mirror the browser convention: which = button + 1.
  Object.defineProperty(window.MouseEvent.prototype, 'which', {
    get() { return (this.button || 0) + 1; },
    configurable: true,
  });
  window.fetch = (url) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(server.respond(String(url))) });

  // The page's inline script declared its state with top-level const/let,
  // which lands in the global lexical environment shared by later scripts
  // (and by window.eval), so the harness can observe it without modifying
  // the template.
  await waitFor(() => window.eval('typeof charts !== "undefined" && Object.keys(charts).length === 6'));
  await waitFor(() => window.eval('Object.values(charts).every((c) => c && c.scales && c.scales.x)'));
  await settle(window);

  return { dom, window, scriptErrors, server };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Two animation frames, so Chart.js rAF-driven rendering completes. */
async function settle(window) {
  await new Promise((r) => window.requestAnimationFrame(() => window.requestAnimationFrame(r)));
  await sleep(10);
}

async function waitFor(fn, timeoutMs = 5000, stepMs = 20) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    let ok = false;
    try { ok = fn(); } catch { ok = false; }
    if (ok) return;
    if (Date.now() > deadline) throw new Error('waitFor timed out');
    await sleep(stepMs);
  }
}

// ---------------------------------------------------------------------------
// Observation + event helpers (all run inside the page realm via window.eval)
// ---------------------------------------------------------------------------

const evalIn = (window, expr) => window.eval(expr);

function xRange(window, name) {
  return evalIn(window, `(function () { const s = charts['${name}'].scales.x; return { min: s.min, max: s.max }; })()`);
}

function yRange(window, name) {
  return evalIn(window, `(function () { const s = charts['${name}'].scales.y; return { min: s.min, max: s.max }; })()`);
}

const CHART_NAMES = ['memory', 'gpuTemp', 'llamaTokS', 'ollamaTokS', 'gpuUtil', 'gpuPower'];

function allXRanges(window) {
  const out = {};
  for (const n of CHART_NAMES) out[n] = xRange(window, n);
  return out;
}

function sharedState(window) {
  return evalIn(window, `({
    sharedTimeRange: { min: sharedTimeRange.min, max: sharedTimeRange.max },
    lastZoomRange: lastZoomRange === null ? null : { min: lastZoomRange.min, max: lastZoomRange.max },
    isResetting: (typeof isResetting === 'undefined') ? null : isResetting,
  })`);
}

const dataBounds = (rows) => ({ min: Date.parse(rows[0].timestamp), max: Date.parse(rows[rows.length - 1].timestamp) });

function fireWheel(window, canvas, deltaY) {
  canvas.dispatchEvent(new window.WheelEvent('wheel', {
    deltaY, clientX: 400, clientY: 200, cancelable: true, bubbles: true,
  }));
}

function fireMouse(window, target, type, x, y, extra = {}) {
  target.dispatchEvent(new window.MouseEvent(type, {
    bubbles: true, cancelable: true, view: window,
    clientX: x, clientY: y, button: 0, buttons: type === 'mouseup' ? 0 : 1, ...extra,
  }));
}

// The plugin debounces onZoomComplete by 250ms on the wheel path.
const DEBOUNCE = 400;

/** Zoom one chart with the mouse wheel and wait for the sync pass. */
async function wheelZoom(window, sourceName, wheels = 2) {
  const canvas = evalIn(window, `charts['${sourceName}'].canvas`);
  for (let i = 0; i < wheels; i++) fireWheel(window, canvas, -120);
  await sleep(DEBOUNCE);
  await settle(window);
}

/** Shift+drag pan on one chart (pan.modifierKey is 'shift'). */
async function shiftDragPan(window, sourceName, dx = 90) {
  const canvas = evalIn(window, `charts['${sourceName}'].canvas`);
  fireMouse(window, canvas, 'mousedown', 400, 200, { shiftKey: true });
  for (let i = 1; i <= 6; i++) fireMouse(window, canvas, 'mousemove', 400 + (dx * i) / 6, 200, { shiftKey: true });
  fireMouse(window, window.document, 'mouseup', 400 + dx, 200, { shiftKey: true });
  await sleep(150); // hammer panend -> onPanComplete -> sync pass
  await settle(window);
}

/**
 * Plain drag (no modifier) = drag-zoom on one chart. The plugin binds
 * mousemove to the canvas itself (and mouseup to the document), while
 * hammer's pan recognizer listens on the document — dispatching moves on
 * the canvas reaches both (events bubble).
 */
async function dragZoom(window, sourceName) {
  const canvas = evalIn(window, `charts['${sourceName}'].canvas`);
  fireMouse(window, canvas, 'mousedown', 200, 200);
  for (let i = 1; i <= 6; i++) fireMouse(window, canvas, 'mousemove', 200 + (300 * i) / 6, 200);
  fireMouse(window, window.document, 'mouseup', 500, 200);
  await sleep(150);
  await settle(window);
}

/** Advance the server dataset and drive the same incremental refresh the page's 30s timer runs. */
async function incrementalRefresh(window, server, newRows) {
  server.dataset = server.dataset.concat(newRows);
  await evalIn(window, 'loadCharts(currentLimit, lastTimestamp !== null)');
  await settle(window);
}

async function fullRefreshViaRangeButton(window, server, limit, dataset) {
  server.datasetsByLimit[limit] = dataset;
  const btn = window.document.querySelector(`.range-btn[data-limit="${limit}"]`);
  if (!btn) throw new Error(`range button data-limit="${limit}" not found`);
  btn.click();
  await waitFor(() => evalIn(window, 'Object.keys(charts).length === 6'));
  await settle(window);
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

const scenarios = {};

// Sanity: the real inline script built six real charts under the real libs.
scenarios.charts_build = async ({ window, server }) => {
  const names = evalIn(window, 'Object.keys(charts)');
  check(Array.isArray(names) && names.length === 6, `expected 6 charts, got ${JSON.stringify(names)}`);
  check(names.every((n) => evalIn(window, `charts['${n}'] instanceof Chart`)), 'charts are not real Chart instances');
  for (const n of CHART_NAMES) {
    const r = xRange(window, n);
    check(r.max > r.min, `${n}: x range collapsed (${r.min}..${r.max})`);
  }
  // x axis is a shared time axis fitted to the served data
  const b = dataBounds(server.dataset);
  const r = xRange(window, 'memory');
  check(r.min <= b.min + 1000 && r.max >= b.max - 1000, `x not fitted to data: ${JSON.stringify(r)} vs ${JSON.stringify(b)}`);
};

// AC1+AC2: wheel zoom on one chart narrows the window and every other chart
// follows; the value axis must not move (zoom mode is confined to x).
scenarios.wheel_zoom_syncs_all = async ({ window }) => {
  const before = allXRanges(window);
  const yBefore = yRange(window, 'memory');
  await wheelZoom(window, 'memory', 2);
  const after = allXRanges(window);

  const src = after.memory;
  check(src.max - src.min < before.memory.max - before.memory.min, 'wheel zoom did not narrow the window');
  for (const n of CHART_NAMES) {
    check(Math.abs(after[n].min - src.min) < 1 && Math.abs(after[n].max - src.max) < 1,
      `${n} not synchronized after wheel zoom: ${JSON.stringify(after[n])} vs ${JSON.stringify(src)}`);
  }
  const yAfter = yRange(window, 'memory');
  check(Math.abs(yAfter.min - yBefore.min) < 1e-6 && Math.abs(yAfter.max - yBefore.max) < 1e-6,
    `y axis moved on wheel zoom (mode must be x): ${JSON.stringify(yBefore)} -> ${JSON.stringify(yAfter)}`);
  check(evalIn(window, 'charts.memory.options.plugins.zoom.zoom.mode') === 'x',
    'real chart zoom mode is not x');
};

// AC1+AC2: shift+drag pans the source chart and the window is propagated,
// unchanged in width, to every chart.
scenarios.shift_drag_pan_syncs_all = async ({ window }) => {
  await wheelZoom(window, 'memory', 1); // start from a zoomed window
  const before = allXRanges(window);
  await shiftDragPan(window, 'gpuTemp', 90);
  const after = allXRanges(window);

  const wBefore = before.gpuTemp.max - before.gpuTemp.min;
  const wAfter = after.gpuTemp.max - after.gpuTemp.min;
  check(Math.abs(wAfter - wBefore) < 2, `pan changed the window width: ${wBefore} -> ${wAfter}`);
  check(after.gpuTemp.min !== before.gpuTemp.min, 'shift+drag did not pan the source chart');
  for (const n of CHART_NAMES) {
    check(Math.abs(after[n].min - after.gpuTemp.min) < 1 && Math.abs(after[n].max - after.gpuTemp.max) < 1,
      `${n} not synchronized after pan: ${JSON.stringify(after[n])} vs ${JSON.stringify(after.gpuTemp)}`);
  }
};

// AC1 (drag-zoom gesture) + AC2: a plain drag zooms the source chart and
// syncs all charts to the selected window.
scenarios.drag_zoom_syncs_all = async ({ window }) => {
  const before = allXRanges(window);
  await dragZoom(window, 'llamaTokS');
  const after = allXRanges(window);
  check(after.llamaTokS.max - after.llamaTokS.min < before.llamaTokS.max - before.llamaTokS.min,
    `drag zoom did not narrow the window: ${JSON.stringify(before.llamaTokS)} -> ${JSON.stringify(after.llamaTokS)}`);
  for (const n of CHART_NAMES) {
    check(Math.abs(after[n].min - after.llamaTokS.min) < 1 && Math.abs(after[n].max - after.llamaTokS.max) < 1,
      `${n} not synchronized after drag zoom: ${JSON.stringify(after[n])}`);
  }
};

// AC3: visible Reset Zoom button resets every chart, and double-click also
// resets — with the shared zoom state actually cleared, not resurrected.
scenarios.reset_button_and_dblclick = async ({ window }) => {
  const btn = window.document.getElementById('reset-zoom-btn');
  check(btn !== null, 'Reset Zoom button missing');
  check((btn.textContent || '').toLowerCase().includes('reset'), 'Reset Zoom button has no visible label');

  await wheelZoom(window, 'gpuUtil', 2);
  const zoomed = allXRanges(window);
  check(zoomed.memory.max - zoomed.memory.min < 20 * 60 * 1000, 'setup: charts did not zoom');

  btn.click();
  await settle(window);
  const reset = allXRanges(window);
  const b = dataBounds(initialDataset);
  for (const n of CHART_NAMES) {
    check(reset[n].min <= b.min + 1000 && reset[n].max >= b.max - 1000,
      `${n} not reset to full data: ${JSON.stringify(reset[n])} vs ${JSON.stringify(b)}`);
  }
  const st = sharedState(window);
  check(st.sharedTimeRange.min === null && st.sharedTimeRange.max === null,
    `sharedTimeRange not cleared after reset: ${JSON.stringify(st.sharedTimeRange)}`);
  check(st.lastZoomRange === null, `lastZoomRange not cleared after reset: ${JSON.stringify(st.lastZoomRange)}`);
  check(st.isResetting === false, 'isResetting stuck true after reset');

  // double-click resets too
  await wheelZoom(window, 'memory', 2);
  const canvas = evalIn(window, `charts['ollamaTokS'].canvas`);
  canvas.dispatchEvent(new window.MouseEvent('dblclick', { bubbles: true, cancelable: true, clientX: 300, clientY: 200 }));
  await settle(window);
  const reset2 = allXRanges(window);
  for (const n of CHART_NAMES) {
    check(reset2[n].min <= b.min + 1000 && reset2[n].max >= b.max - 1000,
      `${n} not reset by dblclick: ${JSON.stringify(reset2[n])}`);
  }
};

// AC4: the zoomed window survives an incremental refresh (clamped to the new
// data bounds) and is clamped (not blindly reapplied) after a range switch.
scenarios.zoom_survives_refresh_and_range_switch = async ({ window, server }) => {
  await wheelZoom(window, 'gpuPower', 1);
  const w = allXRanges(window).gpuPower;

  // simulated 30s incremental refresh with newer data appended
  await incrementalRefresh(window, server, makeDataset(T0 + 40 * STEP_MS, 10));
  const b2 = dataBounds(server.dataset);
  const expect = { min: Math.max(w.min, b2.min), max: Math.min(w.max, b2.max) };
  for (const n of CHART_NAMES) {
    const r = xRange(window, n);
    check(Math.abs(r.min - expect.min) < 1 && Math.abs(r.max - expect.max) < 1,
      `${n}: zoom window not preserved across refresh: ${JSON.stringify(r)} vs ${JSON.stringify(expect)}`);
  }

  // range-button switch to a dataset that only partially overlaps the saved
  // window: the applied window must be clamped to the newly fetched bounds.
  const altDataset = makeDataset(T0 - 20 * 60_000, 60); // 09:40:00 .. 10:09:30
  await fullRefreshViaRangeButton(window, server, 60, altDataset);
  const b3 = dataBounds(altDataset);
  const expect3 = { min: Math.max(w.min, b3.min), max: Math.min(w.max, b3.max) };
  check(expect3.min < expect3.max, 'setup: expected clamped window to be non-empty');
  for (const n of CHART_NAMES) {
    const r = xRange(window, n);
    check(Math.abs(r.min - expect3.min) < 1 && Math.abs(r.max - expect3.max) < 1,
      `${n}: stale window not clamped on range switch: ${JSON.stringify(r)} vs ${JSON.stringify(expect3)}`);
  }
};

// Regression for the reset-resurrection bug: after ANY reset the dashboard
// must keep tracking new telemetry. With the bug, each chart.resetZoom()
// fired onZoomComplete -> synchronizedZoom synchronously, re-populating
// sharedTimeRange with the full-data bounds; the next refresh saved that to
// lastZoomRange and pinned x.max at the reset-time data edge forever, so new
// samples never scrolled in until a full page reload.
scenarios.reset_does_not_freeze_tracking = async ({ window, server }) => {
  await wheelZoom(window, 'memory', 2);
  window.document.getElementById('reset-zoom-btn').click();
  await settle(window);

  const st = sharedState(window);
  check(st.sharedTimeRange.min === null && st.sharedTimeRange.max === null,
    `sharedTimeRange resurrected after reset: ${JSON.stringify(st.sharedTimeRange)}`);
  check(st.lastZoomRange === null, `lastZoomRange resurrected after reset: ${JSON.stringify(st.lastZoomRange)}`);

  const resetMax = xRange(window, 'memory').max;

  // two simulated 30s refreshes; data advances past the reset-time edge
  await incrementalRefresh(window, server, makeDataset(T0 + 40 * STEP_MS, 10)); // newest 10:24:30
  await incrementalRefresh(window, server, makeDataset(T0 + 50 * STEP_MS, 10)); // newest 10:29:30
  const newest = Date.parse(server.dataset[server.dataset.length - 1].timestamp);

  for (const n of CHART_NAMES) {
    const r = xRange(window, n);
    check(r.max >= newest - 1000,
      `${n}: frozen at reset-time window after refresh (new data invisible): x.max=${new Date(r.max).toISOString()}, newest=${new Date(newest).toISOString()}`);
  }
  const afterMax = xRange(window, 'memory').max;
  check(afterMax > resetMax + 60_000,
    `x.max did not advance with new data: ${new Date(resetMax).toISOString()} -> ${new Date(afterMax).toISOString()}`);

  // and zooming still works after reset+refresh (state machine not wedged)
  const before = allXRanges(window);
  await wheelZoom(window, 'gpuUtil', 2);
  const after = allXRanges(window);
  check(after.gpuUtil.max - after.gpuUtil.min < before.gpuUtil.max - before.gpuUtil.min,
    'zoom broken after reset + refresh');
};

// AC5: tooltips stay usable while zoomed — the real constructed chart still
// has the index/no-intersect tooltip config and a real mousemove activates it.
scenarios.tooltip_works_while_zoomed = async ({ window }) => {
  await wheelZoom(window, 'memory', 2);
  const opts = evalIn(window, `({
    mode: charts.memory.options.plugins.tooltip.mode,
    intersect: charts.memory.options.plugins.tooltip.intersect,
    interactionMode: charts.memory.options.interaction.mode,
    interactionIntersect: charts.memory.options.interaction.intersect,
  })`);
  check(opts.mode === 'index' && opts.intersect === false, `tooltip config lost: ${JSON.stringify(opts)}`);
  check(opts.interactionMode === 'index' && opts.interactionIntersect === false,
    `interaction config lost: ${JSON.stringify(opts)}`);

  const canvas = evalIn(window, 'charts.memory.canvas');
  fireMouse(window, canvas, 'mousemove', 400, 200);
  await settle(window);
  const active = evalIn(window, 'charts.memory.tooltip.getActiveElements().length');
  check(active > 0, `tooltip did not activate over zoomed chart (active=${active})`);
};

// ---------------------------------------------------------------------------

function check(cond, msg) {
  if (!cond) throw new Error(msg);
}

const names = Object.keys(scenarios).filter((n) => only === 'all' || n === only);
if (names.length === 0) {
  console.error(`unknown scenario: ${only}`);
  console.error(`available: ${Object.keys(scenarios).join(', ')}`);
  process.exit(2);
}

let failed = 0;
for (const name of names) {
  const server = makeServer();
  const scriptErrors = [];
  try {
    const page = await bootPage(server, scriptErrors);
    try {
      await scenarios[name](page);
      console.log(`PASS ${name}`);
    } finally {
      page.dom.window.close();
    }
  } catch (e) {
    failed++;
    console.log(`FAIL ${name}: ${e && e.message}`);
    for (const err of scriptErrors) console.log(`  script error: ${err.split('\n').slice(0, 4).join('\n  ')}`);
  }
}
process.exit(failed ? 1 : 0);
