import { drawLineChart, PADDING, type Series } from '../shared/chart';
import {
  BOTTOM, DEFAULT_ALPHA, DEFAULT_GAMMA, DEFAULT_WEIGHTS, DIVERGENCE_LIMIT, REWARD, STATES, WEIGHTS,
  features, hasDiverged, stateValues, update, type UpdateMode, type UpdateRecord,
} from './core';

const MAX_POINTS = 400;
const MAX_STEPS_PER_FRAME = 200;
const MINUS = '−';
const SUB = '₀₁₂₃₄₅₆';

/** Compact number for arithmetic: up to 4 decimals, trailing zeros dropped. */
function num(v: number): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? '∞' : `${MINUS}∞`;
  const a = Math.abs(v);
  const body = a >= 1e5 ? a.toExponential(3).replace('e+', 'e') : String(Number(a.toFixed(4)));
  return (v < 0 && body !== '0' ? MINUS : '') + body;
}

/** Fixed 3-decimal number for table columns. */
function fixed(v: number): string {
  if (!Number.isFinite(v) || Math.abs(v) >= 1e5) return num(v);
  const body = Math.abs(v).toFixed(3);
  return (v < 0 && Number(body) !== 0 ? MINUS : '') + body;
}

/** Short label that fits inside a state circle: 3 decimals, or 3 significant digits once large. */
function short(v: number): string {
  if (!Number.isFinite(v) || Math.abs(v) < 1000) return num(Number(v.toFixed(3)));
  return (v < 0 ? MINUS : '') + Math.abs(v).toExponential(2).replace('e+', 'e');
}

/** Wraps negatives in parentheses so "0.95·(−2)" never reads as subtraction. */
const paren = (v: number) => (v < 0 ? `(${num(v)})` : num(v));

const axisLabel = (v: number) => {
  const a = Math.abs(v);
  if (a >= 1000) return (v < 0 ? MINUS : '') + a.toExponential(1).replace('e+', 'e');
  return (v < 0 && a >= 0.05 ? MINUS : '') + String(Number(a.toFixed(a >= 10 ? 0 : 1)));
};

/** Signed log: keeps sign and zero, so oscillating weights still plot on a log-like axis. */
const symlog = (v: number) => Math.sign(v) * Math.log10(1 + Math.abs(v));
const unsymlog = (y: number) => Math.sign(y) * (10 ** Math.abs(y) - 1);

/** Even samples of `length` indices, always including the first and last. */
function sampleIndices(length: number): number[] {
  if (length <= MAX_POINTS) return Array.from({ length }, (_, i) => i);
  return Array.from({ length: MAX_POINTS }, (_, i) => Math.round((i * (length - 1)) / (MAX_POINTS - 1)));
}

/** 0..6 bucket of |v| on a log scale, used to pick a sequential fill step. */
function magnitudeStep(v: number): number {
  if (!Number.isFinite(v)) return 6;
  return Math.round(Math.min(1, Math.log10(1 + Math.abs(v)) / Math.log10(DIVERGENCE_LIMIT)) * 6);
}

function formula(s: number): string {
  return s === BOTTOM ? `2w${SUB[0]} + w${SUB[6]}` : `w${SUB[0]} + 2w${SUB[s]}`;
}

/** "w₀ + 2·w₁ = 1 + 2·1 = 3" — how v̂(s) was computed from these weights. */
function valueArithmetic(s: number, w: readonly number[], v: number): string {
  const terms = features(s)
    .map((x, j) => (x === 0 ? '' : `${x === 1 ? '' : `${x}·`}${paren(w[j])}`))
    .filter(Boolean);
  return `v̂(s${s}) = ${formula(s)} = ${terms.join(' + ')} = ${num(v)}`;
}

/** Mirrors the parent Starlight page's light/dark toggle when embedded. */
function syncTheme(onChange: () => void) {
  try {
    if (window.parent === window) return;
    const parentRoot = window.parent.document.documentElement;
    const apply = () => {
      const theme = parentRoot.dataset.theme;
      if (theme) document.documentElement.dataset.theme = theme;
      onChange();
    };
    apply();
    new MutationObserver(apply).observe(parentRoot, { attributes: true, attributeFilter: ['data-theme'] });
  } catch {
    // Cross-origin parent: fall back to prefers-color-scheme.
  }
}

export function initBaird() {
  const root = document.querySelector<HTMLElement>('[data-baird]');
  if (!root) return;
  const $ = <T extends Element>(selector: string) => root.querySelector<T>(selector)!;

  const alphaInput = $<HTMLInputElement>('[data-alpha]');
  const gammaInput = $<HTMLInputElement>('[data-gamma]');
  const weightInputs = [...root.querySelectorAll<HTMLInputElement>('[data-w0]')];
  const modeButtons = [...root.querySelectorAll<HTMLButtonElement>('[data-mode]')];
  const stepButton = $<HTMLButtonElement>('[data-step]');
  const step10Button = $<HTMLButtonElement>('[data-step10]');
  const playButton = $<HTMLButtonElement>('[data-play]');
  const speedInput = $<HTMLInputElement>('[data-speed]');
  const speedLabel = $<HTMLOutputElement>('[data-speed-value]');
  const scaleButtons = [...root.querySelectorAll<HTMLButtonElement>('[data-scale]')];
  const iterationLabel = $<HTMLElement>('[data-iteration]');
  const badge = $<HTMLElement>('[data-diverged]');
  const status = $<HTMLElement>('[data-status]');
  const weightList = $<HTMLElement>('[data-weight-list]');
  const breakdown = $<HTMLElement>('[data-breakdown]');
  const weightCanvas = $<HTMLCanvasElement>('[data-weight-chart]');
  const valueCanvas = $<HTMLCanvasElement>('[data-value-chart]');
  const weightReadout = $<HTMLElement>('[data-weight-readout]');
  const valueReadout = $<HTMLElement>('[data-value-readout]');

  alphaInput.value = String(DEFAULT_ALPHA);
  gammaInput.value = String(DEFAULT_GAMMA);
  weightInputs.forEach((input, j) => { input.value = String(DEFAULT_WEIGHTS[j]); });

  let mode: UpdateMode = 'batch';
  let logScale = false;
  let alpha = DEFAULT_ALPHA;
  let gamma = DEFAULT_GAMMA;
  let initial = [...DEFAULT_WEIGHTS];
  let w = [...initial];
  let history: number[][] = [[...w]];
  let last: UpdateRecord | null = null;
  let diverged = false;
  let playing = false;
  let frame = 0;
  let hover: { canvas: HTMLCanvasElement; iteration: number } | null = null;

  /** Reads a numeric input, flagging it and keeping `fallback` when it is not a number. */
  function readNumber(input: HTMLInputElement, fallback: number, ok: (v: number) => boolean = () => true): number {
    const v = Number(input.value);
    const valid = input.value.trim() !== '' && Number.isFinite(v) && ok(v);
    input.setAttribute('aria-invalid', String(!valid));
    return valid ? v : fallback;
  }

  function reset() {
    pause();
    alpha = readNumber(alphaInput, alpha, v => v >= 0);
    gamma = readNumber(gammaInput, gamma, v => v >= 0 && v <= 1);
    initial = weightInputs.map((input, j) => readNumber(input, initial[j]));
    w = [...initial];
    history = [[...w]];
    last = null;
    diverged = hasDiverged(w);
    render();
  }

  function advance(count: number) {
    for (let i = 0; i < count && !diverged; i += 1) {
      last = update(mode, w, alpha, gamma);
      w = last.after;
      history.push([...w]);
      diverged = hasDiverged(w);
    }
    if (diverged) pause();
  }

  function play() {
    if (diverged) return;
    playing = true;
    let previous = performance.now();
    let owed = 0;
    const tick = (now: number) => {
      owed += ((now - previous) / 1000) * Number(speedInput.value);
      previous = now;
      const whole = Math.min(MAX_STEPS_PER_FRAME, Math.floor(owed));
      owed -= whole;
      if (whole > 0) { advance(whole); render(); }
      if (playing) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    renderControls();
  }

  function pause() {
    playing = false;
    cancelAnimationFrame(frame);
    renderControls();
  }

  function cssVar(name: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /** Colour for weight j (or, with the same index, state j): w0 is shared, so it wears neutral ink. */
  const seriesColor = (j: number) => cssVar(j === 0 ? '--series-shared' : `--series-${j}`);

  function renderControls() {
    stepButton.disabled = diverged;
    step10Button.disabled = diverged;
    playButton.disabled = diverged;
    playButton.textContent = playing ? 'Pause' : 'Play';
    stepButton.textContent = mode === 'batch' ? 'Step (1 batch)' : 'Step (1 sweep)';
    playButton.setAttribute('aria-pressed', String(playing));
    speedLabel.value = `${speedInput.value}/s`;
    for (const b of modeButtons) b.setAttribute('aria-pressed', String(b.dataset.mode === mode));
    for (const b of scaleButtons) b.setAttribute('aria-pressed', String((b.dataset.scale === 'log') === logScale));
  }

  function renderDiagram(values: readonly number[]) {
    for (let s = 1; s <= STATES; s += 1) {
      const node = $<SVGGElement>(`[data-node="${s}"]`);
      const v = values[s - 1];
      node.dataset.k = String(magnitudeStep(v));
      node.querySelector('[data-node-value]')!.textContent = short(v);
      node.querySelector('title')!.textContent = valueArithmetic(s, w, v);
    }
  }

  function renderWeights() {
    weightList.replaceChildren(...w.map((wj, j) => {
      const item = document.createElement('div');
      const name = document.createElement('dt');
      const cell = document.createElement('dd');
      name.textContent = `w${SUB[j]}`;
      cell.textContent = fixed(wj);
      const change = history.length > 1 ? wj - history[history.length - 2][j] : 0;
      cell.dataset.trend = change > 0 ? 'up' : change < 0 ? 'down' : '';
      item.append(name, cell);
      return item;
    }));
  }

  function renderBreakdown() {
    if (!last) {
      breakdown.innerHTML = '<p class="empty">Press <strong>Step</strong> to run one sweep over the six transitions. Every row of the arithmetic appears here.</p>';
      return;
    }
    const { alpha: a, gamma: g, steps, deltaW, before, after } = last;
    const head = `<tr><th scope="col">s → s′</th><th scope="col">v̂(s)</th><th scope="col">v̂(s′)</th><th scope="col" class="delta">δ = R + γ·v̂(s′) − v̂(s)</th>${
      Array.from({ length: WEIGHTS }, (_, j) => `<th scope="col">Δw${SUB[j]}</th>`).join('')}</tr>`;
    const rows = steps.map(st => {
      const x = features(st.from);
      const cells = st.contribution.map((c, j) => x[j] === 0
        ? '<td class="zero">·</td>'
        : `<td title="α·δ·x${SUB[j]}(s) = ${num(a)}·${paren(st.delta)}·${x[j]} = ${num(c)}">${fixed(c)}</td>`).join('');
      return `<tr><th scope="row">s${st.from} → s${st.to}</th><td>${fixed(st.vFrom)}</td><td>${fixed(st.vTo)}</td>`
        + `<td class="delta"><code>δ = ${num(REWARD)} + ${num(g)}·${paren(st.vTo)} − ${paren(st.vFrom)} = ${num(st.delta)}</code></td>${cells}</tr>`;
    }).join('');
    const footRow = (label: string, values: readonly number[], cls = '') =>
      `<tr class="${cls}"><th scope="row" colspan="4">${label}</th>${values.map(v => `<td>${fixed(v)}</td>`).join('')}</tr>`;
    const order = last.mode === 'batch'
      ? 'Batch: every δ uses the same frozen weights (w before), and w moves once by Σ Δw.'
      : 'Online: w moves after each row, so each row reads the weights the row above left behind.';
    breakdown.innerHTML = `
      <p class="order">${order} Each Δw<sub>j</sub> = α·δ·x<sub>j</sub>(s) with α = ${num(a)}; “·” marks a zero feature. Hover a cell for its arithmetic.</p>
      <div class="table-scroll"><table class="steps">
        <thead>${head}</thead>
        <tbody>${rows}</tbody>
        <tfoot>
          ${footRow('Σ Δw', deltaW, 'sum')}
          ${footRow('w before', before)}
          ${footRow('w after = w before + Σ Δw', after, 'after')}
        </tfoot>
      </table></div>`;
  }

  function drawChart(
    canvas: HTMLCanvasElement,
    readout: HTMLElement,
    count: number,
    pick: (weights: readonly number[], j: number) => number,
    name: (j: number) => string,
    colorIndex: (j: number) => number,
  ) {
    const indices = sampleIndices(history.length);
    const iteration = history.length - 1;
    const transform = logScale ? symlog : (v: number) => v;
    const series: Series[] = Array.from({ length: count }, (_, j) => ({
      label: name(j),
      color: seriesColor(colorIndex(j)),
      points: indices.map(i => transform(pick(history[i], j))),
      xEnd: iteration,
    }));
    let lo = 0;
    let hi = 0;
    for (const s of series) for (const p of s.points) if (Number.isFinite(p)) { lo = Math.min(lo, p); hi = Math.max(hi, p); }
    if (hi - lo < 1e-9) { lo = -1; hi = 1; }
    // On the signed-log axis, centre on 0 so the middle tick is exactly the true value.
    if (logScale) { hi = Math.max(-lo, hi); lo = -hi; }
    const pad = (hi - lo) * 0.06;
    const xMax = Math.max(10, iteration);
    // The true value of every state is 0, so 0 is the line every curve should settle on.
    series.unshift({ label: 'True value 0', color: cssVar('--muted'), points: [0, 0], xEnd: xMax, dash: [5, 4] });
    drawLineChart(canvas, series, {
      yMin: lo - pad,
      yMax: hi + pad,
      xMax,
      xLabel: mode === 'batch' ? 'Batches' : 'Sweeps',
      formatY: v => axisLabel(logScale ? unsymlog(v) : v),
      gridColor: cssVar('--grid'),
      textColor: cssVar('--muted'),
    });

    const at = hover?.canvas === canvas ? Math.min(hover.iteration, iteration) : null;
    if (at === null || at > xMax) {
      readout.textContent = `Iteration ${iteration} · ` + Array.from({ length: count }, (_, j) => `${name(j)} ${num(pick(w, j))}`).join(' · ');
      return;
    }
    const ctx = canvas.getContext('2d')!;
    const rect = canvas.getBoundingClientRect();
    const x = PADDING.left + (at / xMax) * (rect.width - PADDING.left - PADDING.right);
    ctx.strokeStyle = cssVar('--muted');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, PADDING.top);
    ctx.lineTo(x, rect.height - PADDING.bottom);
    ctx.stroke();
    readout.textContent = `Iteration ${at} · ` + Array.from({ length: count }, (_, j) => `${name(j)} ${num(pick(history[at], j))}`).join(' · ');
  }

  function renderCharts() {
    drawChart(weightCanvas, weightReadout, WEIGHTS, (ws, j) => ws[j], j => `w${SUB[j]}`, j => j);
    drawChart(valueCanvas, valueReadout, STATES, (ws, j) => stateValues(ws)[j], j => `s${j + 1}`, j => j + 1);
  }

  function render() {
    const values = stateValues(w);
    const iteration = history.length - 1;
    iterationLabel.textContent = String(iteration);
    badge.hidden = !diverged;
    const largest = Math.max(...values.map(Math.abs));
    const unit = `${iteration} ${mode === 'batch' ? 'batch' : 'sweep'}${iteration === 1 ? '' : mode === 'batch' ? 'es' : 's'}`;
    status.textContent = diverged
      ? `Diverged after ${unit}: |v̂| passed 10⁶ while every true value is 0. Reset to try other settings.`
      : iteration === 0
        ? `Every true value is 0, and w = 0 represents it exactly. Start from these weights and step.`
        : `Largest |v̂(s)| is ${num(largest)} after ${unit}. Every true value is 0.`;
    renderControls();
    renderDiagram(values);
    renderWeights();
    renderBreakdown();
    renderCharts();
  }

  for (const input of [alphaInput, gammaInput, ...weightInputs]) input.addEventListener('change', reset);
  for (const b of modeButtons) b.addEventListener('click', () => { mode = b.dataset.mode as UpdateMode; reset(); });
  for (const b of scaleButtons) b.addEventListener('click', () => { logScale = b.dataset.scale === 'log'; render(); });
  stepButton.addEventListener('click', () => { advance(1); render(); });
  step10Button.addEventListener('click', () => { advance(10); render(); });
  playButton.addEventListener('click', () => (playing ? pause() : play()));
  speedInput.addEventListener('input', renderControls);
  $<HTMLButtonElement>('[data-reset]').addEventListener('click', reset);
  $<HTMLButtonElement>('[data-defaults]').addEventListener('click', () => {
    alphaInput.value = String(DEFAULT_ALPHA);
    gammaInput.value = String(DEFAULT_GAMMA);
    weightInputs.forEach((input, j) => { input.value = String(DEFAULT_WEIGHTS[j]); });
    reset();
  });

  for (const canvas of [weightCanvas, valueCanvas]) {
    canvas.addEventListener('pointermove', event => {
      const rect = canvas.getBoundingClientRect();
      const xMax = Math.max(10, history.length - 1);
      const frac = (event.clientX - rect.left - PADDING.left) / (rect.width - PADDING.left - PADDING.right);
      hover = { canvas, iteration: Math.round(Math.min(1, Math.max(0, frac)) * xMax) };
      renderCharts();
    });
    canvas.addEventListener('pointerleave', () => { hover = null; renderCharts(); });
  }

  let resizeFrame = 0;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(renderCharts);
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderCharts);
  syncTheme(renderCharts);
  render();
}
