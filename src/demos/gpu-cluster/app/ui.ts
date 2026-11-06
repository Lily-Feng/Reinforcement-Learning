/**
 * Tiny DOM helpers shared by the lab's modules: element builder, labelled
 * controls, cards, canvas panels that redraw on resize, and a time-sliced
 * runner for interactive training (≤ 8 ms of work per animation frame).
 */

type Child = Node | string | null | undefined | false;

export interface ElProps {
  class?: string;
  text?: string;
  html?: string;
  attrs?: Record<string, string>;
  on?: Partial<Record<keyof HTMLElementEventMap, (e: Event) => void>>;
}

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: ElProps = {}, ...children: Child[]): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (props.class) node.className = props.class;
  if (props.text !== undefined) node.textContent = props.text;
  if (props.html !== undefined) node.innerHTML = props.html;
  for (const [k, v] of Object.entries(props.attrs ?? {})) node.setAttribute(k, v);
  for (const [k, fn] of Object.entries(props.on ?? {})) if (fn) node.addEventListener(k, fn);
  for (const c of children) if (c !== null && c !== undefined && c !== false) node.append(c);
  return node;
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888888';
}

/** Theme colors read at draw time, so light/dark switches apply on the next redraw. */
export function theme() {
  const v = cssVar;
  return {
    text: v('--muted'), ink: v('--text'), grid: v('--grid'), surface: v('--surface'), accent: v('--accent'),
    accept: v('--accept'), reject: v('--reject'), ref: v('--ref'), danger: v('--danger'),
    tier: [v('--tier-0'), v('--tier-1'), v('--tier-2'), v('--tier-3')],
    series: [v('--series-1'), v('--series-2'), v('--series-3'), v('--series-4'), v('--series-5'), v('--series-6')],
    bucket: [v('--bucket-0'), v('--bucket-1'), v('--bucket-2'), v('--bucket-3')],
    seqLow: v('--seq-low'), seqHigh: v('--seq-high'), divLow: v('--div-low'), divMid: v('--div-mid'), divHigh: v('--div-high'),
  };
}

export const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : v === Infinity ? '∞' : '—');

export interface Control<T> {
  el: HTMLElement;
  get: () => T;
  set: (v: T) => void;
}

export function range(label: string, o: { min: number; max: number; step: number; value: number; format?: (v: number) => string; title?: string }, onInput: (v: number) => void): Control<number> {
  const out = el('output');
  const input = el('input', { attrs: { type: 'range', min: String(o.min), max: String(o.max), step: String(o.step), value: String(o.value) } });
  const format = o.format ?? (v => String(v));
  out.textContent = format(o.value);
  input.addEventListener('input', () => { out.textContent = format(Number(input.value)); onInput(Number(input.value)); });
  const wrap = el('label', { class: 'ctl', attrs: o.title ? { title: o.title } : {} }, el('span', { text: label }), out, input);
  return { el: wrap, get: () => Number(input.value), set: v => { input.value = String(v); out.textContent = format(v); } };
}

export function select<T extends string>(label: string, options: readonly { value: T; label: string }[], value: T, onChange: (v: T) => void): Control<T> {
  const s = el('select');
  for (const o of options) s.append(el('option', { text: o.label, attrs: { value: o.value } }));
  s.value = value;
  s.addEventListener('change', () => onChange(s.value as T));
  return { el: el('label', { class: 'ctl' }, el('span', { text: label }), s), get: () => s.value as T, set: v => { s.value = v; } };
}

export function check(label: string, checked: boolean, onChange: (v: boolean) => void): Control<boolean> {
  const input = el('input', { attrs: { type: 'checkbox' } });
  input.checked = checked;
  input.addEventListener('change', () => onChange(input.checked));
  return { el: el('label', { class: 'check' }, input, el('span', { text: label })), get: () => input.checked, set: v => { input.checked = v; } };
}

export function numberInput(label: string, value: number, onChange: (v: number) => void, attrs: Record<string, string> = {}): Control<number> {
  const input = el('input', { attrs: { type: 'number', value: String(value), ...attrs } });
  input.addEventListener('change', () => { const v = Number(input.value); if (Number.isFinite(v)) onChange(v); });
  return { el: el('label', { class: 'ctl' }, el('span', { text: label }), input), get: () => Number(input.value), set: v => { input.value = String(v); } };
}

export function button(label: string, onClick: () => void, cls = ''): HTMLButtonElement {
  return el('button', { class: cls, text: label, attrs: { type: 'button' }, on: { click: () => onClick() } });
}

export function card(title: string | null, ...children: Child[]): HTMLElement {
  return el('section', { class: 'card' }, title ? el('h3', { text: title }) : null, ...children);
}

/** The module's "what to observe" box. */
export function observeBox(items: string[]): HTMLElement {
  return el('details', { class: 'observe', attrs: { open: '' } }, el('summary', { text: 'What to observe' }), el('ul', {}, ...items.map(t => el('li', { html: t }))));
}

export function moduleHeader(id: string, title: string, lede: string): HTMLElement {
  return el('header', { class: 'mod-head' }, el('h2', {}, el('span', { class: 'mod-id', text: id.toUpperCase() }), ` ${title}`), el('p', { class: 'lede', html: lede }));
}

/** A canvas in a card that redraws whenever its size changes. */
export function canvasPanel(title: string | null, height: number, draw: (canvas: HTMLCanvasElement) => void, opts: { legend?: HTMLElement; note?: HTMLElement; aria?: string } = {}) {
  const canvas = el('canvas', { attrs: { role: 'img', 'aria-label': opts.aria ?? title ?? 'chart' } });
  canvas.style.height = `${height}px`;
  const box = card(title, opts.legend ?? null, canvas, opts.note ?? null);
  let raf = 0;
  const redraw = () => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => { if (canvas.isConnected) draw(canvas); });
  };
  const ro = new ResizeObserver(redraw);
  ro.observe(canvas);
  return { el: box, canvas, redraw, destroy: () => { ro.disconnect(); cancelAnimationFrame(raf); } };
}

export function legend(items: { label: string; color: string; dash?: boolean }[]): HTMLElement {
  return el('div', { class: 'legend' }, ...items.map(i => el('span', {}, el('i', { attrs: { style: i.dash ? `border-top: 2px dashed ${i.color}; background: none; height: 0` : `background: ${i.color}` } }), i.label)));
}

export function progressBar(): { el: HTMLElement; set: (fraction: number | null, label?: string) => void } {
  const bar = el('div', { class: 'progress', attrs: { hidden: '' } }, el('div', { class: 'progress-fill' }), el('span', { class: 'progress-label' }));
  return {
    el: bar,
    set: (f, label = '') => {
      if (f === null) { bar.setAttribute('hidden', ''); return; }
      bar.removeAttribute('hidden');
      (bar.firstChild as HTMLElement).style.width = `${Math.round(f * 100)}%`;
      (bar.lastChild as HTMLElement).textContent = label || `${Math.round(f * 100)}%`;
    },
  };
}

/** Collects cleanup callbacks; `run()` calls them all once. */
export function disposer() {
  const fns: (() => void)[] = [];
  return {
    add: (fn: () => void) => { fns.push(fn); },
    run: () => { while (fns.length) { try { fns.pop()!(); } catch { /* keep cleaning up */ } } },
  };
}

export interface Runner {
  readonly playing: boolean;
  play: () => void;
  pause: () => void;
  step: (units: number) => void;
  fastForward: (units: number) => void;
  /** Units completed since creation. */
  done: () => number;
  destroy: () => void;
}

/**
 * Time-sliced work loop. `unit()` does one unit of work (one step, one
 * episode…); each animation frame runs units until `perFrame()` is reached or
 * the 8 ms budget is used, then `render()` once.
 */
export function createRunner(o: { unit: () => void; render: () => void; perFrame?: () => number; onState?: (playing: boolean, ffLeft: number) => void; budgetMs?: number }): Runner {
  const budget = o.budgetMs ?? 8;
  let playing = false;
  let pending = 0;
  let ff = 0;
  let completed = 0;
  let raf = 0;
  let alive = true;
  const notify = () => o.onState?.(playing, ff);
  const frame = () => {
    raf = 0;
    if (!alive) return;
    const start = performance.now();
    let n = 0;
    const cap = ff > 0 ? Infinity : playing ? (o.perFrame?.() ?? 1) : pending;
    while (n < cap && performance.now() - start < budget) {
      if (ff > 0) ff -= 1;
      else if (!playing) pending -= 1;
      o.unit();
      completed += 1;
      n += 1;
      if (ff === 0 && !playing && pending <= 0) break;
    }
    if (pending < 0) pending = 0;
    o.render();
    notify();
    if (playing || pending > 0 || ff > 0) raf = requestAnimationFrame(frame);
  };
  const kick = () => { if (!raf && alive) raf = requestAnimationFrame(frame); };
  return {
    get playing() { return playing; },
    play: () => { playing = true; notify(); kick(); },
    pause: () => { playing = false; ff = 0; pending = 0; notify(); },
    step: units => { pending += units; kick(); },
    fastForward: units => { ff += units; notify(); kick(); },
    done: () => completed,
    destroy: () => { alive = false; playing = false; if (raf) cancelAnimationFrame(raf); raf = 0; },
  };
}

/** Play / Pause / Step / ×10 / Fast-forward buttons bound to a runner. */
export function runnerControls(runner: Runner, opts: { stepLabel?: string; ff?: { label: string; units: number }[]; reducedMotion?: boolean } = {}) {
  const play = button('Play', () => (runner.playing ? runner.pause() : runner.play()), 'primary');
  const stepBtn = button(opts.stepLabel ?? 'Step', () => runner.step(1));
  const ten = button('×10', () => runner.step(10));
  const ffSel = el('select', { attrs: { 'aria-label': 'Fast-forward amount' } });
  for (const f of opts.ff ?? []) ffSel.append(el('option', { text: f.label, attrs: { value: String(f.units) } }));
  const ffBtn = button('Fast-forward', () => runner.fastForward(Number(ffSel.value)));
  const status = el('span', { class: 'hint' });
  const box = el('div', { class: 'runner' }, play, stepBtn, ten, ...(opts.ff?.length ? [ffSel, ffBtn] : []), status);
  return {
    el: box,
    sync: (playing: boolean, ffLeft: number) => {
      play.textContent = playing ? 'Pause' : 'Play';
      play.setAttribute('aria-pressed', String(playing));
      status.textContent = ffLeft > 0 ? `fast-forwarding… ${ffLeft.toLocaleString()} left` : '';
    },
  };
}

export function prefersReducedMotion(): boolean {
  try { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch { return false; }
}

/** Keeps a rolling window mean of the last `size` values. */
export function rollingMean(size: number) {
  const buf = new Float64Array(size);
  let n = 0;
  let i = 0;
  let sum = 0;
  return {
    push: (v: number) => { if (n === size) sum -= buf[i]!; else n += 1; buf[i] = v; sum += v; i = (i + 1) % size; },
    value: () => (n ? sum / n : NaN),
    count: () => n,
  };
}
