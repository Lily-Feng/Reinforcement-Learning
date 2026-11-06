/**
 * Pieces several modules share: KPI tiles, tier legend, decision maps over
 * free GPUs × job length, and the bucketed optimum read as a map.
 */

import { drawHeatmap, attachHeatmapHover, mix, type HeatmapLayout } from '../../shared/heatmap';
import { lengthAt } from '../core/buckets.ts';
import { TIER_NAMES, type ClusterConfig } from '../core/config.ts';
import type { Reference } from '../core/jobs.ts';
import { stateIndex, type Model } from '../core/model.ts';
import { canvasPanel, el, legend, range, theme } from '../app/ui';

export function kpi(label: string, value = '—', small = ''): { el: HTMLElement; set: (value: string, small?: string) => void } {
  const v = el('strong', { text: value });
  const s = el('small', { text: small });
  return { el: el('div', { class: 'kpi' }, el('span', { text: label }), v, s), set: (a, b) => { v.textContent = a; if (b !== undefined) s.textContent = b; } };
}

export function tierLegend(): HTMLElement {
  const t = theme();
  return legend(TIER_NAMES.map((name, c) => ({ label: `${name} ($${[1, 2, 4, 8][c]})`, color: t.tier[c]! })));
}

/** Log-spaced job lengths 2 … 120 for map columns. */
export function lengthGrid(n = 24): number[] {
  return Array.from({ length: n }, (_, i) => 2 * 60 ** (i / (n - 1)));
}

/** Upper length boundaries of each bucket (the last is ∞). */
export function bucketEdges(model: Model, cfg: ClusterConfig): number[] {
  return [...model.buckets.zCuts.map(z => lengthAt(cfg, z)), Infinity];
}

export function bucketOfLength(edges: number[], len: number): number {
  let b = 0;
  while (b < edges.length - 1 && len >= edges[b]!) b += 1;
  return b;
}

/** The most likely count vector with g free GPUs under the reference's stationary distribution. */
export function typicalCounts(model: Model, nu: Float64Array): number[] {
  const best = new Array<number>(model.K + 1).fill(-1);
  const mass = new Array<number>(model.K + 1).fill(-1);
  for (let ci = 0; ci < model.nCounts; ci += 1) {
    const g = model.free[ci]!;
    if (nu[ci]! > mass[g]!) { mass[g] = nu[ci]!; best[g] = ci; }
  }
  return best;
}

/** Optimal decision as a function of (tier, free GPUs, length), with the busy jobs at their most likely mix. */
export function optimumDecider(model: Model, cfg: ClusterConfig, ref: Reference): (c: number, g: number, len: number) => number {
  const edges = bucketEdges(model, cfg);
  const typical = typicalCounts(model, ref.nu);
  return (c, g, len) => {
    if (g === 0) return 0;
    const ci = typical[g]!;
    if (ci < 0) return 0;
    return ref.policy[stateIndex(model, ci, c, bucketOfLength(edges, len))]!;
  };
}

export interface MapPanel {
  el: HTMLElement;
  redraw: () => void;
  destroy: () => void;
}

/**
 * One tier's decision map: rows are free GPUs (K at the top), columns are job
 * lengths on a log scale. `decide` returns 1 (accept), 0 (reject) or null.
 * `weight` (0..1) fades rarely visited cells.
 */
export function decisionMap(title: string, K: () => number, lengths: number[], decide: (g: number, li: number) => number | null, opts: { weight?: (g: number, li: number) => number; height?: number; readout?: (g: number, li: number) => string } = {}): MapPanel {
  let layout: HeatmapLayout = { cellAt: () => null };
  const readout = el('p', { class: 'readout' });
  const panel = canvasPanel(title, opts.height ?? 170, canvas => {
    const t = theme();
    const k = K();
    layout = drawHeatmap(canvas, {
      rows: k + 1, cols: lengths.length,
      color: (r, c) => {
        const g = k - r;
        const d = decide(g, c);
        if (d === null) return null;
        const base = d ? t.accept : t.reject;
        const w = opts.weight?.(g, c) ?? 1;
        return w >= 1 ? base : mix(t.surface, base, 0.25 + 0.75 * w);
      },
      rowLabel: r => String(k - r),
      colLabel: c => (c % 6 === 0 || c === lengths.length - 1 ? lengths[c]!.toFixed(0) : ''),
      colLabelEvery: 1,
      xLabel: 'job length ℓ (log)',
      yLabel: 'free GPUs g',
      textColor: t.text, gridColor: t.grid,
    });
  }, { note: readout });
  const off = attachHeatmapHover(panel.canvas, () => layout, cell => {
    if (!cell) { readout.textContent = ''; return; }
    const g = K() - cell.row;
    const d = decide(g, cell.col);
    readout.textContent = `g = ${g}, ℓ ≈ ${lengths[cell.col]!.toFixed(1)}: ${d === null ? '—' : d ? 'accept' : 'reject'}${opts.readout ? ` · ${opts.readout(g, cell.col)}` : ''}`;
  });
  return { el: panel.el, redraw: panel.redraw, destroy: () => { off(); panel.destroy(); } };
}

export const TIER_SHORT = ['Free', 'Basic', 'Pro', 'Ent'];

export function mapLegend(): HTMLElement {
  const t = theme();
  return legend([{ label: 'accept', color: t.accept }, { label: 'reject', color: t.reject }]);
}

/** Records (x, y) points and keeps at most `cap` of them by halving resolution. */
export function recorder(cap = 1200) {
  let points: [number, number][] = [];
  let stride = 1;
  let n = 0;
  return {
    push: (x: number, y: number) => {
      if (n % stride === 0) {
        points.push([x, y]);
        if (points.length > cap) { points = points.filter((_, i) => i % 2 === 0); stride *= 2; }
      }
      n += 1;
    },
    points: () => points,
    count: () => n,
  };
}

/** θ and Λ sliders for a threshold policy; `onChange` fires after each edit. */
export function thresholdControls(th: { theta: number[]; lambda: number[] }, K: number, onChange: () => void, withLength = true): HTMLElement {
  return el('div', { class: 'controls' },
    ...TIER_NAMES.map((name, c) => range(`${name}: accept if g ≥`, { min: 1, max: K + 1, step: 1, value: Math.min(K + 1, th.theta[c]!), format: v => (v > K ? 'never' : String(v)) }, v => { th.theta[c] = v; onChange(); }).el),
    ...(withLength ? TIER_NAMES.map((name, c) => range(`${name}: and ℓ ≤`, { min: 2, max: 121, step: 1, value: Math.min(121, th.lambda[c]!), format: v => (v > 120 ? '∞' : String(v)) }, v => { th.lambda[c] = v > 120 ? Infinity : v; onChange(); }).el) : []));
}

/** Debounces calls to the next animation frame. */
export function onNextFrame(fn: () => void): () => void {
  let raf = 0;
  return () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; fn(); }); };
}

export function bucketWords(B: number): string[] {
  return [['all'], ['short', 'long'], ['short', 'medium', 'long'], ['short', 'medium', 'long', 'very long']][B - 1] ?? [];
}

/**
 * Count-vector index with g free GPUs and the busy jobs split across buckets
 * in proportion to bucket length (longer jobs stay longer, so they pile up).
 */
export function occupancyCounts(model: Model): number[] {
  const index = new Map(model.counts.map((n, i) => [n.join(','), i]));
  const total = model.buckets.len.reduce((s, v) => s + v, 0);
  return Array.from({ length: model.K + 1 }, (_, g) => {
    const busy = model.K - g;
    const raw = model.buckets.len.map(l => (busy * l) / total);
    const n = raw.map(Math.floor);
    let left = busy - n.reduce((s, v) => s + v, 0);
    raw.map((v, b) => ({ b, frac: v - Math.floor(v) })).sort((a, b) => b.frac - a.frac).forEach(({ b }) => { if (left > 0) { n[b]! += 1; left -= 1; } });
    return index.get(n.join(',')) ?? -1;
  });
}
