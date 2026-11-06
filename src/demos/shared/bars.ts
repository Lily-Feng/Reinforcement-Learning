/**
 * Canvas bar charts (plain, stacked, histograms, with optional error bars and
 * reference lines) and an x–y plot with explicit x values (lines, points,
 * error bars). Same visual language as chart.ts.
 */

import { PADDING, fitCanvas } from './chart';

export interface BarSegment {
  value: number;
  color: string;
  /** ± half-width of an error bar on top of this segment. */
  err?: number;
}

export interface BarGroup {
  label: string;
  /** Segments, stacked bottom to top (or side by side when `stacked` is false). */
  segments: BarSegment[];
  highlight?: boolean;
}

export interface RefLine {
  value: number;
  color: string;
  label?: string;
  dash?: number[];
  /** Shaded band [value − band, value + band]. */
  band?: number;
}

export interface BarOptions {
  yMin?: number;
  yMax?: number;
  stacked?: boolean;
  formatY?: (v: number) => string;
  xLabel?: string;
  refs?: RefLine[];
  textColor?: string;
  gridColor?: string;
  highlightColor?: string;
  /** Label every k-th group. */
  labelEvery?: number;
  /** Per-group marker values drawn as short horizontal ticks with a dot (e.g. an empirical estimate). */
  markers?: { values: readonly number[]; color: string }[];
}

function axes(ctx: CanvasRenderingContext2D, width: number, height: number, yMin: number, yMax: number, o: { formatY?: (v: number) => string; gridColor?: string; textColor?: string; xLabel?: string }) {
  const plotW = width - PADDING.left - PADDING.right;
  const plotH = height - PADDING.top - PADDING.bottom;
  const toY = (v: number) => PADDING.top + (1 - (v - yMin) / (yMax - yMin || 1)) * plotH;
  ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
  ctx.strokeStyle = o.gridColor ?? '#334155';
  ctx.fillStyle = o.textColor ?? '#94a3b8';
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const fmt = o.formatY ?? ((v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)));
  for (let i = 0; i <= 4; i += 1) {
    const v = yMin + (i / 4) * (yMax - yMin);
    const y = toY(v);
    ctx.beginPath();
    ctx.moveTo(PADDING.left, y);
    ctx.lineTo(width - PADDING.right, y);
    ctx.stroke();
    ctx.fillText(fmt(v), PADDING.left - 6, y);
  }
  if (o.xLabel) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(o.xLabel, PADDING.left + plotW / 2, height - 3);
  }
  return { plotW, plotH, toY };
}

function drawRefs(ctx: CanvasRenderingContext2D, refs: RefLine[] | undefined, toY: (v: number) => number, width: number) {
  for (const r of refs ?? []) {
    if (r.band) {
      ctx.fillStyle = r.color;
      ctx.globalAlpha = 0.15;
      const y0 = toY(r.value + r.band);
      ctx.fillRect(PADDING.left, y0, width - PADDING.left - PADDING.right, toY(r.value - r.band) - y0);
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = r.color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(r.dash ?? [5, 4]);
    ctx.beginPath();
    ctx.moveTo(PADDING.left, toY(r.value));
    ctx.lineTo(width - PADDING.right, toY(r.value));
    ctx.stroke();
    ctx.setLineDash([]);
    if (r.label) {
      ctx.fillStyle = r.color;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText(r.label, width - PADDING.right - 2, toY(r.value) - 2);
    }
  }
}

export function drawBars(canvas: HTMLCanvasElement, groups: readonly BarGroup[], o: BarOptions = {}): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const totals = groups.map((g, i) => Math.max(o.stacked ? g.segments.reduce((s, v) => s + v.value, 0) : Math.max(0, ...g.segments.map(s => s.value + (s.err ?? 0))), ...(o.markers ?? []).map(m => m.values[i] ?? 0)));
  const mins = groups.map(g => Math.min(0, ...g.segments.map(s => s.value - (s.err ?? 0))));
  const yMax = o.yMax ?? Math.max(1e-9, ...totals, ...(o.refs ?? []).map(r => r.value + (r.band ?? 0))) * 1.05;
  const yMin = o.yMin ?? Math.min(0, ...mins);
  const { plotW, toY } = axes(ctx, width, height, yMin, yMax, o);
  const n = Math.max(1, groups.length);
  const slot = plotW / n;
  const barW = Math.max(1, slot * 0.78);
  const every = o.labelEvery ?? Math.max(1, Math.ceil(26 / slot));
  groups.forEach((g, i) => {
    const x0 = PADDING.left + i * slot + (slot - barW) / 2;
    if (g.highlight) {
      ctx.fillStyle = o.highlightColor ?? 'rgba(250, 204, 21, 0.18)';
      ctx.fillRect(PADDING.left + i * slot, PADDING.top, slot, toY(yMin) - PADDING.top);
    }
    let base = 0;
    const w = o.stacked ? barW : barW / Math.max(1, g.segments.length);
    g.segments.forEach((s, k) => {
      const x = o.stacked ? x0 : x0 + k * w;
      const lo = o.stacked ? base : 0;
      const hi = lo + s.value;
      ctx.fillStyle = s.color;
      const yTop = toY(Math.max(lo, hi));
      ctx.fillRect(x, yTop, Math.max(1, w - (o.stacked ? 0 : 1)), Math.abs(toY(lo) - toY(hi)));
      if (s.err) {
        ctx.strokeStyle = o.textColor ?? '#94a3b8';
        ctx.lineWidth = 1;
        const cx = x + w / 2;
        ctx.beginPath();
        ctx.moveTo(cx, toY(hi - s.err));
        ctx.lineTo(cx, toY(hi + s.err));
        ctx.moveTo(cx - 3, toY(hi - s.err));
        ctx.lineTo(cx + 3, toY(hi - s.err));
        ctx.moveTo(cx - 3, toY(hi + s.err));
        ctx.lineTo(cx + 3, toY(hi + s.err));
        ctx.stroke();
      }
      if (o.stacked) base = hi;
    });
    for (const m of o.markers ?? []) {
      const v = m.values[i];
      if (v === undefined || !Number.isFinite(v)) continue;
      const y = toY(v);
      ctx.strokeStyle = m.color;
      ctx.fillStyle = m.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x0 - 2, y);
      ctx.lineTo(x0 + barW + 2, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x0 + barW / 2, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    if (i % every === 0) {
      ctx.fillStyle = o.textColor ?? '#94a3b8';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(g.label, PADDING.left + (i + 0.5) * slot, height - PADDING.bottom + 4);
    }
  });
  drawRefs(ctx, o.refs, toY, width);
}

export interface XYSeries {
  label?: string;
  color: string;
  /** [x, y] or [x, y, ±err]. */
  points: readonly (readonly number[])[];
  line?: boolean;
  dots?: boolean;
  dash?: number[];
  /** Dot radius. */
  r?: number;
}

export interface XYOptions {
  xMin?: number;
  xMax?: number;
  yMin?: number;
  yMax?: number;
  xLabel?: string;
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  /** Log-scale x axis (values must be positive). */
  logX?: boolean;
  /** Draw y = x. */
  diagonal?: boolean;
  refs?: RefLine[];
  /** Vertical marker lines. */
  vlines?: { x: number; color: string; label?: string; band?: [number, number] }[];
  textColor?: string;
  gridColor?: string;
}

export function drawXY(canvas: HTMLCanvasElement, series: readonly XYSeries[], o: XYOptions = {}): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const all = series.flatMap(s => s.points);
  const tx = (x: number) => (o.logX ? Math.log10(x) : x);
  const xs = all.map(p => tx(p[0]!));
  const ys = all.flatMap(p => [p[1]! - (p[2] ?? 0), p[1]! + (p[2] ?? 0)]);
  let xMin = o.xMin !== undefined ? tx(o.xMin) : Math.min(...xs);
  let xMax = o.xMax !== undefined ? tx(o.xMax) : Math.max(...xs);
  let yMin = o.yMin ?? Math.min(...ys, ...(o.refs ?? []).map(r => r.value));
  let yMax = o.yMax ?? Math.max(...ys, ...(o.refs ?? []).map(r => r.value));
  if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) { xMin = 0; xMax = 1; }
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) { yMin = 0; yMax = 1; }
  if (xMax === xMin) { xMax += 0.5; xMin -= 0.5; }
  if (yMax === yMin) { yMax += 0.5; yMin -= 0.5; }
  if (o.yMin === undefined && o.yMax === undefined) {
    const pad = (yMax - yMin) * 0.06;
    yMin -= pad;
    yMax += pad;
  }
  const { plotW, toY } = axes(ctx, width, height, yMin, yMax, o);
  const toX = (x: number) => PADDING.left + ((tx(x) - xMin) / (xMax - xMin)) * plotW;
  const text = o.textColor ?? '#94a3b8';

  ctx.fillStyle = text;
  ctx.textBaseline = 'top';
  const fmtX = o.formatX ?? ((v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : String(Math.round(v * 1000) / 1000)));
  const ticks = o.logX
    ? Array.from({ length: Math.ceil(xMax) - Math.floor(xMin) + 1 }, (_, i) => Math.floor(xMin) + i)
      .flatMap(e => [1, 2, 5].map(m => m * 10 ** e))
      .filter(v => Math.log10(v) >= xMin - 1e-9 && Math.log10(v) <= xMax + 1e-9)
    : Array.from({ length: 5 }, (_, i) => xMin + (i / 4) * (xMax - xMin));
  ctx.textAlign = 'center';
  for (const t of ticks) ctx.fillText(fmtX(t), toX(t), height - PADDING.bottom + 4);

  if (o.diagonal) {
    const lo = Math.max(yMin, o.logX ? -Infinity : xMin);
    const hi = Math.min(yMax, o.logX ? Infinity : xMax);
    ctx.strokeStyle = text;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(toX(lo), toY(lo));
    ctx.lineTo(toX(hi), toY(hi));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const v of o.vlines ?? []) {
    if (v.band) {
      ctx.fillStyle = v.color;
      ctx.globalAlpha = 0.15;
      ctx.fillRect(toX(v.band[0]), PADDING.top, toX(v.band[1]) - toX(v.band[0]), toY(yMin) - PADDING.top);
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = v.color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(toX(v.x), PADDING.top);
    ctx.lineTo(toX(v.x), toY(yMin));
    ctx.stroke();
    ctx.setLineDash([]);
    if (v.label) {
      ctx.fillStyle = v.color;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(v.label, toX(v.x) + 3, PADDING.top + 2);
    }
  }
  drawRefs(ctx, o.refs, toY, width);

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    if (s.line !== false && s.points.length > 1) {
      ctx.lineWidth = 2;
      ctx.setLineDash(s.dash ?? []);
      ctx.beginPath();
      s.points.forEach((p, i) => (i === 0 ? ctx.moveTo(toX(p[0]!), toY(p[1]!)) : ctx.lineTo(toX(p[0]!), toY(p[1]!))));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const p of s.points) {
      const x = toX(p[0]!);
      const y = toY(p[1]!);
      if (p[2]) {
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, toY(p[1]! - p[2]));
        ctx.lineTo(x, toY(p[1]! + p[2]));
        ctx.stroke();
      }
      if (s.dots || s.line === false) {
        ctx.beginPath();
        ctx.arc(x, y, s.r ?? 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}
