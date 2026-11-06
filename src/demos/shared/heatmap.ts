/**
 * A canvas grid / heatmap with tick labels, an optional color legend and a
 * hover readout. Cells are addressed (row, col) with row 0 at the top.
 */

import { fitCanvas } from './chart';

export interface HeatmapOptions {
  rows: number;
  cols: number;
  /** Cell fill; null leaves the cell empty. */
  color: (row: number, col: number) => string | null;
  /** Optional text drawn in a cell. */
  label?: (row: number, col: number) => string | null;
  labelColor?: (row: number, col: number) => string;
  rowLabel?: (row: number) => string;
  /** Width reserved for row labels (px). */
  rowLabelWidth?: number;
  colLabel?: (col: number) => string;
  /** Label every k-th column (default chosen to fit). */
  colLabelEvery?: number;
  xLabel?: string;
  yLabel?: string;
  /** A horizontal color ramp under the grid. */
  legend?: { min: string; max: string; ramp: (t: number) => string };
  textColor?: string;
  gridColor?: string;
  /** Extra drawing on top, with the cell geometry. */
  overlay?: (ctx: CanvasRenderingContext2D, cell: (row: number, col: number) => { x: number; y: number; w: number; h: number }) => void;
}

export interface HeatmapLayout {
  cellAt: (x: number, y: number) => { row: number; col: number } | null;
}

export function drawHeatmap(canvas: HTMLCanvasElement, o: HeatmapOptions): HeatmapLayout {
  const ctx = canvas.getContext('2d');
  const { width, height } = fitCanvas(canvas);
  const none: HeatmapLayout = { cellAt: () => null };
  if (!ctx) return none;
  ctx.clearRect(0, 0, width, height);
  const text = o.textColor ?? '#94a3b8';
  const left = o.rowLabel ? (o.rowLabelWidth ?? (o.yLabel ? 58 : 44)) : 8;
  const bottom = (o.colLabel ? 18 : 4) + (o.xLabel ? 16 : 0) + (o.legend ? 24 : 0);
  const top = 6;
  const right = 8;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  if (plotW <= 0 || plotH <= 0 || o.rows <= 0 || o.cols <= 0) return none;
  const cw = plotW / o.cols;
  const ch = plotH / o.rows;
  const cell = (row: number, col: number) => ({ x: left + col * cw, y: top + row * ch, w: cw, h: ch });

  ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
  for (let r = 0; r < o.rows; r += 1) {
    for (let c = 0; c < o.cols; c += 1) {
      const fill = o.color(r, c);
      const g = cell(r, c);
      if (fill) {
        ctx.fillStyle = fill;
        ctx.fillRect(g.x, g.y, Math.ceil(g.w), Math.ceil(g.h));
      }
      const label = o.label?.(r, c);
      if (label && cw > 14 && ch > 11) {
        ctx.fillStyle = o.labelColor?.(r, c) ?? text;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, g.x + g.w / 2, g.y + g.h / 2);
      }
    }
  }
  if (o.gridColor && cw > 6 && ch > 6) {
    ctx.strokeStyle = o.gridColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(left + 0.5, top + 0.5, plotW - 1, plotH - 1);
  }
  o.overlay?.(ctx, cell);

  ctx.fillStyle = text;
  if (o.rowLabel) {
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const every = Math.max(1, Math.ceil(13 / ch));
    for (let r = 0; r < o.rows; r += every) ctx.fillText(o.rowLabel(r), left - 5, top + (r + 0.5) * ch);
  }
  if (o.colLabel) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const every = o.colLabelEvery ?? Math.max(1, Math.ceil(30 / cw));
    for (let c = 0; c < o.cols; c += every) ctx.fillText(o.colLabel(c), left + (c + 0.5) * cw, top + plotH + 4);
  }
  if (o.xLabel) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(o.xLabel, left + plotW / 2, top + plotH + (o.colLabel ? 18 : 4) + 12);
  }
  if (o.yLabel) {
    ctx.save();
    ctx.translate(10, top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(o.yLabel, 0, 0);
    ctx.restore();
  }
  if (o.legend) {
    const y = height - 14;
    const x0 = left;
    const w = Math.min(220, plotW - 80);
    for (let i = 0; i < w; i += 1) {
      ctx.fillStyle = o.legend.ramp(i / (w - 1));
      ctx.fillRect(x0 + 34 + i, y - 5, 1, 10);
    }
    ctx.fillStyle = text;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    ctx.fillText(o.legend.min, x0 + 30, y);
    ctx.textAlign = 'left';
    ctx.fillText(o.legend.max, x0 + 38 + w, y);
  }

  return {
    cellAt: (x, y) => {
      const col = Math.floor((x - left) / cw);
      const row = Math.floor((y - top) / ch);
      return row >= 0 && row < o.rows && col >= 0 && col < o.cols ? { row, col } : null;
    },
  };
}

/** Hover / click readout for a heatmap. Returns a cleanup function. */
export function attachHeatmapHover(
  canvas: HTMLCanvasElement,
  layout: () => HeatmapLayout,
  onHover: (cell: { row: number; col: number } | null) => void,
  onClick?: (cell: { row: number; col: number }) => void,
): () => void {
  const pos = (e: MouseEvent) => {
    const r = canvas.getBoundingClientRect();
    return layout().cellAt(e.clientX - r.left, e.clientY - r.top);
  };
  const move = (e: MouseEvent) => onHover(pos(e));
  const leave = () => onHover(null);
  const click = (e: MouseEvent) => { const c = pos(e); if (c && onClick) onClick(c); };
  canvas.addEventListener('mousemove', move);
  canvas.addEventListener('mouseleave', leave);
  canvas.addEventListener('click', click);
  return () => {
    canvas.removeEventListener('mousemove', move);
    canvas.removeEventListener('mouseleave', leave);
    canvas.removeEventListener('click', click);
  };
}

/** Linear blend of two #rrggbb colors. */
export function mix(a: string, b: string, t: number): string {
  const pa = parseInt(a.slice(1), 16);
  const pb = parseInt(b.slice(1), 16);
  const ch = (p: number, s: number) => (p >> s) & 255;
  const v = (s: number) => Math.round(ch(pa, s) + (ch(pb, s) - ch(pa, s)) * Math.min(1, Math.max(0, t)));
  return `rgb(${v(16)}, ${v(8)}, ${v(0)})`;
}

/** Three-stop diverging ramp: t ∈ [0, 1], 0.5 is the midpoint. */
export function diverging(low: string, mid: string, high: string): (t: number) => string {
  return t => (t < 0.5 ? mix(low, mid, t * 2) : mix(mid, high, (t - 0.5) * 2));
}
