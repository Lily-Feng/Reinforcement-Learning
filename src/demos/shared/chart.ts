/**
 * A minimal canvas line chart, shared by every demo in the book.
 *
 * Deliberately not a charting library: demos plot one or two series against
 * step count, and a hundred lines here beats a dependency on every page.
 */

export interface Series {
  label: string;
  color: string;
  points: readonly number[];
  /**
   * The x value the LAST point corresponds to. Series are downsampled, so the
   * number of points is not the number of steps; without this the curve is
   * drawn in index space and lands in the wrong place.
   */
  xEnd?: number;
  /** Shade the area under the line. Only sensible for a single series. */
  fill?: boolean;
}

export interface ChartOptions {
  yMin?: number;
  yMax?: number;
  /** The x extent of the axis. Series shorter than this end part-way across. */
  xMax?: number;
  yTicks?: number;
  xLabel?: string;
  yLabel?: string;
  gridColor?: string;
  textColor?: string;
  /** Formats y tick labels. */
  formatY?: (value: number) => string;
}

const PADDING = { top: 12, right: 14, bottom: 28, left: 44 };

/**
 * Sizes the backing store to the element's real pixel dimensions.
 *
 * Without this the canvas renders at CSS resolution and looks soft on any
 * retina display. Returns the CSS-pixel size to draw against.
 */
export function fitCanvas(canvas: HTMLCanvasElement): { width: number; height: number } {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));

  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);

  const ctx = canvas.getContext('2d');
  if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  return { width, height };
}

export function clearCanvas(canvas: HTMLCanvasElement): void {
  const { width, height } = fitCanvas(canvas);
  canvas.getContext('2d')?.clearRect(0, 0, width, height);
}

export function drawLineChart(
  canvas: HTMLCanvasElement,
  series: readonly Series[],
  options: ChartOptions = {},
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const { width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const yMin = options.yMin ?? 0;
  const yMax = options.yMax ?? 1;
  const yTicks = options.yTicks ?? 4;
  const gridColor = options.gridColor ?? '#334155';
  const textColor = options.textColor ?? '#94a3b8';
  const formatY = options.formatY ?? ((v: number) => v.toFixed(2));

  const plotW = width - PADDING.left - PADDING.right;
  const plotH = height - PADDING.top - PADDING.bottom;
  if (plotW <= 0 || plotH <= 0) return;

  const longest = series.reduce((max, s) => Math.max(max, s.xEnd ?? s.points.length), 0);
  const xMax = Math.max(1, options.xMax ?? longest);

  /** Maps point `i` of an `n`-point series ending at step `xEnd` to a pixel x. */
  const toX = (i: number, n: number, xEnd: number) => {
    const frac = n <= 1 ? 0 : (i / (n - 1)) * (xEnd / xMax);
    return PADDING.left + Math.min(1, frac) * plotW;
  };
  const toY = (v: number) => {
    const t = (v - yMin) / (yMax - yMin || 1);
    return PADDING.top + (1 - Math.min(1, Math.max(0, t))) * plotH;
  };

  // Horizontal grid and y-axis labels.
  ctx.lineWidth = 1;
  ctx.strokeStyle = gridColor;
  ctx.fillStyle = textColor;
  ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  for (let i = 0; i <= yTicks; i += 1) {
    const value = yMin + (i / yTicks) * (yMax - yMin);
    const y = toY(value);
    ctx.beginPath();
    ctx.moveTo(PADDING.left, y);
    ctx.lineTo(width - PADDING.right, y);
    ctx.stroke();
    ctx.fillText(formatY(value), PADDING.left - 8, y);
  }

  // Axis labels.
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  if (options.xLabel) {
    ctx.fillText(options.xLabel, PADDING.left + plotW / 2, height - 6);
  }
  ctx.textAlign = 'left';
  if (options.xLabel) {
    ctx.fillText('0', PADDING.left, height - PADDING.bottom + 16);
    ctx.textAlign = 'right';
    ctx.fillText(String(Math.round(xMax)), width - PADDING.right, height - PADDING.bottom + 16);
  }

  for (const s of series) {
    if (s.points.length === 0) continue;

    const n = s.points.length;
    const xEnd = s.xEnd ?? n;

    if (s.fill && n > 1) {
      ctx.beginPath();
      ctx.moveTo(toX(0, n, xEnd), toY(yMin));
      s.points.forEach((v, i) => ctx.lineTo(toX(i, n, xEnd), toY(v)));
      ctx.lineTo(toX(n - 1, n, xEnd), toY(yMin));
      ctx.closePath();
      ctx.fillStyle = s.color;
      ctx.globalAlpha = 0.12;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    ctx.beginPath();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    s.points.forEach((v, i) => {
      const x = toX(i, n, xEnd);
      const y = toY(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

/** Reduces a long series to at most `maxPoints` evenly-spaced samples. */
export function downsample(series: readonly number[], maxPoints: number): number[] {
  if (series.length <= maxPoints) return [...series];
  const stride = series.length / maxPoints;
  const out: number[] = [];
  for (let i = 0; i < maxPoints; i += 1) {
    out.push(series[Math.min(series.length - 1, Math.floor(i * stride))]!);
  }
  return out;
}
