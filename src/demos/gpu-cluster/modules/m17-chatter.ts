/**
 * M17 · No policy improvement theorem: chattering. The greedy policy of a
 * differential Sarsa agent is recorded at checkpoints on a fixed sample of
 * bucketed states, with its exact r(π) and the μ*-weighted fraction of
 * states whose decision flipped.
 */

import { drawXY } from '../../shared/bars';
import { drawHeatmap } from '../../shared/heatmap';
import type { ChatterFeatures, ChatterResult } from '../core/jobs.ts';
import type { ModuleContext } from '../app/context';
import type { Job } from '../app/jobs';
import { button, canvasPanel, card, el, fmt, legend, moduleHeader, observeBox, progressBar, range, select, theme } from '../app/ui';
import { kpi, mapLegend } from './common';

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = { ...ctx.cfg, amp: 0 };
  const L = ctx.store.local('m17', { kind: 'coarse' as ChatterFeatures, alpha: 0.2, eps: 0.1, steps: 400000 });
  const results = new Map<ChatterFeatures, ChatterResult>();
  let job: Job<ChatterResult> | null = null;
  const prog = progressBar();
  const status = el('p', { class: 'status', text: 'Press Run. Constant α: the agent never stops adapting.' });
  const k = { drops: kpi('Checkpoints where r(π) fell > 0.02'), range: kpi('r(π) range, last half'), flips: kpi('Mean flipped mass per checkpoint', '—', 'μ*-weighted, last half') };
  const t0 = theme();

  const raster = canvasPanel('Greedy decision over time (rows = fixed sample of states, columns = checkpoints)', 300, canvas => {
    const r = results.get(L.kind);
    if (!r) return;
    const t = theme();
    drawHeatmap(canvas, {
      rows: r.sample.length, cols: r.raster.length,
      color: (row, col) => (r.raster[col]![row] ? t.accept : t.reject),
      rowLabel: row => (row % 4 === 0 ? r.sampleLabels[row]! : ''), rowLabelWidth: 128,
      colLabel: col => (col % 10 === 0 ? `${Math.round(((col + 1) * r.every) / 1000)}k` : ''), colLabelEvery: 1,
      xLabel: 'steps', textColor: t.text,
    });
  }, { legend: mapLegend() });
  const avgChart = canvasPanel('Exact r(π) of the greedy policy at each checkpoint', 220, canvas => {
    const t = theme();
    const series = [...results.entries()].map(([kind, r]) => ({ color: t.series[['coarse', 'default', 'tabular'].indexOf(kind)]!, points: r.avg.map((v, i) => [(i + 1) * r.every, v]), dash: kind === L.kind ? undefined : [4, 3] }));
    if (!series.length) return;
    const cur = results.get(L.kind);
    drawXY(canvas, series, { xLabel: 'steps', refs: cur ? [{ value: cur.rStar, color: t.ref, label: 'r*' }] : [], formatX: x => `${Math.round(x / 1000)}k`, textColor: t.text, gridColor: t.grid });
  }, { legend: legend([{ label: 'coarse tiles (G1 only, 2 tilings × 2 tiles)', color: t0.series[0]! }, { label: 'default tiles (G1 + G2)', color: t0.series[1]! }, { label: 'tabular (σ = 0, book cluster)', color: t0.series[2]! }, { label: 'r* of that cluster', color: t0.ref, dash: true }]) });
  const flipChart = canvasPanel('Flipped decisions since the previous checkpoint (μ*-weighted share of states)', 180, canvas => {
    const r = results.get(L.kind);
    if (!r) return;
    const t = theme();
    drawXY(canvas, [{ color: t.series[3]!, points: r.flips.map((v, i) => [(i + 1) * r.every, 100 * v]) }], { xLabel: 'steps', yMin: 0, formatY: v => `${fmt(v, 0)}%`, formatX: x => `${Math.round(x / 1000)}k`, textColor: t.text, gridColor: t.grid });
  });

  const render = () => {
    const r = results.get(L.kind);
    if (r) {
      const half = Math.floor(r.avg.length / 2);
      const tail = r.avg.slice(half);
      const flips = r.flips.slice(half);
      k.drops.set(String(r.drops), `of ${r.avg.length - 1} checkpoints`);
      k.range.set(`${fmt(Math.min(...tail), 3)} – ${fmt(Math.max(...tail), 3)}`, `r* = ${fmt(r.rStar, 3)} (exact, ${r.nStates} states)`);
      k.flips.set(`${fmt((100 * flips.reduce((a, b) => a + b, 0)) / flips.length, 1)}%`);
    }
    [raster, avgChart, flipChart].forEach(p => p.redraw());
  };

  const run = () => {
    job?.cancel();
    status.textContent = `Training (${L.kind}, α × active = ${L.alpha}, ε = ${L.eps})…`;
    job = ctx.job<ChatterResult>('chatter', { cfg, kind: L.kind, alpha: L.alpha, eps: L.eps, steps: L.steps, every: 5000, seed: cfg.seed }, f => prog.set(f, `${Math.round(f * 100)}%`));
    ctx.when(job.promise, r => {
      results.set(L.kind, r);
      prog.set(null);
      status.textContent = `${L.kind === 'tabular' ? 'Tabular on the σ = 0 book cluster' : `${L.kind} tiles on the current cluster`}: ${r.drops} of ${r.avg.length - 1} checkpoints made the greedy policy worse by more than 0.02.`;
      render();
    }, e => { status.textContent = `Failed: ${String(e)}`; prog.set(null); });
  };

  root.append(
    moduleHeader('m17', 'No policy improvement theorem: chattering', 'With tabular values, making one state\'s action greedier never hurts. With shared features, an update aimed at one state changes the values, and so the greedy decisions, of others. Nothing guarantees each new greedy policy is better. Run each feature setting and compare.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Run',
          select<ChatterFeatures>('Features', [{ value: 'coarse', label: 'Coarse tiles (G1, 2 tilings × 2 tiles)' }, { value: 'default', label: 'Default tiles' }, { value: 'tabular', label: 'Tabular (σ = 0 book cluster)' }], L.kind, v => { L.kind = v; render(); }).el,
          range('Constant α × active features', { min: 0.01, max: 0.5, step: 0.01, value: L.alpha, format: v => fmt(v, 2) }, v => { L.alpha = v; }).el,
          range('ε', { min: 0, max: 0.3, step: 0.01, value: L.eps, format: v => fmt(v, 2) }, v => { L.eps = v; }).el,
          range('Steps', { min: 100000, max: 1000000, step: 100000, value: L.steps, format: v => `${v / 1000}k` }, v => { L.steps = v; }).el,
          el('div', { class: 'buttons' }, button('Run', run, 'primary'), button('Cancel', () => { job?.cancel(); prog.set(null); })), prog.el)),
      el('div', { class: 'panels' }, status, el('div', { class: 'kpis' }, k.drops.el, k.range.el, k.flips.el), raster.el, avgChart.el, flipChart.el),
    ),
    observeBox([
      'Coarse tiles with α × active = 0.2: the raster keeps changing colour in whole blocks of states, and the exact r(π) swings. At the defaults (seed 1, 400k steps) it moved between about 2.03 and 2.48 in the second half, against r* = 2.81, and about 40% of checkpoints were worse than the one before. Updates aimed at a few states flip decisions everywhere their tiles reach.',
      'With function approximation, a greedy step need not be an improvement.',
      'Tabular (book cluster): far fewer states flip, and the policy stays near the optimum. It is not perfectly still with a constant α: noise flips a few near-tied states. Default tiles sit in between.',
      'Honest note: drops also happen in the tabular case at large α, so compare the size of the swings and the flipped mass, not just the count of drops.',
    ]),
  );
  render();
  return () => { job?.cancel(); [raster, avgChart, flipChart].forEach(p => p.destroy()); };
}
