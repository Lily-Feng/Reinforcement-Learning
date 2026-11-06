/**
 * M07 · Which n learns fastest? A worker sweep over n and α (Figure 10.4
 * analogue) and best-α learning curves (Figure 10.3 analogue).
 */

import { drawXY } from '../../shared/bars';
import type { SweepResult } from '../core/jobs.ts';
import type { ModuleContext } from '../app/context';
import type { Job } from '../app/jobs';
import { button, canvasPanel, card, el, fmt, legend, moduleHeader, observeBox, progressBar, range, theme } from '../app/ui';

const NS = [1, 2, 4, 8, 16];
const ALPHAS = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8];

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = ctx.cfg;
  const L = ctx.store.local('m07', { runs: 8, shifts: 50, q0: 0, eps: 0.1 });
  let data: SweepResult | null = null;
  let job: Job<SweepResult> | null = null;
  const prog = progressBar();
  const status = el('p', { class: 'status', text: 'Press Run sweep. It trains 5 × 6 × runs agents in the worker.' });
  const t0 = theme();

  const sweep = canvasPanel('Mean revenue per shift over the first shifts vs. α × active features', 260, canvas => {
    if (!data) return;
    const t = theme();
    drawXY(canvas, data.ns.map((_, ni) => ({ color: t.series[ni]!, dots: true, points: data!.alphas.map((a, ai) => [a, data!.perf[ni]![ai]!.mean, data!.perf[ni]![ai]!.half]) })), {
      logX: true, xLabel: 'α × active features (log scale)', formatX: v => String(v), textColor: t.text, gridColor: t.grid,
    });
  }, { legend: legend(NS.map((n, i) => ({ label: `n = ${n}`, color: t0.series[i]! }))) });
  const curves = canvasPanel('Learning curves at each n\'s best α (n = 1 vs. n = 8)', 220, canvas => {
    if (!data) return;
    const t = theme();
    const pick = data.curves.filter(c => c.n === 1 || c.n === 8);
    drawXY(canvas, pick.map(c => ({ color: t.series[NS.indexOf(c.n)]!, points: c.points.map((v, k) => [k + 1, v]) })), { xLabel: 'shift', textColor: t.text, gridColor: t.grid });
  });
  const curveLegend = el('div');
  const table = el('table');

  const render = () => {
    if (!data) return;
    curveLegend.replaceChildren(legend(data.curves.filter(c => c.n === 1 || c.n === 8).map(c => ({ label: `n = ${c.n}, best α = ${c.alpha}`, color: theme().series[NS.indexOf(c.n)]! }))));
    table.replaceChildren(
      el('thead', {}, el('tr', {}, el('th', { text: 'n \\ α' }), ...data.alphas.map(a => el('th', { text: String(a) })))),
      el('tbody', {}, ...data.ns.map((n, ni) => {
        const best = Math.max(...data!.perf[ni]!.map(p => p.mean));
        return el('tr', {}, el('th', { text: String(n) }), ...data!.perf[ni]!.map(p => el('td', { class: p.mean === best ? 'hl' : '', text: `${fmt(p.mean, 1)} ± ${fmt(p.half, 1)}` })));
      })));
    sweep.redraw();
    curves.redraw();
  };

  const run = () => {
    job?.cancel();
    data = null;
    status.textContent = 'Running…';
    job = ctx.job<SweepResult>('sweep', { cfg, ns: NS, alphas: ALPHAS, runs: L.runs, shifts: L.shifts, seed: cfg.seed, q0: L.q0, eps: L.eps }, (f, note) => prog.set(f, `${Math.round(f * 100)}% · ${note ?? ''}`));
    ctx.when(job.promise, r => {
      data = r;
      prog.set(null);
      const bestN = r.ns.map((n, ni) => ({ n, v: Math.max(...r.perf[ni]!.map(p => p.mean)) })).sort((a, b) => b.v - a.v)[0]!;
      status.textContent = `Best n at its best α in this run: n = ${bestN.n} (${fmt(bestN.v, 1)} per shift). Error bars: 95% CI over ${L.runs} runs.`;
      render();
    }, e => { status.textContent = `Failed: ${String(e)}`; prog.set(null); });
  };
  const cancel = () => { job?.cancel(); prog.set(null); status.textContent = 'Cancelled.'; };

  root.append(
    moduleHeader('m07', 'Which n learns fastest?', 'Episodic n-step semi-gradient Sarsa on the shift task. Performance is the mean revenue per shift over the first shifts of learning (the area under the learning curve), averaged over independent runs. ε = 0.1 by default.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Sweep',
          range('Runs per (n, α)', { min: 3, max: 20, step: 1, value: L.runs }, v => { L.runs = v; }).el,
          range('Shifts per run', { min: 20, max: 100, step: 10, value: L.shifts }, v => { L.shifts = v; }).el,
          range('q₀', { min: 0, max: 200, step: 10, value: L.q0 }, v => { L.q0 = v; }).el,
          range('ε', { min: 0, max: 0.2, step: 0.01, value: L.eps, format: v => fmt(v, 2) }, v => { L.eps = v; }).el,
          el('div', { class: 'buttons' }, button('Run sweep', run, 'primary'), button('Cancel', cancel)), prog.el),
        card('Results', el('div', { class: 'table-scroll' }, table))),
      el('div', { class: 'panels' }, status, sweep.el, card(null, curveLegend, curves.el)),
    ),
    observeBox([
      'Exercise 10.3: the error bars grow with n. Longer targets sum more random rewards, so runs disagree more.',
      'Larger n tends to want a smaller α. The large-n curves fall off sooner at the right of the plot.',
      '<strong>Honest finding:</strong> in this task n = 1 usually learns fastest, unlike Mountain Car where n = 4–8 wins. We tried q₀ ∈ {0, 100, 160}, ε ∈ {0, 0.1} and α from 0.025 to 1.2 (8 runs, 50 shifts); n = 1 or 2 was best each time. Rewards here are immediate and dense, and the queue is very noisy, so bootstrapping early helps more than seeing further ahead.',
    ]),
  );
  return () => { job?.cancel(); sweep.destroy(); curves.destroy(); };
}
