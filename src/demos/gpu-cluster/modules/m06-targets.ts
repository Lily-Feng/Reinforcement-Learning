/**
 * M06 · Building G_{t:t+n}. One sampled trajectory with the n-step target
 * assembled term by term, and the distribution of targets for several n over
 * many trajectories, against the Monte Carlo estimate of q_π(s, a).
 */

import { drawBars, drawXY } from '../../shared/bars';
import { histogram } from '../core/stats.ts';
import type { TargetsResult } from '../core/jobs.ts';
import type { ModuleContext } from '../app/context';
import { canvasPanel, card, el, fmt, legend, moduleHeader, observeBox, progressBar, range, theme } from '../app/ui';

const NS: (number | null)[] = [1, 2, 4, 8, 16, null];

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = { ...ctx.cfg, amp: 0 };
  const L = ctx.store.local('m06', { n: 4 });
  let data: TargetsResult | null = null;
  const prog = progressBar();
  const status = el('p', { class: 'status', text: 'Sampling trajectories…' });
  const formula = el('p', { class: 'math' });

  const t0 = theme();
  const timeline = canvasPanel('One trajectory from (S_t, A_t): rewards R_{t+k} and the bootstrap q̂(S_{t+n}, A_{t+n})', 200, canvas => {
    if (!data) return;
    const t = theme();
    const ex = data.example;
    const n = Math.min(L.n, ex.rewards.length);
    drawBars(canvas, ex.rewards.map((r, k) => ({
      label: String(k + 1),
      highlight: k < n,
      segments: [{ value: r, color: k < n ? t.series[0]! : t.grid }],
    })), { textColor: t.text, gridColor: t.grid, xLabel: 'k (reward R_{t+k})', labelEvery: 4, highlightColor: 'rgba(124, 58, 237, 0.10)' });
  }, { legend: legend([{ label: 'rewards summed into G_{t:t+n}', color: t0.series[0]! }, { label: 'later rewards (replaced by the bootstrap)', color: t0.grid }]) });

  const hist = canvasPanel('Targets over many trajectories (frequency polygons)', 260, canvas => {
    if (!data) return;
    const t = theme();
    const all = data.targets.flat();
    const lo = Math.floor(Math.min(...all));
    const hi = Math.ceil(Math.max(...all));
    const bins = 40;
    const w = (hi - lo) / bins;
    const series = data.targets.map((xs, i) => ({
      color: t.series[i % 6]!,
      points: histogram(xs, lo, hi, bins).map((c, b) => [lo + (b + 0.5) * w, c / xs.length]),
    }));
    drawXY(canvas, series, {
      xLabel: 'target value', textColor: t.text, gridColor: t.grid, yMin: 0,
      vlines: [{ x: data.mc.mean, color: t.ref, label: 'MC q_π(s,a)', band: [data.mc.mean - data.mc.half, data.mc.mean + data.mc.half] }],
    });
  }, { legend: legend([...NS.map((n, i) => ({ label: n === null ? 'n = ∞ (Monte Carlo return)' : `n = ${n}`, color: t0.series[i % 6]! })), { label: 'Monte Carlo q_π(s, a) ± 95% CI', color: t0.ref, dash: true }]) });

  const table = el('table');
  const render = () => {
    if (!data) return;
    const ex = data.example;
    const n = Math.min(L.n, ex.rewards.length);
    const sum = ex.rewards.slice(0, n).reduce((a, b) => a + b, 0);
    const boot = n < ex.rewards.length ? ex.qhat[n]! : 0;
    formula.innerHTML = `G<sub>t:t+${n}</sub> = ${ex.rewards.slice(0, n).map(r => fmt(r, 0)).join(' + ')}${n < ex.rewards.length ? ` + q̂(S<sub>t+${n}</sub>, A<sub>t+${n}</sub>) = ${fmt(sum, 0)} + ${fmt(boot, 2)} = <strong>${fmt(sum + boot, 2)}</strong>` : ` = <strong>${fmt(sum, 2)}</strong> (the shift ends first: the full return)`}`;
    table.replaceChildren(
      el('thead', {}, el('tr', {}, ...['n', 'mean target', 'bias vs MC', 'std. dev.'].map(h => el('th', { text: h })))),
      el('tbody', {}, ...data.targets.map((xs, i) => {
        const m = xs.reduce((a, b) => a + b, 0) / xs.length;
        const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
        return el('tr', {}, el('td', { text: NS[i] === null ? '∞' : String(NS[i]) }), el('td', { text: fmt(m, 2) }), el('td', { text: fmt(m - data!.mc.mean, 2) }), el('td', { text: fmt(sd, 2) }));
      })));
    timeline.redraw();
    hist.redraw();
  };

  root.append(
    moduleHeader('m06', 'Building G_{t:t+n}', 'From one fixed state-action pair in the shift task, trajectories follow a fixed threshold policy. The n-step target sums n rewards and then bootstraps from a <em>deliberately imperfect</em> q̂ = 0.8·q_π + 5, built from the exact q_π of the same policy. Trajectories come from the bucketed simulator, so q_π is known exactly.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' }, card('Target', range('n', { min: 1, max: cfg.H, step: 1, value: L.n }, v => { L.n = v; render(); }).el, prog.el), card('Bias and spread', el('div', { class: 'table-scroll' }, table))),
      el('div', { class: 'panels' }, status, card(null, formula), timeline.el, hist.el),
    ),
    observeBox([
      'n = 1 targets are tightly packed but centred away from the Monte Carlo band: they inherit q̂\'s error.',
      'As n grows the centre moves onto the band (fewer bootstrapped terms) but the spread widens (more random rewards summed). n = ∞ is unbiased and widest.',
      'This is the bias–variance trade that M07 turns into learning speed.',
    ]),
  );

  const j = ctx.job<TargetsResult>('targets', { cfg, ns: NS, samples: 3000, seed: cfg.seed }, f => prog.set(f, `sampling ${Math.round(f * 100)}%`));
  ctx.when(j.promise, r => {
    data = r;
    prog.set(null);
    status.textContent = `State: ${r.stateLabel}. Monte Carlo q_π(s, a) = ${fmt(r.mc.mean, 2)} ± ${fmt(r.mc.half, 2)} (95% CI, 3,000 rollouts); exact on the bucketed model = ${fmt(r.exact, 2)}.`;
    render();
  }, e => { status.textContent = `Failed: ${String(e)}`; });
  return () => { timeline.destroy(); hist.destroy(); };
}
