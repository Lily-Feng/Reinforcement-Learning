/**
 * M14 · γ cannot change the ranking. About twenty threshold policies are
 * evaluated exactly on the bucketed model. Averaged over each policy's own
 * on-policy distribution, J_γ(π)(1 − γ) = r(π) for every γ; ranked from one
 * fixed start state, the order can swap.
 */

import { drawBars, drawXY } from '../../shared/bars';
import type { FamilyResult } from '../core/jobs.ts';
import type { ModuleContext } from '../app/context';
import { canvasPanel, card, el, fmt, legend, moduleHeader, observeBox, progressBar, range, theme } from '../app/ui';

const GAMMAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999];

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = { ...ctx.cfg, amp: 0 };
  const L = ctx.store.local('m14', { gi: 9 });
  let data: FamilyResult | null = null;
  const prog = progressBar();
  const status = el('p', { class: 'status', text: 'Evaluating policies exactly…' });
  const gammaCtl = range('γ', { min: 0, max: GAMMAS.length - 1, step: 1, value: L.gi, format: i => String(GAMMAS[i]) }, i => { L.gi = i; render(); });

  const t0 = theme();
  const byR = canvasPanel('Ranked by r(π)', 220, canvas => {
    if (!data) return;
    const t = theme();
    const order = data.entries.map((e, i) => ({ e, i })).sort((a, b) => b.e.avg - a.e.avg);
    drawBars(canvas, order.map(({ e, i }) => ({ label: `#${i + 1}`, segments: [{ value: e.avg, color: t.series[0]! }] })), { textColor: t.text, gridColor: t.grid, formatY: v => fmt(v, 2) });
  });
  const byJ = canvasPanel('Ranked by J_γ(π) = Σ μ_π v^γ_π', 220, canvas => {
    if (!data) return;
    const t = theme();
    const order = data.entries.map((e, i) => ({ e, i })).sort((a, b) => b.e.J[L.gi]! - a.e.J[L.gi]!);
    drawBars(canvas, order.map(({ e, i }) => ({ label: `#${i + 1}`, segments: [{ value: e.J[L.gi]!, color: t.series[1]! }] })), { textColor: t.text, gridColor: t.grid, formatY: v => fmt(v, 1) });
  });
  const scatter = canvasPanel('J_γ(π)(1 − γ) against r(π)', 220, canvas => {
    if (!data) return;
    const t = theme();
    const g = GAMMAS[L.gi]!;
    drawXY(canvas, [{ color: t.series[1]!, line: false, r: 4, points: data.entries.map(e => [e.avg, e.J[L.gi]! * (1 - g)]) }], { diagonal: true, xLabel: 'r(π)', textColor: t.text, gridColor: t.grid });
  }, { legend: legend([{ label: 'one policy', color: t0.series[1]! }, { label: 'y = x', color: t0.text, dash: true }]) });
  const swapChart = canvasPanel('From one start state: (1 − γ)·v^γ_π(s₀) for a swapping pair', 220, canvas => {
    if (!data || !data.swaps.length) return;
    const t = theme();
    const sw = data.swaps[0]!;
    const pts = (i: number) => GAMMAS.map((g, k) => [k, (1 - g) * data!.entries[i]!.v0[k]!]);
    drawXY(canvas, [{ color: t.series[2]!, points: pts(sw.i), dots: true }, { color: t.series[3]!, points: pts(sw.j), dots: true }], {
      xLabel: 'γ', formatX: k => String(GAMMAS[Math.round(k)] ?? ''), vlines: [{ x: L.gi, color: t.ref }], textColor: t.text, gridColor: t.grid,
    });
  });
  const swapLegend = el('div');
  const table = el('table');
  const swapNote = el('p', { class: 'status' });

  const render = () => {
    if (!data) return;
    const g = GAMMAS[L.gi]!;
    const rankBy = (val: (i: number) => number) => {
      const idx = data!.entries.map((_, i) => i).sort((a, b) => val(b) - val(a));
      const rank = new Array<number>(idx.length);
      idx.forEach((i, r) => { rank[i] = r + 1; });
      return rank;
    };
    const rR = rankBy(i => data!.entries[i]!.avg);
    const rJ = rankBy(i => data!.entries[i]!.J[L.gi]!);
    const rV = rankBy(i => data!.entries[i]!.v0[L.gi]!);
    const rV0 = rankBy(i => data!.entries[i]!.v0[0]!);
    const involved = new Set(data.swaps.flatMap(s => [s.i, s.j]));
    table.replaceChildren(
      el('thead', {}, el('tr', {}, ...['#', 'policy', 'r(π)', `J(1−γ)`, 'rank by r', 'rank by J', 'rank by v^γ(s₀)', 'rank by v⁰(s₀)'].map((h, i) => el('th', { class: i === 1 ? 'l' : '', text: h })))),
      el('tbody', {}, ...data.entries.map((e, i) => el('tr', { class: rV[i] !== rV0[i] ? 'hl' : '' },
        el('td', { text: `#${i + 1}` }), el('td', { class: 'l', text: e.label }), el('td', { text: fmt(e.avg, 4) }), el('td', { text: fmt(e.J[L.gi]! * (1 - g), 4) }),
        el('td', { text: String(rR[i]) }), el('td', { text: String(rJ[i]) }), el('td', { text: String(rV[i]) }), el('td', { text: String(rV0[i]) })))));
    const same = rR.every((r, i) => r === rJ[i]);
    const maxErr = Math.max(...data.entries.map(e => Math.abs(e.J[L.gi]! * (1 - g) - e.avg)));
    status.textContent = `γ = ${g}: ranking by J_γ ${same ? 'is identical to' : 'DIFFERS from'} the ranking by r(π); max |J(1 − γ) − r(π)| = ${maxErr.toExponential(1)}. Start state s₀: ${data.startLabel}.`;
    if (data.swaps.length) {
      const sw = data.swaps[0]!;
      swapNote.textContent = `${data.swaps.length} pairs swap their order by v^γ(s₀) somewhere on the γ grid (e.g. #${sw.i + 1} and #${sw.j + 1} between γ = ${sw.gammaA} and γ = ${sw.gammaB}). Highlighted rows rank differently from s₀ at this γ than at γ = 0.`;
      swapLegend.replaceChildren(legend([{ label: `#${sw.i + 1} ${data.entries[sw.i]!.label}`, color: theme().series[2]! }, { label: `#${sw.j + 1} ${data.entries[sw.j]!.label}`, color: theme().series[3]! }]));
    } else swapNote.textContent = 'The search found no pair whose order by v^γ(s₀) changes on this γ grid, for this configuration.';
    void involved;
    [byR, byJ, scatter, swapChart].forEach(p => p.redraw());
  };

  root.append(
    moduleHeader('m14', 'γ cannot change the ranking', 'Each policy is evaluated exactly: r(π), and the discounted objective J_γ(π) = Σ_s μ_π(s) v^γ_π(s) weighted by the policy\'s <em>own</em> on-policy distribution. Section 10.4 shows J_γ(π) = r(π)/(1 − γ), so γ only rescales.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' }, card('Discount', gammaCtl.el, prog.el, el('p', { class: 'hint', text: 'γ moves on a grid that includes exactly 0 and goes up to 0.999. Policies are threshold rules (#1 accept-all … #20 short jobs only).' }))),
      el('div', { class: 'panels' }, status, el('div', { class: 'grid2' }, byR.el, byJ.el), el('div', { class: 'grid2' }, scatter.el, card(null, swapNote, swapLegend, swapChart.el))),
    ),
    card('All policies', el('div', { class: 'table-scroll' }, table)),
    observeBox([
      'Slide γ from 0 to 0.999: the two sorted bar charts keep the same order, and every point stays on the diagonal. At γ = 0, J is exactly r(π), because μ_π already encodes the policy\'s long-run effect.',
      'From a single start state the story changes. Policies that earn quickly from s₀ but sit in worse states later win at small γ and lose at large γ. That is the swapping pair.',
      'The theorem compares policies averaged over their own on-policy distributions. It does not say discounting is harmless inside an algorithm (see M15).',
    ]),
  );

  const j = ctx.job<FamilyResult>('family', { cfg, gammas: GAMMAS }, f => prog.set(f, `evaluating policies ${Math.round(f * 100)}%`));
  ctx.when(j.promise, r => { data = r; prog.set(null); render(); }, e => { status.textContent = `Failed: ${String(e)}`; });
  return () => { [byR, byJ, scatter, swapChart].forEach(p => p.destroy()); };
}
