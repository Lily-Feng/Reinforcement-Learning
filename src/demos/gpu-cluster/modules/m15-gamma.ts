/**
 * M15 · γ as an algorithm parameter. Discounted Sarsa at four γ and
 * differential Sarsa on identical seeds in the continuous simulator; every
 * learned greedy policy is scored exactly on the bucketed model.
 */

import { drawBars, drawXY } from '../../shared/bars';
import { configKey, TIER_NAMES } from '../core/config.ts';
import { evaluatePolicy } from '../core/exact.ts';
import { greedyOnModel } from '../core/evaluate.ts';
import type { CompareEntry, Reference } from '../core/jobs.ts';
import type { ModuleContext } from '../app/context';
import type { Job } from '../app/jobs';
import { button, canvasPanel, card, el, fmt, legend, moduleHeader, observeBox, progressBar, range, select, theme } from '../app/ui';
import { decisionMap, mapLegend, type MapPanel } from './common';

const GAMMAS = [0, 0.5, 0.9, 0.99];

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = { ...ctx.cfg, amp: 0 };
  const store = ctx.store;
  const L = store.local('m15', { steps: 200000, alpha: 0.1, tier: 1 });
  const model = store.model(cfg);
  let data: CompareEntry[] | null = null;
  let ref: Reference | null = null;
  let labScore: { label: string; avg: number } | null = null;
  let job: Job<CompareEntry[]> | null = null;
  const prog = progressBar();
  const status = el('p', { class: 'status', text: 'Press Train to run five agents on identical seeds in the worker.' });

  const t0 = theme();
  const bars = canvasPanel('Exact r(π) of each learned greedy policy (bucketed model)', 240, canvas => {
    if (!data) return;
    const t = theme();
    const rows = [...data.map(d => ({ label: d.gamma === null ? 'diff.' : `γ=${d.gamma}`, avg: d.avg })), ...(labScore ? [{ label: 'M13', avg: labScore.avg }] : [])];
    drawBars(canvas, rows.map((r, i) => ({ label: r.label, segments: [{ value: r.avg, color: i === rows.length - 1 && labScore ? t.series[5]! : i === data!.length - 1 ? t.series[2]! : t.series[0]! }] })), {
      textColor: t.text, gridColor: t.grid, formatY: v => fmt(v, 2), refs: ref ? [{ value: ref.rStar, color: t.ref, label: 'r*' }, { value: ref.acceptAll, color: t.text, label: 'accept-all' }] : [],
    });
  });
  const curves = canvasPanel('Exact r(greedy) during training', 220, canvas => {
    if (!data) return;
    const t = theme();
    drawXY(canvas, data.map((d, i) => ({ color: t.series[i]!, points: d.curve.map((v, k) => [((k + 1) * L.steps) / d.curve.length, v]) })), {
      xLabel: 'steps', refs: ref ? [{ value: ref.rStar, color: t.ref, label: 'r*' }] : [], textColor: t.text, gridColor: t.grid,
      formatX: x => `${Math.round(x / 1000)}k`,
    });
  }, { legend: legend([...GAMMAS.map((g, i) => ({ label: `discounted γ = ${g}`, color: t0.series[i]! })), { label: 'differential', color: t0.series[4]! }]) });

  const mapsBox = el('div', { class: 'grid3' });
  let maps: MapPanel[] = [];
  const buildMaps = () => {
    maps.forEach(m => m.destroy());
    if (!data) { mapsBox.replaceChildren(); maps = []; return; }
    maps = data.map(d => decisionMap(`${d.label} · ${TIER_NAMES[L.tier]}`, () => cfg.K, d.map.lengths, (g, li) => d.map.accept[L.tier]![g]![li]!, { height: 150 }));
    mapsBox.replaceChildren(...maps.map(m => m.el));
  };

  const train = () => {
    job?.cancel();
    status.textContent = 'Training…';
    job = ctx.job<CompareEntry[]>('compare', { cfg, gammas: GAMMAS, steps: L.steps, alpha: L.alpha, seed: cfg.seed }, (f, note) => prog.set(f, `${Math.round(f * 100)}% · ${note ?? ''}`));
    ctx.when(job.promise, r => {
      data = r;
      prog.set(null);
      const best = [...r].sort((a, b) => b.avg - a.avg)[0]!;
      status.textContent = `${L.steps.toLocaleString()} steps each, α × active = ${L.alpha}, ε = 0.1, same seed. Best: ${best.label} (${fmt(best.avg, 3)}${ref ? `, ${fmt((100 * best.avg) / ref.rStar, 1)}% of r*` : ''}).`;
      buildMaps();
      bars.redraw();
      curves.redraw();
    }, e => { status.textContent = `Failed: ${String(e)}`; prog.set(null); });
  };
  const labNote = el('p', { class: 'hint' });
  const useLab = () => {
    const lab = store.labAgent;
    if (!lab) { labNote.textContent = 'Train an agent in M13 first.'; return; }
    if (configKey(lab.cfg) !== configKey(ctx.cfg)) { labNote.textContent = 'The M13 agent was trained on a different configuration.'; return; }
    labScore = { label: lab.label, avg: evaluatePolicy(model, greedyOnModel(model, lab.agent)).avg };
    labNote.textContent = `${lab.label}: exact r(greedy) = ${fmt(labScore.avg, 3)} (purple bar).`;
    bars.redraw();
  };

  root.append(
    moduleHeader('m15', 'γ as an algorithm parameter', 'Section 10.4 removes γ from the <em>problem</em>, but it can still be a <em>solution-method</em> parameter. Here discounted semi-gradient Sarsa at four values of γ and differential Sarsa learn on the same seed in the continuous simulator. Each learned greedy policy is then scored exactly.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Train', range('Steps per agent', { min: 50000, max: 600000, step: 50000, value: L.steps, format: v => `${v / 1000}k` }, v => { L.steps = v; }).el,
          range('α × active features', { min: 0.02, max: 0.4, step: 0.02, value: L.alpha, format: v => fmt(v, 2) }, v => { L.alpha = v; }).el,
          el('div', { class: 'buttons' }, button('Train', train, 'primary'), button('Cancel', () => { job?.cancel(); prog.set(null); status.textContent = 'Cancelled.'; })), prog.el),
        card('Compare with the lab', button('Use the agent from the lab (M13)', useLab), labNote),
        card('Maps', select('Tier', TIER_NAMES.map((n, c) => ({ value: String(c), label: n })), String(L.tier), v => { L.tier = Number(v); buildMaps(); }).el, mapLegend())),
      el('div', { class: 'panels' }, status, el('div', { class: 'grid2' }, bars.el, curves.el), mapsBox),
    ),
    observeBox([
      'γ = 0 learns only the immediate reward: accepting always pays more than rejecting now, so it accepts everything and scores accept-all\'s r(π).',
      'Larger γ helps, but not monotonically. γ = 0.99 makes values large and slow to learn at this step size, so γ has to be tuned like any hyperparameter.',
      'Differential Sarsa needs no γ and here scores best. The discounted algorithms are not optimizing J_γ (or r(π)) either: they are semi-gradient methods with their own fixed points.',
    ]),
  );
  ctx.when(store.reference(cfg), r => { ref = r; bars.redraw(); curves.redraw(); });
  return () => { job?.cancel(); bars.destroy(); curves.destroy(); maps.forEach(m => m.destroy()); };
}
