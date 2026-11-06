/**
 * M11 · Values are defined up to a constant. Exact h_π for a threshold
 * policy; adding c to every value leaves every TD error and Bellman residual
 * unchanged. Then learned values from the M13 agent against exact optimal
 * differential values, with the fitted offset.
 */

import { drawXY } from '../../shared/bars';
import { mulberry32 } from '../../shared/rng';
import { ACCEPT, TIER_NAMES } from '../core/config.ts';
import { bucketCounts, createEnv, observe, step } from '../core/env.ts';
import { applyT, differentialValues, evaluatePolicy } from '../core/exact.ts';
import { learnedValues } from '../core/evaluate.ts';
import { afterstate, countKey, policyFrom, reward, stateIndex } from '../core/model.ts';
import { handTuned, thresholdDecide, thresholdOnModel, type Threshold } from '../core/policies.ts';
import { configKey } from '../core/config.ts';
import type { ModuleContext } from '../app/context';
import { button, canvasPanel, card, el, fmt, legend, moduleHeader, observeBox, range, select, theme } from '../app/ui';
import { bucketWords, onNextFrame, thresholdControls, typicalCounts } from './common';

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = { ...ctx.cfg, amp: 0 };
  const store = ctx.store;
  const L = store.local<{ th: Threshold; c: number; tier: number }>('m11', { th: handTuned(cfg.K), c: 0, tier: 0 });
  if (Math.max(...L.th.theta) > cfg.K + 1) L.th = handTuned(cfg.K);
  const model = store.model(cfg);
  const t0 = () => theme();
  const index = new Map(model.counts.map((n, i) => [countKey(n), i]));
  let pol = policyFrom(model, thresholdOnModel(L.th));
  let ev = evaluatePolicy(model, pol);
  let dv = differentialValues(model, pol, ev);
  let typical = typicalCounts(model, ev.nu);

  // A fixed sample of transitions from the bucketed simulator under the policy.
  const sampleTransitions = () => {
    const env = createEnv(cfg, mulberry32(cfg.seed + 41), { mode: 'bucketed' });
    for (let i = 0; i < 300; i += 1) step(env, thresholdDecide(L.th, observe(env)));
    const out: { s: number; a: number; r: number; s2: number }[] = [];
    for (let i = 0; i < 8; i += 1) {
      const o = observe(env);
      const s = stateIndex(model, index.get(countKey(bucketCounts(env)))!, o.c, o.bucket);
      const a = pol[s]!;
      const r = step(env, a).reward;
      const o2 = observe(env);
      out.push({ s, a, r, s2: stateIndex(model, index.get(countKey(bucketCounts(env)))!, o2.c, o2.bucket) });
    }
    return out;
  };
  let samples = sampleTransitions();

  const lines = canvasPanel('h_π(s) vs. free GPUs, per head length bucket (busy jobs at their most likely mix)', 230, canvas => {
    const t = theme();
    drawXY(canvas, Array.from({ length: model.B }, (_, b) => ({
      color: t.bucket[Math.min(3, b + (model.B === 1 ? 1 : 0))]!, dots: true,
      points: Array.from({ length: cfg.K + 1 }, (_, g) => [g, typical[g]! >= 0 ? dv.h[stateIndex(model, typical[g]!, L.tier, b)]! + L.c : NaN]).filter(p => Number.isFinite(p[1]!)),
    })), { xLabel: 'free GPUs g', textColor: t.text, gridColor: t.grid });
  });
  const linesLegend = el('div');
  const table = el('table');
  const TH = () => applyT(model, dv.hbar.map(v => v + L.c));
  const renderTable = () => {
    const shiftedTH = TH();
    const baseTH = applyT(model, dv.hbar);
    table.replaceChildren(
      el('thead', {}, el('tr', {}, ...['s → s′', 'a', 'r', 'h(s) + c', "h(s′) + c", 'TD error δ = r − r(π) + h(s′) − h(s)', 'Bellman residual'].map((x, i) => el('th', { class: i === 0 ? 'l' : '', text: x })))),
      el('tbody', {}, ...samples.map(({ s, a, r, s2 }) => {
        const hs = dv.h[s]! + L.c;
        const hs2 = dv.h[s2]! + L.c;
        const delta = r - ev.avg + hs2 - hs;
        // Expected-update residual: h(s) − [E r − r(π) + Σ p h(s′)] using the shifted values.
        const resid = hs - (reward(model, s, a) - ev.avg + shiftedTH[afterstate(model, s, a)]!);
        const resid0 = dv.h[s]! - (reward(model, s, a) - ev.avg + baseTH[afterstate(model, s, a)]!);
        const o = { g: model.free[Math.floor(s / (4 * model.B))]!, c: Math.floor(s / model.B) % 4 };
        return el('tr', {}, el('td', { class: 'l', text: `g=${o.g}, ${TIER_NAMES[o.c]} → …` }), el('td', { text: a === ACCEPT ? 'acc' : 'rej' }), el('td', { text: fmt(r, 1) }),
          el('td', { text: fmt(hs, 3) }), el('td', { text: fmt(hs2, 3) }), el('td', { text: fmt(delta, 4) }), el('td', { text: `${resid.toExponential(1)} (c=0: ${resid0.toExponential(1)})` }));
      })));
  };

  // --- learned vs exact ------------------------------------------------------
  const scatterNote = el('p', { class: 'status' });
  let scatterPts: number[][] = [];
  const scatter = canvasPanel('Learned q̂ (M13 agent) vs. exact optimal differential values q*(s, a*)', 260, canvas => {
    const t = theme();
    if (!scatterPts.length) return;
    drawXY(canvas, [{ color: t.series[0]!, line: false, r: 2.5, points: scatterPts }], { diagonal: true, xLabel: 'exact q*(s, a*) (bucketed model)', textColor: t.text, gridColor: t.grid });
  }, { legend: legend([{ label: 'one bucketed state (μ* > 10⁻⁴), after removing the fitted offset', color: t0().series[0]! }, { label: 'y = x', color: t0().text, dash: true }]) });
  const useLab = () => {
    const lab = store.labAgent;
    if (!lab) { scatterNote.textContent = 'No trained agent yet: train one in M13 (Play or Fast-forward), then come back.'; return; }
    if (configKey(lab.cfg) !== configKey(ctx.cfg)) { scatterNote.textContent = 'The M13 agent was trained on a different cluster configuration. Retrain it on the current settings.'; return; }
    ctx.when(store.reference(cfg), ref => {
      const q = learnedValues(model, lab.agent, s => ref.policy[s]!);
      const TU = applyT(model, ref.U);
      const pts: number[][] = [];
      let wsum = 0;
      let off = 0;
      for (let s = 0; s < model.nStates; s += 1) {
        if (ref.mu[s]! < 1e-4) continue;
        const a = ref.policy[s]!;
        const exact = reward(model, s, a) - ref.rStar + TU[afterstate(model, s, a)]!;
        pts.push([exact, q[s]!]);
        off += ref.mu[s]! * (q[s]! - exact);
        wsum += ref.mu[s]!;
      }
      off /= wsum;
      scatterPts = pts.map(([x, y]) => [x, y! - off]);
      const resid = Math.sqrt(pts.reduce((acc, [x, y]) => acc + (y! - off - x!) ** 2, 0) / pts.length);
      scatterNote.textContent = `${lab.label}. Fitted offset (μ*-weighted mean of q̂ − q*) = ${fmt(off, 3)}; RMS difference after removing it = ${fmt(resid, 3)} over ${pts.length} states. The learned values match the exact ones up to that constant (and approximation error).`;
      scatter.redraw();
    });
  };

  const recompute = onNextFrame(() => {
    pol = policyFrom(model, thresholdOnModel(L.th));
    ev = evaluatePolicy(model, pol);
    dv = differentialValues(model, pol, ev);
    typical = typicalCounts(model, ev.nu);
    samples = sampleTransitions();
    status.textContent = `r(π) = ${fmt(ev.avg, 4)} (exact, ${model.B}-bucket model). h_π is normalized so Σ μ_π h_π = 0; the slider then adds c to every value.`;
    renderTable();
    lines.redraw();
  });
  const status = el('p', { class: 'status' });
  linesLegend.replaceChildren(legend(bucketWords(model.B).map((w, b) => ({ label: `${w} request`, color: t0().bucket[Math.min(3, b + (model.B === 1 ? 1 : 0))]! }))));

  root.append(
    moduleHeader('m11', 'Values are defined up to a constant', 'Differential Bellman equations and TD errors only involve <em>differences</em> of values, so h and h + c are equally good solutions. The book normalizes with Σ μ h = 0. A learning algorithm settles on whatever constant its initialization and updates lead to.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Shift every value', range('c added to every h(s)', { min: -20, max: 20, step: 0.5, value: L.c, format: v => fmt(v, 1) }, v => { L.c = v; renderTable(); lines.redraw(); }).el,
          select('Head tier for the plot', TIER_NAMES.map((n, c) => ({ value: String(c), label: n })), String(L.tier), v => { L.tier = Number(v); lines.redraw(); }).el),
        card('Threshold policy', thresholdControls(L.th, cfg.K, recompute, false))),
      el('div', { class: 'panels' }, status, card(null, linesLegend, lines.el),
        card('Sampled transitions (bucketed simulator)', el('div', { class: 'table-scroll' }, table)),
        card('Learned vs. exact', button('Use the agent from the lab (M13)', useLab, 'primary'), scatterNote, scatter.el)),
    ),
    observeBox([
      'Drag c: every h(s) and h(s′) moves, but each TD error stays the same to the last digit, and the Bellman residuals stay at rounding level.',
      'So the data cannot pin down the constant. The normalization Σ μ h = 0 is a convention, the one produced by formula 10.13.',
      'The M13 agent\'s values line up with the exact ones once a single offset is removed. That offset is arbitrary: it depends on q₀, the step sizes and the trajectory.',
    ]),
  );
  recompute();
  return () => { lines.destroy(); scatter.destroy(); };
}
