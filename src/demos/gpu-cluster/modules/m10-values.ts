/**
 * M10 · Defining values when the sum oscillates. Partial sums of
 * E[R_{t+1} | S₀ = s] − r(π) oscillate on periodic chains; the Abel limit
 * f_s(γ) = v^γ(s) − r(π)/(1 − γ) converges as γ → 1. Then exact differential
 * values on the cluster.
 */

import katex from 'katex';
import 'katex/dist/katex.min.css';
import { drawXY } from '../../shared/bars';
import { alternating, averageReward, differentialValues, discountedValues, expectedRewardSequence, ring, type Chain } from '../core/toy-chains.ts';
import { differentialValues as clusterValues, evaluatePolicy } from '../core/exact.ts';
import { stateIndex } from '../core/model.ts';
import type { ModuleContext } from '../app/context';
import { canvasPanel, card, check, el, fmt, legend, moduleHeader, observeBox, select, theme } from '../app/ui';
import { bucketWords, typicalCounts } from './common';

const ALGEBRA: Record<string, string[]> = {
  alternating: [
    String.raw`\textstyle\sum_{t\ge0}\gamma^t \mathbb{E}[R_{t+1}\mid S_0=A] = 1 + \gamma^2 + \gamma^4 + \cdots = \frac{1}{1-\gamma^2}`,
    String.raw`f_A(\gamma) = \frac{1}{1-\gamma^2} - \frac{1/2}{1-\gamma} = \frac{2 - (1+\gamma)}{2(1-\gamma^2)} = \frac{1-\gamma}{2(1-\gamma)(1+\gamma)} = \frac{1}{2(1+\gamma)} \xrightarrow{\gamma\to1} \tfrac14`,
    String.raw`f_B(\gamma) = \frac{\gamma}{1-\gamma^2} - \frac{1/2}{1-\gamma} = \frac{2\gamma - 1 - \gamma}{2(1-\gamma^2)} = -\frac{1}{2(1+\gamma)} \xrightarrow{\gamma\to1} -\tfrac14`,
    String.raw`\text{(L'Hôpital on } \tfrac{1-\gamma}{2(1-\gamma^2)} \text{ gives the same: } \tfrac{-1}{-4\gamma}\big|_{\gamma=1} = \tfrac14.)`,
  ],
  ring: [
    String.raw`\textstyle\sum_t \gamma^t \mathbb{E}[R_{t+1}\mid S_0] = \frac{\gamma^{k}}{1-\gamma^3},\quad k_A = 2,\ k_B = 1,\ k_C = 0`,
    String.raw`f_s(\gamma) = \frac{\gamma^{k}}{1-\gamma^3} - \frac{1/3}{1-\gamma} = \frac{3\gamma^{k} - (1+\gamma+\gamma^2)}{3(1-\gamma^3)}`,
    String.raw`\text{numerator and denominator both vanish at } \gamma = 1;\ \text{L'Hôpital: } \frac{3k\gamma^{k-1} - (1 + 2\gamma)}{-9\gamma^2}\Big|_{\gamma=1} = \frac{3 - 3k}{9}`,
    String.raw`v(A) = -\tfrac13,\quad v(B) = 0,\quad v(C) = \tfrac13`,
  ],
};

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const L = ctx.store.local('m10', { chain: 'alternating' as 'alternating' | 'ring', algebra: false, tier: 0 });
  const t0 = theme();
  let chain: Chain = alternating();
  const gammas = Array.from({ length: 60 }, (_, i) => 1 - 10 ** (-0.3 - (i * 3.2) / 59));

  const partial = canvasPanel('Partial sums Σ_{t<T} (E[R_{t+1} | S₀ = s] − r(π)): no limit', 200, canvas => {
    const t = theme();
    const r = averageReward(chain);
    drawXY(canvas, chain.states.map((_, s) => {
      const seq = expectedRewardSequence(chain, s, 24);
      let sum = 0;
      return { color: t.series[s]!, dots: true, r: 2, points: seq.map((x, k) => { sum += x - r; return [k + 1, sum]; }) };
    }), { xLabel: 'T', textColor: t.text, gridColor: t.grid });
  });
  const abel = canvasPanel('f_s(γ) = v^γ(s) − r(π)/(1 − γ) as γ → 1', 200, canvas => {
    const t = theme();
    const r = averageReward(chain);
    const limits = differentialValues(chain);
    drawXY(canvas, chain.states.map((_, s) => ({ color: t.series[s]!, points: gammas.map(g => [-Math.log10(1 - g), discountedValues(chain, g)[s]! - r / (1 - g)]) })), {
      xLabel: '−log₁₀(1 − γ)   (γ = 0.5 … 0.9997)', refs: limits.map((v, s) => ({ value: v, color: t.series[s]!, dash: [5, 4] })), textColor: t.text, gridColor: t.grid,
    });
  });
  const diverge = canvasPanel('v^γ(s) and r(π)/(1 − γ) both diverge; their difference does not', 200, canvas => {
    const t = theme();
    const r = averageReward(chain);
    drawXY(canvas, [
      ...chain.states.map((_, s) => ({ color: t.series[s]!, points: gammas.map(g => [-Math.log10(1 - g), Math.log10(Math.max(1e-9, discountedValues(chain, g)[s]!))]) })),
      { color: t.ink, dash: [4, 3], points: gammas.map(g => [-Math.log10(1 - g), Math.log10(r / (1 - g))]) },
    ], { xLabel: '−log₁₀(1 − γ)', formatY: v => `10^${fmt(v, 1)}`, textColor: t.text, gridColor: t.grid });
  });
  const chainLegend = el('div');
  const algebra = el('div', { class: 'math' });

  // --- cluster -------------------------------------------------------------
  const cfg = { ...ctx.cfg, amp: 0 };
  const model = ctx.store.model(cfg);
  let clusterReady = false;
  let h: Float64Array | null = null;
  let typical: number[] = [];
  const clusterNote = el('p', { class: 'hint', text: 'Computing the bucketed optimum…' });
  const clusterPanel = canvasPanel('Exact differential values h(s) of the optimal policy: k free GPUs when a request arrives', 230, canvas => {
    if (!clusterReady || !h) return;
    const t = theme();
    const words = bucketWords(model.B);
    drawXY(canvas, Array.from({ length: model.B }, (_, b) => ({
      color: t.bucket[Math.min(3, b + (model.B === 1 ? 1 : 0))]!, dots: true,
      points: Array.from({ length: cfg.K + 1 }, (_, g) => [g, typical[g]! >= 0 ? h![stateIndex(model, typical[g]!, L.tier, b)]! : NaN] as number[]).filter(p => Number.isFinite(p[1])),
    })), { xLabel: 'free GPUs g', textColor: t.text, gridColor: t.grid });
    clusterLegend.replaceChildren(legend(words.map((w, b) => ({ label: `${w} request (ℓ ≈ ${model.buckets.len[b]!.toFixed(0)})`, color: t.bucket[Math.min(3, b + (model.B === 1 ? 1 : 0))]! }))));
  });
  const clusterLegend = el('div');

  const update = () => {
    chain = L.chain === 'alternating' ? alternating() : ring();
    chainLegend.replaceChildren(legend([...chain.states.map((s, i) => ({ label: `s = ${s}`, color: t0.series[i]! })), { label: 'dashed: the limit (differential value)', color: t0.text, dash: true }]));
    algebra.hidden = !L.algebra;
    algebra.replaceChildren(...(ALGEBRA[L.chain] ?? []).map(tex => { const d = el('div'); katex.render(tex, d, { displayMode: true, throwOnError: false }); return d; }));
    [partial, abel, diverge].forEach(p => p.redraw());
  };

  root.append(
    moduleHeader('m10', 'Defining values when the sum oscillates', 'For a periodic chain the differential return has no limit. Discount it, so it converges, then let γ → 1: the Abel limit f_s(γ) = v^γ(s) − r(π)/(1 − γ) defines the differential value. When the ordinary limit exists, the two definitions agree.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Chain', select('Chain', [{ value: 'alternating', label: 'Alternating A ↔ B (r = 1/2)' }, { value: 'ring', label: 'Ring A → B → C (r = 1/3)' }], L.chain, v => { L.chain = v as 'alternating' | 'ring'; update(); }).el,
          check('Show algebra', L.algebra, v => { L.algebra = v; update(); }).el),
        card('Cluster', select('Request tier', ['Free', 'Basic', 'Pro', 'Enterprise'].map((n, c) => ({ value: String(c), label: n })), String(L.tier), v => { L.tier = Number(v); clusterPanel.redraw(); }).el, clusterNote)),
      el('div', { class: 'panels' }, chainLegend, algebra, el('div', { class: 'grid2' }, partial.el, abel.el), diverge.el, card(null, clusterLegend, clusterPanel.el)),
    ),
    observeBox([
      'Partial sums oscillate (between 1/2 and 0 from A on the alternating chain), so the plain differential return has no limit.',
      'f_s(γ) settles onto the dashed limits: 1/4 and −1/4 for the alternating chain; −1/3, 0 and 1/3 for the ring.',
      'Both v^γ(s) and r(π)/(1 − γ) blow up as γ → 1, at the same rate. The part they share is the same for every state, so only the difference carries information.',
      'Cluster: h(s) says how much better than average it is to be in s. It rises with free GPUs, from about −10 with none free to about +12 with all free at the defaults: free capacity is worth future revenue.',
      'Switch the tier: an Enterprise request arriving is worth roughly 7 more than a Free one at the same g. For Enterprise, a long job is worth less than a short one when few GPUs are free. For a Free request the length barely changes h: the optimum never accepts a long Free job, and a short one adds little.',
    ]),
  );
  update();

  ctx.when(ctx.store.reference(cfg), ref => {
    const ev = evaluatePolicy(model, ref.policy);
    h = clusterValues(model, ref.policy, ev).h;
    typical = typicalCounts(model, ref.nu);
    clusterReady = true;
    clusterNote.textContent = `Optimal policy (r* = ${fmt(ref.rStar, 3)}, exact on ${ref.B}-bucket model). Values normalized so Σ μ h = 0; busy jobs at their most likely mix for each g.`;
    clusterPanel.redraw();
  });
  return () => { [partial, abel, diverge, clusterPanel].forEach(p => p.destroy()); };
}
