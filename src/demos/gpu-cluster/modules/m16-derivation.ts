/**
 * M16 · The derivation J(π) = r(π)/(1 − γ), line by line, each line evaluated
 * numerically for the current threshold policy and γ. Replacing μ_π by a
 * uniform distribution shows which step needs the invariance of μ_π.
 */

import katex from 'katex';
import 'katex/dist/katex.min.css';
import { discountedValues, evaluatePolicy, stepFull } from '../core/exact.ts';
import { policyFrom, reward } from '../core/model.ts';
import { handTuned, thresholdOnModel, type Threshold } from '../core/policies.ts';
import type { ModuleContext } from '../app/context';
import { card, check, el, fmt, moduleHeader, observeBox, range } from '../app/ui';
import { onNextFrame, thresholdControls } from './common';

const LINES: { tex: string; note: string }[] = [
  { tex: String.raw`J(\pi) \doteq \sum_s \mu(s)\, v^\gamma_\pi(s)`, note: 'definition' },
  { tex: String.raw`= \sum_s \mu(s) \sum_a \pi(a\mid s) \sum_{s',r} p(s',r\mid s,a)\,\big[r + \gamma v^\gamma_\pi(s')\big]`, note: 'Bellman equation for v^γ_π' },
  { tex: String.raw`= r(\pi) + \gamma \sum_{s'} v^\gamma_\pi(s') \sum_s \mu(s) \sum_a \pi(a\mid s)\, p(s'\mid s,a)`, note: 'split the reward term; swap the sums' },
  { tex: String.raw`= r(\pi) + \gamma \sum_{s'} v^\gamma_\pi(s')\, \mu(s')`, note: 'invariance: Σ_s μ(s) Σ_a π(a|s) p(s′|s,a) = μ(s′)  ← needs μ = μ_π' },
  { tex: String.raw`= r(\pi) + \gamma J(\pi)`, note: 'definition of J again' },
  { tex: String.raw`\Rightarrow\ J(\pi) = \frac{r(\pi)}{1-\gamma}`, note: 'solve for J' },
];

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = { ...ctx.cfg, amp: 0 };
  const L = ctx.store.local<{ th: Threshold; gamma: number; uniform: boolean }>('m16', { th: handTuned(cfg.K), gamma: 0.9, uniform: false });
  if (Math.max(...L.th.theta) > cfg.K + 1) L.th = handTuned(cfg.K);
  const model = ctx.store.model(cfg);
  const list = el('ol', { class: 'derivation' });

  const render = onNextFrame(() => {
    const pol = policyFrom(model, thresholdOnModel(L.th));
    const ev = evaluatePolicy(model, pol);
    const dv = discountedValues(model, pol, L.gamma, ev);
    const d = L.uniform ? new Float64Array(model.nStates).fill(1 / model.nStates) : ev.mu;
    const dP = stepFull(model, pol, d);
    const g = L.gamma;
    let line1 = 0;
    let rd = 0;
    let line3sum = 0;
    let line4sum = 0;
    for (let s = 0; s < model.nStates; s += 1) {
      line1 += d[s]! * dv.v[s]!;
      rd += d[s]! * reward(model, s, pol[s]!);
      line3sum += dP[s]! * dv.v[s]!;
      line4sum += d[s]! * dv.v[s]!;
    }
    const values = [line1, rd + g * line3sum, rd + g * line3sum, rd + g * line4sum, rd + g * line1, rd / (1 - g)];
    const ref = values[0]!;
    let maxDiff = 0;
    for (let s = 0; s < model.nStates; s += 1) maxDiff = Math.max(maxDiff, Math.abs(dP[s]! - d[s]!));
    list.replaceChildren(...LINES.map((ln, i) => {
      const box = el('div', { class: 'tex' });
      katex.render(ln.tex, box, { displayMode: true, throwOnError: false });
      const broken = Math.abs(values[i]! - ref) > 1e-6 * Math.max(1, Math.abs(ref));
      return el('li', { class: `${i === 3 ? 'key ' : ''}${broken ? 'broken' : ''}` }, box,
        el('div', { class: 'num' }, el('strong', { text: fmt(values[i]!, 6) }), el('span', { class: 'hint', text: ` ${ln.note}${broken ? '  ✗ no longer equal to line 1' : ''}` })));
    }));
    summary.textContent = L.uniform
      ? `μ replaced by the uniform distribution over ${model.nStates} states: max_s |(μP_π)(s) − μ(s)| = ${fmt(maxDiff, 6)}, so line 4 no longer follows from line 3. Here r_μ = ${fmt(rd, 4)} is the uniform average of expected rewards.`
      : `μ = μ_π (exact, ${model.B}-bucket model): max_s |(μ_π P_π)(s) − μ_π(s)| = ${maxDiff.toExponential(1)}. r(π) = ${fmt(ev.avg, 6)}, J(π) = ${fmt(dv.J, 6)}, r(π)/(1 − γ) = ${fmt(ev.avg / (1 - g), 6)}.`;
  });
  const summary = el('p', { class: 'status' });

  root.append(
    moduleHeader('m16', 'The derivation, line by line, with numbers', 'Each line of Section 10.4\'s argument is evaluated for the current policy and γ. Every line should show the same number. The highlighted step is where the on-policy distribution earns its place.'),
    el('style', { text: `
      .derivation { margin: 0; padding-left: 1.4rem; display: flex; flex-direction: column; gap: .3rem; }
      .derivation li { padding: .2rem .5rem; border-radius: 8px; }
      .derivation li.key { background: var(--accent-bg); }
      .derivation li.broken .num strong { color: var(--danger); }
      .derivation .tex { overflow-x: auto; }
      .derivation .num { font-variant-numeric: tabular-nums; }
    ` }),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Discount and distribution',
          range('γ', { min: 0, max: 0.999, step: 0.001, value: L.gamma, format: v => fmt(v, 3) }, v => { L.gamma = v; render(); }).el,
          check('Replace μ_π with a uniform distribution', L.uniform, v => { L.uniform = v; render(); }).el),
        card('Threshold policy', thresholdControls(L.th, cfg.K, render, false))),
      el('div', { class: 'panels' }, summary, card(null, list)),
    ),
    observeBox([
      'With μ = μ_π every line agrees to rounding, for every γ, including γ = 0 where J = r(π).',
      'Switch to a uniform distribution: lines 1–3 still agree, since they are just the Bellman equation rearranged. Line 4 breaks, because the uniform distribution is not invariant under the policy\'s dynamics. Everything after it is wrong.',
      'So the result is a statement about the on-policy distribution. Weighting states any other way would let γ change the ranking.',
    ]),
  );
  render();
  return () => {};
}
