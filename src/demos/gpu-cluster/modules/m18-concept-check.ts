/**
 * M18 · Check yourself: true/false and numeric cards, each revealing an
 * explanation and a link to the module that shows it.
 */

import { parseNumber } from '../core/stats.ts';
import type { ModuleContext } from '../app/context';
import { button, card, el, moduleHeader } from '../app/ui';

type Question =
  | { kind: 'tf'; text: string; answer: boolean; why: string; link?: string }
  | { kind: 'num'; text: string; answers: { label: string; value: number }[]; why: string; link?: string };

const QUESTIONS: Question[] = [
  { kind: 'tf', text: 'In the average-reward setting, every policy that attains the maximal r(π) is optimal.', answer: true, why: 'This is gain optimality: only the long-run rate counts, so two policies that differ only in transient behaviour are both optimal. That is usually adequate in practice.', link: 'm08' },
  { kind: 'tf', text: 'Averaged over each policy\'s own on-policy distribution, policies are ranked the same with or without discounting.', answer: true, why: 'J_γ(π) = r(π)/(1 − γ), a positive rescaling of r(π). It does not say rankings from a single start state are unchanged (they can swap), nor that discounted algorithms optimize J.', link: 'm14' },
  { kind: 'tf', text: 'γ = 0 is allowed in the objective J_γ, and it is also a good choice inside a learning algorithm.', answer: false, why: 'In the objective, γ = 0 is fine: J = r(π) because μ_π already accounts for the future. In an algorithm, γ = 0 makes the agent myopic: it accepts every request and scores like accept-all.', link: 'm15' },
  { kind: 'tf', text: 'Discounted control with function approximation is hard because the policy improvement theorem no longer holds.', answer: true, why: 'With shared parameters, making one state greedier can make others worse, so ε-greedy improvement can chatter. The same gap exists in the episodic and average-reward settings; policy-gradient methods (Chapter 13) are a partial answer.', link: 'm17' },
  { kind: 'tf', text: 'The agent\'s observation (g, c, ℓ, ρ, φ) is a Markov state of the cluster.', answer: false, why: 'Different per-GPU job lists can share g and ρ = Σ q_i but free GPUs at different rates. The observation is a lossy summary, which is Section 10.4\'s situation of features that cannot distinguish states.', link: 'm02' },
  { kind: 'num', text: 'Exercise 10.6 (rewards 1, 0, 1, 0, … from A): give E[R₁ | S₀ = A] and E[R₁ | S₀ = B].', answers: [{ label: 'E[R₁ | A]', value: 1 }, { label: 'E[R₁ | B]', value: 0 }], why: 'Leaving A pays 1, leaving B pays 0. The process is periodic, so Pr{S_t} has no limit, but the average reward 1/2 exists.', link: 'm09' },
  { kind: 'num', text: 'A two-state chain has P(A→B) = 0.3, P(B→A) = 0.6 and expected rewards 1 in A and 5 in B. What is r(π)?', answers: [{ label: 'r(π)', value: 7 / 3 }], why: 'Balance of flow: 0.3 μ(A) = 0.6 μ(B), so μ = (2/3, 1/3) and r(π) = 2/3·1 + 1/3·5 = 7/3.', link: 'm08' },
  { kind: 'num', text: 'Same alternating chain as Exercise 10.6: the differential values v(A) and v(B) (normalized so they average to 0).', answers: [{ label: 'v(A)', value: 0.25 }, { label: 'v(B)', value: -0.25 }], why: 'f_A(γ) = 1/(2(1 + γ)) → 1/4 and f_B(γ) → −1/4. Check: v(A) = 1 − 1/2 + v(B).', link: 'm10' },
  { kind: 'num', text: 'Ring A → B → C → A with +1 on arriving at A: the differential values v(A), v(B), v(C).', answers: [{ label: 'v(A)', value: -1 / 3 }, { label: 'v(B)', value: 0 }, { label: 'v(C)', value: 1 / 3 }], why: 'r(π) = 1/3. The value rises as the reward gets closer: C is one step from the +1. These are the values that make δ exactly 0 in Exercise 10.8.', link: 'm12' },
];

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const L = ctx.store.local<{ done: Record<number, boolean> }>('m18', { done: {} });
  const score = el('p', { class: 'status' });
  const updateScore = () => {
    const right = Object.values(L.done).filter(Boolean).length;
    score.textContent = `${right} of ${QUESTIONS.length} answered correctly.`;
  };

  const cards = QUESTIONS.map((q, i) => {
    const feedback = el('div', { class: 'feedback', attrs: { 'aria-live': 'polite' } });
    const reveal = (ok: boolean, prefix: string) => {
      L.done[i] = ok;
      feedback.replaceChildren(
        el('p', {}, el('strong', { class: ok ? 'ok' : 'bad', text: ok ? 'Correct. ' : 'Not quite. ' }), prefix),
        el('p', { text: q.why }));
      if (q.link) feedback.append(button(`See it in ${q.link.toUpperCase()} →`, () => ctx.go(q.link!), 'link'));
      updateScore();
    };
    if (q.kind === 'tf') {
      return card(`Q${i + 1}`, el('p', { text: q.text }), el('div', { class: 'buttons' },
        button('True', () => reveal(q.answer === true, `The statement is ${q.answer ? 'true' : 'false'}.`)),
        button('False', () => reveal(q.answer === false, `The statement is ${q.answer ? 'true' : 'false'}.`))), feedback);
    }
    const inputs = q.answers.map(a => {
      const input = el('input', { attrs: { type: 'text', inputmode: 'decimal', placeholder: 'e.g. 1/4 or 0.25', 'aria-label': a.label } });
      return { a, input, el: el('label', { class: 'ctl' }, el('span', { text: a.label }), input) };
    });
    const checkBtn = button('Check', () => {
      const ok = inputs.every(({ a, input }) => Math.abs(parseNumber(input.value) - a.value) <= 1e-3);
      inputs.forEach(({ a, input }) => input.setAttribute('aria-invalid', String(!(Math.abs(parseNumber(input.value) - a.value) <= 1e-3))));
      reveal(ok, `Answer: ${q.answers.map(a => `${a.label} = ${fmtFrac(a.value)}`).join(', ')}.`);
    }, 'primary');
    return card(`Q${i + 1}`, el('p', { text: q.text }), el('div', { class: 'grid3' }, ...inputs.map(x => x.el)), checkBtn, feedback);
  });

  root.append(
    moduleHeader('m18', 'Check yourself', 'Each card reveals an explanation and links to the module that makes the point visible. Numeric answers accept fractions (7/3) or decimals within 0.001.'),
    el('style', { text: '.feedback .ok { color: var(--accept); } .feedback .bad { color: var(--danger); } .feedback p { font-size: .85rem; margin-top: .25rem; } input[aria-invalid="true"] { border-color: var(--danger); } input[type=text] { display: block; width: 100%; margin-top: .2rem; font: inherit; color: inherit; background: var(--inset); border: 1px solid var(--border); border-radius: 6px; padding: .25rem .4rem; }' }),
    score,
    el('div', { class: 'grid2' }, ...cards),
  );
  updateScore();
  return () => {};
}

function fmtFrac(v: number): string {
  for (const d of [1, 2, 3, 4]) {
    const n = Math.round(v * d);
    if (Math.abs(n / d - v) < 1e-9) return d === 1 ? String(n) : `${n}/${d}`;
  }
  return v.toFixed(3);
}
