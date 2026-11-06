/**
 * M03 · One semi-gradient Sarsa update, fully exposed, in the one-shift task.
 * Every number comes from the agent's own trace record.
 */

import { mulberry32 } from '../../shared/rng';
import { createEpisodicAgent, episodicBegin, episodicStep, qValue, type EpisodicAgent, type UpdateRecord } from '../core/agents.ts';
import { ACCEPT, TIER_NAMES } from '../core/config.ts';
import { createEnv, observe, resetEnv, step, type Env, type Observation } from '../core/env.ts';
import { createFeaturizer, defaultFeatures, GROUP_LABELS } from '../core/features.ts';
import type { ModuleContext } from '../app/context';
import { button, card, check, el, fmt, moduleHeader, observeBox, range } from '../app/ui';

interface Entry {
  rec: UpdateRecord;
  sNext: Observation | null;
  aNext: number;
  nextFeatures: number[];
  groups: { group: string; indices: number[] }[];
  /** q̂(S′, A′) recomputed with the weights after the update. */
  bootstrapAfter: number | null;
  shift: number;
}

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = ctx.cfg;
  const L = ctx.store.local('m03', { alpha: 0.3, eps: 0.1, recompute: false });
  let env: Env;
  let agent: EpisodicAgent;
  let action = 0;
  let shift = 1;
  const history: Entry[] = [];
  let cursor = -1;

  const reset = () => {
    env = createEnv(cfg, mulberry32(cfg.seed), { episodic: true });
    const feat = createFeaturizer(cfg, env.buckets, defaultFeatures(cfg, true));
    agent = createEpisodicAgent(feat, { alpha: L.alpha, eps: L.eps, n: 1, trace: true }, mulberry32(cfg.seed + 3));
    action = episodicBegin(agent, observe(env));
    history.length = 0;
    cursor = -1;
    shift = 1;
  };
  reset();

  const advance = (n: number) => {
    for (let i = 0; i < n; i += 1) {
      if (env.done) {
        resetEnv(env);
        shift += 1;
        action = episodicBegin(agent, observe(env));
      }
      const r = step(env, action);
      const o = observe(env);
      const res = episodicStep(agent, r.reward, o, r.done);
      for (const rec of res.records) {
        const sNext = r.done && rec.tau === agent.T - 1 ? null : o;
        const aNext = res.action;
        const nextFeatures = sNext ? agent.feat.indices(sNext, aNext) : [];
        history.push({
          rec, sNext, aNext, nextFeatures, shift,
          groups: agent.feat.groupIndices(rec.obs, rec.action).map(g => ({ group: GROUP_LABELS[g.group], indices: g.indices })),
          bootstrapAfter: sNext ? qValue(agent.w, nextFeatures) : null,
        });
        if (history.length > 300) history.shift();
      }
      action = res.action;
    }
    cursor = history.length - 1;
    render();
  };

  const view = el('div', { class: 'panels' });
  const pos = el('span', { class: 'hint' });
  const describe = (o: Observation) => `g = ${o.g}, ${TIER_NAMES[o.c]} request, ℓ = ${fmt(o.len, 1)}, ρ = ${fmt(o.rho, 2)}, ${o.togo} steps left`;
  const row = (k: string, v: string, cls = '') => el('tr', { class: cls }, el('th', { class: 'l', text: k }), el('td', { class: 'l', html: v }));

  function render(): void {
    pos.textContent = history.length ? `update ${cursor + 1} of ${history.length} kept` : '';
    if (cursor < 0) { view.replaceChildren(card(null, el('p', { text: 'Press Step to take one action and make one update.' }))); return; }
    const e = history[cursor]!;
    const r = e.rec;
    const n = agent.feat.active;
    const target = L.recompute && e.bootstrapAfter !== null ? r.rewards[0]! + e.bootstrapAfter : r.target;
    const transition = el('table', {}, el('tbody', {},
      row('Shift / τ', `shift ${e.shift}, step τ = ${r.tau}`),
      row('S (observation)', describe(r.obs)),
      row('A', r.action === ACCEPT ? 'accept' : 'reject'),
      row('R', fmt(r.rewards[0]!, 2)),
      row("S′", e.sNext ? describe(e.sNext) : 'terminal (end of shift)'),
      row("A′", e.sNext ? (e.aNext === ACCEPT ? 'accept' : 'reject') : '—')));
    const groups = el('table', {},
      el('thead', {}, el('tr', {}, el('th', { class: 'l', text: 'group' }), el('th', { class: 'l', text: 'active weight indices' }))),
      el('tbody', {}, ...e.groups.map(g => el('tr', {}, el('td', { class: 'l', text: g.group }), el('td', { class: 'l', text: g.indices.join(', ') })))));
    const math = el('table', {}, el('tbody', {},
      row('q̂(S, A, w) before', `${fmt(r.qBefore, 4)} <span class="hint">sum of ${n} active weights</span>`),
      row("q̂(S′, A′, w)", e.sNext ? fmt(r.bootstrap ?? 0, 4) : '0 (terminal)'),
      row('Target U = R + q̂(S′, A′, w)', `${fmt(r.rewards[0]!, 2)} + ${e.sNext ? fmt(r.bootstrap ?? 0, 4) : '0'} = <strong>${fmt(r.target, 4)}</strong>`),
      row('δ = U − q̂(S, A, w)', `<strong>${fmt(r.delta, 4)}</strong>`),
      row('Δw per active weight = (α / active)·δ', `(${L.alpha} / ${n}) × ${fmt(r.delta, 4)} = <strong>${fmt(r.dw, 5)}</strong>`),
      row('q̂(S, A, w) after', `${fmt(r.qAfter, 4)} <span class="hint">moved ${fmt(r.qAfter - r.qBefore, 4)} = α·δ toward the target</span>`),
      ...(L.recompute ? [
        row("q̂(S′, A′, w) recomputed after the update", e.bootstrapAfter === null ? 'terminal' : `${fmt(e.bootstrapAfter, 4)} <span class="hint">changed by ${fmt(e.bootstrapAfter - (r.bootstrap ?? 0), 4)}</span>`, 'hl'),
        row('Target recomputed', `${fmt(target, 4)} <span class="hint">the target moved too: it depends on w</span>`, 'hl'),
        row("Shared tiles between (S, A) and (S′, A′)", String(r.features.filter(i => e.nextFeatures.includes(i)).length), 'hl'),
      ] : [])));
    view.replaceChildren(card('Transition', el('div', { class: 'table-scroll' }, transition)), card('Active features of (S, A)', el('div', { class: 'table-scroll' }, groups)), card('The update', el('div', { class: 'table-scroll' }, math)));
  }

  root.append(
    moduleHeader('m03', 'One semi-gradient Sarsa update, fully exposed', 'One shift of H steps starting with every GPU free. The agent uses episodic one-step semi-gradient Sarsa with tile coding (G1, G2 and G4 = free GPUs × time to go). Step once and read every number in the update.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Step', el('div', { class: 'buttons' }, button('Step', () => advance(1), 'primary'), button('×10', () => advance(10)), button('One shift', () => advance(cfg.H))),
          el('div', { class: 'buttons' }, button('◀ Back', () => { if (cursor > 0) { cursor -= 1; render(); } }), button('Forward ▶', () => { if (cursor < history.length - 1) { cursor += 1; render(); } })), pos,
          check('Recompute target after update', L.recompute, v => { L.recompute = v; render(); }).el),
        card('Agent',
          range('α × active features', { min: 0.01, max: 1, step: 0.01, value: L.alpha, format: v => fmt(v, 2) }, v => { L.alpha = v; agent.opts.alpha = v; }).el,
          range('ε', { min: 0, max: 1, step: 0.01, value: L.eps, format: v => fmt(v, 2) }, v => { L.eps = v; agent.opts.eps = v; }).el,
          button('Reset', () => { reset(); render(); })),
      ),
      view,
    ),
    observeBox([
      'The weights of (S, A) move by (α / active)·δ each, so q̂(S, A) moves by α·δ: the step size reads as "α × number of active features".',
      'Turn on <strong>Recompute target after update</strong>: q̂(S′, A′) changed as well, because S and S′ share tiles. The target depends on w, but the update treated it as a constant. That ignored dependence is why the method is only <em>semi</em>-gradient.',
      'At the last step of a shift the target is just R: the next state is terminal and contributes 0.',
    ]),
  );
  render();
  return () => {};
}
