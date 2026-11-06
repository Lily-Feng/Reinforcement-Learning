/**
 * M09 · When does the limit exist? Pr{S_t = s} over time and the running
 * time-average reward for toy chains (exact) and for the diurnal cluster
 * (2,000 simulator replicas), with ergodicity badges.
 */

import { drawXY } from '../../shared/bars';
import { mulberry32 } from '../../shared/rng';
import type { ReplicaResult } from '../core/jobs.ts';
import { alternating, averageReward, chainProperties, distributionOverTime, expectedRewardSequence, lazyRing, reduciblePair, ring, sampleStep, type Chain } from '../core/toy-chains.ts';
import type { ModuleContext } from '../app/context';
import { canvasPanel, card, el, fmt, legend, moduleHeader, observeBox, progressBar, range, select, theme } from '../app/ui';

type Preset = 'alternating' | 'ring' | 'lazy' | 'reducible' | 'cluster';

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const L = ctx.store.local('m09', { preset: 'alternating' as Preset, eps: 0.1, start: 0, T: 40 });
  const prog = progressBar();
  const badges = el('div', { class: 'buttons' });
  const note = el('p', { class: 'status' });
  let replicas: ReplicaResult | null = null;
  let chain: Chain = alternating();

  const distPanel = canvasPanel('Pr{S_t = s}', 230, canvas => draw(canvas, 'dist'));
  const avgPanel = canvasPanel('Running time-average reward (1/t) Σ R', 200, canvas => draw(canvas, 'avg'));
  const distLegend = el('div');
  const startSel = el('div');

  const chainFor = (p: Preset): Chain => (p === 'alternating' ? alternating() : p === 'ring' ? ring() : p === 'lazy' ? lazyRing(L.eps) : reduciblePair());

  function draw(canvas: HTMLCanvasElement, which: 'dist' | 'avg'): void {
    const t = theme();
    if (L.preset === 'cluster') {
      if (!replicas) return;
      const K = ctx.cfg.K;
      if (which === 'dist') {
        const show = [0, Math.floor(K / 2), K];
        drawXY(canvas, show.map((g, i) => ({ color: t.series[i]!, points: replicas!.dist.map((d, k) => [k, d[g]!]) })), { xLabel: 'step t (96 steps = one day)', yMin: 0, textColor: t.text, gridColor: t.grid });
      } else {
        drawXY(canvas, [
          { color: t.series[0]!, points: replicas.running.map((v, k) => [k + 1, v]) },
          { color: t.series[1]!, points: replicas.meanReward.map((v, k) => [k, v]) },
        ], { xLabel: 'step t', textColor: t.text, gridColor: t.grid });
      }
      return;
    }
    const n = chain.states.length;
    const start = chain.states.map((_, i) => (i === L.start ? 1 : 0));
    if (which === 'dist') {
      const d = distributionOverTime(chain, start, L.T);
      drawXY(canvas, chain.states.map((_, s) => ({ color: t.series[s % 6]!, dots: true, r: 2, points: d.map((p, k) => [k, p[s]!]) })), { xLabel: 'step t', yMin: 0, yMax: 1, textColor: t.text, gridColor: t.grid });
    } else {
      const er = expectedRewardSequence(chain, L.start, L.T);
      let sum = 0;
      const exact = er.map((r, k) => { sum += r; return [k + 1, sum / (k + 1)]; });
      const rng = mulberry32(ctx.cfg.seed);
      let s = L.start;
      let tot = 0;
      const sample: number[][] = [];
      for (let k = 1; k <= L.T; k += 1) { const st = sampleStep(chain, s, rng); tot += st.reward; s = st.next; sample.push([k, tot / k]); }
      drawXY(canvas, [{ color: t.series[0]!, points: exact }, { color: t.series[1]!, points: sample, dash: [4, 3] }], { xLabel: 'step t', textColor: t.text, gridColor: t.grid });
      void n;
    }
  }

  const badge = (label: string, yes: boolean) => el('span', { class: `badge ${yes ? 'yes' : 'no'}`, text: `${yes ? '✓' : '✗'} ${label}` });

  const update = () => {
    const t = theme();
    if (L.preset === 'cluster') {
      startSel.replaceChildren();
      badges.replaceChildren(badge('irreducible', true), badge('aperiodic', false), badge('limiting distribution exists', false), badge('time-average exists', true));
      const K = ctx.cfg.K;
      distLegend.replaceChildren(legend([0, Math.floor(K / 2), K].map((g, i) => ({ label: `Pr{g = ${g} free}`, color: t.series[i]! }))));
      avgLegend.replaceChildren(legend([{ label: 'running average of one replica', color: t.series[0]! }, { label: 'mean reward at step t over replicas', color: t.series[1]! }]));
      note.textContent = replicas ? 'Cluster with A = 0.8, hand-tuned threshold policy, 2,000 replicas from an empty cluster at midnight. The clock makes the chain periodic with period 96: Pr{g} keeps oscillating day after day, but one replica\'s running average still settles.' : 'Simulating 2,000 replicas…';
      if (!replicas) {
        const j = ctx.job<ReplicaResult>('replicas', { cfg: { ...ctx.cfg, amp: 0.8 }, replicas: 2000, T: 480, seed: ctx.cfg.seed }, f => prog.set(f));
        ctx.when(j.promise, r => { replicas = r; prog.set(null); update(); });
      }
    } else {
      chain = chainFor(L.preset);
      if (L.start >= chain.states.length) L.start = 0;
      const props = chainProperties(chain);
      const reducible = L.preset === 'reducible';
      badges.replaceChildren(badge('irreducible', props.irreducible), badge('aperiodic', props.aperiodic), badge('limiting distribution exists', props.limitExists && props.irreducible), badge('time-average exists', props.averageExists));
      distLegend.replaceChildren(legend(chain.states.map((s, i) => ({ label: s, color: t.series[i % 6]! }))));
      avgLegend.replaceChildren(legend([{ label: 'exact (1/t) Σ E[R]', color: t.series[0]! }, { label: 'one sampled path', color: t.series[1]!, dash: true }]));
      startSel.replaceChildren(select('Start state', chain.states.map((s, i) => ({ value: String(i), label: s })), String(L.start), v => { L.start = Number(v); update(); }).el);
      note.textContent = `${chain.note} ${reducible ? `Long-run average from this start: ${L.start < 2 ? '1/2 (ring A)' : '1 (ring B)'}; it depends on where you start.` : `r(π) = ${fmt(averageReward(chain), 4)}.`}`;
    }
    epsBox.hidden = L.preset !== 'lazy';
    distPanel.redraw();
    avgPanel.redraw();
  };

  const avgLegend = el('div');
  const epsBox = el('div', {}, range('Laziness ε (stay probability)', { min: 0, max: 0.9, step: 0.05, value: L.eps, format: v => fmt(v, 2) }, v => { L.eps = v; update(); }).el);
  root.append(
    moduleHeader('m09', 'When does the limit exist? (ergodicity)', 'The second form of r(π) needs a steady-state distribution that does not depend on S₀. An ergodic chain (irreducible and aperiodic) has one. But the time-average reward can exist without it.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Chain', select<Preset>('Preset', [
          { value: 'alternating', label: 'Alternating A ↔ B (Ex. 10.6)' }, { value: 'ring', label: 'Three-state ring' },
          { value: 'lazy', label: 'Lazy ring' }, { value: 'reducible', label: 'Reducible pair of rings' }, { value: 'cluster', label: 'GPU cluster, A = 0.8' },
        ], L.preset, v => { L.preset = v; update(); }).el, startSel, epsBox,
        range('Steps shown', { min: 10, max: 200, step: 10, value: L.T }, v => { L.T = v; update(); }).el, prog.el)),
      el('div', { class: 'panels' }, badges, note, card(null, distLegend, distPanel.el), card(null, avgLegend, avgPanel.el)),
    ),
    observeBox([
      'Alternating and ring chains are periodic: Pr{S_t = s} jumps between 0 and 1 forever, but the time-average reward converges (to 1/2 and 1/3). Ergodicity is sufficient, not necessary.',
      'Lazy ring: any ε > 0 adds self-loops, the chain becomes aperiodic, and Pr{S_t = s} converges to 1/3 each. Smaller ε converges more slowly.',
      'Reducible pair: the long-run average is 1/2 from ring A and 1 from ring B. With no single μ, "the" r(π) does not exist.',
      'GPU cluster with A = 0.8: the time-of-day clock makes it periodic. The distribution of free GPUs repeats every day and never settles, yet the running average reward converges.',
    ]),
  );
  update();
  return () => { distPanel.destroy(); avgPanel.destroy(); };
}
