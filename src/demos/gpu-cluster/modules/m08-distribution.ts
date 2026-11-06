/**
 * M08 · Where does the cluster spend its time? The exact on-policy
 * distribution μ_π of a threshold policy on the bucketed model, the transient
 * distribution from a chosen start, and a continuous simulation converging
 * toward it; r(π) = Σ μ_π(s) E[r | s] decomposed by free GPUs.
 */

import { drawBars } from '../../shared/bars';
import { mulberry32 } from '../../shared/rng';
import { createEnv, observe, step, type Env } from '../core/env.ts';
import { countChain, evaluatePolicy, pushForward, stepFull, type CountChain, type Evaluation } from '../core/exact.ts';
import { policyFrom, reward, type ModelPolicy } from '../core/model.ts';
import { handTuned, thresholdDecide, thresholdOnModel, type Threshold } from '../core/policies.ts';
import type { ModuleContext } from '../app/context';
import { button, canvasPanel, card, createRunner, el, fmt, legend, moduleHeader, observeBox, range, runnerControls, select, theme } from '../app/ui';
import { bucketWords, kpi, onNextFrame, thresholdControls } from './common';

interface Local {
  th: Threshold;
  start: 'empty' | 'full-long';
  speed: number;
}

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = { ...ctx.cfg, amp: 0 };
  const store = ctx.store;
  const L = store.local<Local>('m08', { th: handTuned(cfg.K), start: 'empty', speed: 50 });
  if (Math.max(...L.th.theta) > cfg.K + 1) L.th = handTuned(cfg.K);
  const model = store.model(cfg);
  const B = model.B;

  let policy: ModelPolicy;
  let ev: Evaluation;
  let chain: CountChain;
  let nuT: Float64Array;
  let t = 0;
  let env: Env;
  let simCounts: number[] = [];
  let simSteps = 0;
  let simReward = 0;
  let oneStepErr: number | null = null;

  const startDistribution = (): Float64Array => {
    const nu = new Float64Array(model.nCounts);
    if (L.start === 'empty') nu[model.empty] = 1;
    else {
      const full = model.counts.findIndex(n => n[B - 1] === cfg.K);
      nu[full] = 1;
    }
    return nu;
  };

  const resetSim = () => {
    env = createEnv(cfg, mulberry32(cfg.seed));
    if (L.start === 'full-long') {
      // Fill every GPU with a long job (the longest bucket's representative length).
      for (let i = 0; i < cfg.K; i += 1) {
        env.q[i] = model.buckets.q[B - 1]!;
        env.len[i] = model.buckets.len[B - 1]!;
        env.tier[i] = 0;
        env.bucket[i] = B - 1;
      }
    }
    simCounts = new Array<number>(cfg.K + 1).fill(0);
    simSteps = 0;
    simReward = 0;
    nuT = startDistribution();
    t = 0;
  };

  const recompute = () => {
    policy = policyFrom(model, thresholdOnModel(L.th));
    ev = evaluatePolicy(model, policy);
    chain = countChain(model, policy);
    oneStepErr = null;
    resetSim();
  };
  recompute();

  const unit = () => {
    const o = observe(env);
    simCounts[o.g]! += 1;
    simReward += step(env, thresholdDecide(L.th, o)).reward;
    simSteps += 1;
    nuT = pushForward(chain, nuT);
    t += 1;
  };

  /** Pr{g} and the busy mix by bucket for a count-level distribution. */
  const byFree = (nu: Float64Array) => {
    const pg = new Array<number>(cfg.K + 1).fill(0);
    const seg = Array.from({ length: cfg.K + 1 }, () => new Array<number>(B).fill(0));
    for (let ci = 0; ci < model.nCounts; ci += 1) {
      const g = model.free[ci]!;
      const m = nu[ci]!;
      pg[g]! += m;
      const busy = cfg.K - g;
      if (busy > 0) model.counts[ci]!.forEach((nb, b) => { seg[g]![b]! += (m * nb) / busy; });
    }
    return { pg, seg };
  };

  const k = { r: kpi('r(π), exact'), sim: kpi('Simulated average reward'), t: kpi('Steps from start'), tv: kpi('max |Pr_t(g) − μ(g)|') };
  const t0 = theme();
  const bucketNames = B === 1 ? ['busy'] : bucketWords(B).map((w, b) => `busy, ${w} jobs (ℓ≈${model.buckets.len[b]!.toFixed(0)})`);
  const bars = canvasPanel('μ_π over free GPUs (stack = busy jobs by length bucket)', 260, canvas => {
    const t = theme();
    const { pg, seg } = byFree(ev.nu);
    const trans = byFree(nuT).pg;
    const emp = simSteps ? simCounts.map(c => c / simSteps) : [];
    drawBars(canvas, pg.map((p, g) => ({
      label: String(g),
      segments: g === cfg.K ? [{ value: p, color: t.grid }] : seg[g]!.map((v, b) => ({ value: v, color: t.bucket[Math.min(3, b + (B === 1 ? 1 : 0))]! })),
    })), {
      stacked: true, xLabel: 'free GPUs g', textColor: t.text, gridColor: t.grid, formatY: v => fmt(v, 2),
      markers: [{ values: trans, color: t.series[1]! }, ...(emp.length ? [{ values: emp, color: t.ref }] : [])],
    });
  }, {
    legend: legend([
      ...bucketNames.map((name, b) => ({ label: name, color: t0.bucket[Math.min(3, b + (B === 1 ? 1 : 0))]! })),
      { label: 'Pr{g at step t} from the start (exact)', color: t0.series[1]! },
      { label: 'continuous simulation, time average', color: t0.ref },
    ]),
  });

  const table = el('table');
  const renderTable = () => {
    const pg = new Array<number>(cfg.K + 1).fill(0);
    const rg = new Array<number>(cfg.K + 1).fill(0);
    for (let s = 0; s < model.nStates; s += 1) {
      const ci = Math.floor(s / (4 * B));
      const g = model.free[ci]!;
      pg[g]! += ev.mu[s]!;
      rg[g]! += ev.mu[s]! * reward(model, s, policy[s]!);
    }
    table.replaceChildren(
      el('thead', {}, el('tr', {}, ...['free GPUs g', 'Σ μ(s) over g', 'E[r | g, π]', 'contribution μ·E[r]'].map(h => el('th', { text: h })))),
      el('tbody', {}, ...pg.map((p, g) => el('tr', {}, el('td', { text: String(g) }), el('td', { text: fmt(p, 4) }), el('td', { text: p > 1e-12 ? fmt(rg[g]! / p, 3) : '—' }), el('td', { text: fmt(rg[g]!, 4) })))),
      el('tfoot', {}, el('tr', { class: 'hl' }, el('td', { text: 'total' }), el('td', { text: fmt(pg.reduce((a, b) => a + b, 0), 4) }), el('td', { text: '' }), el('td', { text: `r(π) = ${fmt(rg.reduce((a, b) => a + b, 0), 4)}` }))));
  };

  const oneStep = el('p', { class: 'status' });
  const render = () => {
    const { pg } = byFree(ev.nu);
    const trans = byFree(nuT).pg;
    k.r.set(fmt(ev.avg), `exact, ${B}-bucket model`);
    k.sim.set(simSteps ? fmt(simReward / simSteps) : '—', `${simSteps.toLocaleString()} continuous steps`);
    k.t.set(t.toLocaleString());
    k.tv.set(fmt(Math.max(...pg.map((p, g) => Math.abs(p - trans[g]!))), 4));
    oneStep.textContent = oneStepErr === null ? '' : `One step of the dynamics applied to μ_π: max |μ_π P_π − μ_π| = ${oneStepErr.toExponential(1)}. μ_π is unchanged (equation 10.8).`;
    bars.redraw();
  };

  const runner = createRunner({ unit, render, perFrame: () => L.speed, onState: (p, ff) => rc.sync(p, ff) });
  const rc = runnerControls(runner, { ff: [{ label: '1,000 steps', units: 1000 }, { label: '20,000 steps', units: 20000 }, { label: '200,000 steps', units: 200000 }] });
  const update = onNextFrame(() => { runner.pause(); recompute(); renderTable(); render(); });

  root.append(
    moduleHeader('m08', 'Where does the cluster spend its time?', 'For a fixed threshold policy, the steady-state distribution μ_π is computed <strong>exactly</strong> on the bucketed model. A simulation of the continuous cluster runs under the same policy, and its histogram approaches μ_π. The long-run reward is just μ_π-weighted expected reward.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Run from a start state', select('Start', [{ value: 'empty', label: 'All GPUs free' }, { value: 'full-long', label: 'All GPUs busy with long jobs' }], L.start, v => { L.start = v; runner.pause(); resetSim(); render(); }).el, rc.el,
          range('Steps per frame', { min: 1, max: 2000, step: 1, value: L.speed }, v => { L.speed = v; }).el,
          button('Apply one step to μ_π', () => {
            const next = stepFull(model, policy, ev.mu);
            oneStepErr = Math.max(...Array.from(next, (v, s) => Math.abs(v - ev.mu[s]!)));
            render();
          })),
        card('Threshold policy', thresholdControls(L.th, cfg.K, update)),
      ),
      el('div', { class: 'panels' },
        el('div', { class: 'kpis' }, k.r.el, k.sim.el, k.t.el, k.tv.el),
        ctx.cfg.amp > 0 ? el('p', { class: 'hint', text: 'This module uses the base tier mix (A = 0): μ_π is only defined without the clock.' }) : null,
        oneStep, bars.el,
        card('r(π) = Σ_s μ_π(s) E[r | s, π], grouped by free GPUs', el('div', { class: 'table-scroll' }, table)),
      ),
    ),
    observeBox([
      'Run from <strong>All GPUs free</strong>, then from <strong>All GPUs busy with long jobs</strong>. The orange ticks (exact Pr{g at step t}) start in opposite corners and end on the same bars. The start state is forgotten.',
      'The continuous simulation (amber ticks) settles close to the bars. Small gaps remain because the model represents each job length by its bucket.',
      '<strong>Apply one step</strong> leaves μ_π unchanged to machine precision. That is what "steady state" means (equation 10.8).',
      'Move a threshold, for example make Basic need more free GPUs. μ_π shifts toward more free GPUs and r(π) changes. A policy changes long-run reward only by changing where the cluster spends its time and what it earns there.',
    ]),
  );
  renderTable();
  render();
  return () => { runner.destroy(); bars.destroy(); };
}
