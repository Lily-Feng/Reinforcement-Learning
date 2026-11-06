/**
 * M13 · The main lab: differential semi-gradient n-step Sarsa on the
 * continuous cluster, scored exactly on the bucketed model at checkpoints.
 */

import { drawXY } from '../../shared/bars';
import { mulberry32 } from '../../shared/rng';
import { continuingBegin, continuingStep, createContinuingAgent, greedyAction, qOf, RULE_LABELS, type AverageRule, type ContinuingAgent } from '../core/agents.ts';
import { ACCEPT, BOOK_CONFIG, DEFAULT_CONFIG, REJECT, TIER_NAMES } from '../core/config.ts';
import { createEnv, observe, step, type Env, type Observation } from '../core/env.ts';
import { agreement, greedyOnModel } from '../core/evaluate.ts';
import { evaluatePolicy } from '../core/exact.ts';
import { createFeaturizer, defaultFeatures, TABULAR, type FeatureConfig } from '../core/features.ts';
import { typicalRho, type Reference } from '../core/jobs.ts';
import type { ModuleContext } from '../app/context';
import { referenceLabel } from '../app/store';
import { button, canvasPanel, card, check, createRunner, el, fmt, legend, moduleHeader, observeBox, range, rollingMean, runnerControls, select, theme } from '../app/ui';
import { decisionMap, kpi, lengthGrid, mapLegend, optimumDecider, recorder, tierLegend } from './common';

interface Local {
  features: 'tiles' | 'tabular';
  g1: boolean;
  g2: boolean;
  g3: boolean;
  tilingsG1: number;
  tilesPerDim: number;
  alpha: number;
  beta: number;
  eps: number;
  n: number;
  rule: AverageRule;
  speed: number;
}

const DEFAULTS: Local = { features: 'tiles', g1: true, g2: true, g3: false, tilingsG1: 8, tilesPerDim: 4, alpha: 0.1, beta: 0.01, eps: 0.1, n: 1, rule: 'td', speed: 200 };

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const store = ctx.store;
  const cfg = ctx.cfg;
  const L = store.local<Local>('m13', { ...DEFAULTS, g3: cfg.amp > 0 });
  if (cfg.sigma !== 0 && L.features === 'tabular') L.features = 'tiles';

  const model = store.model(cfg);
  const lengths = lengthGrid(24);
  const spec = (): FeatureConfig => {
    if (L.features === 'tabular') return TABULAR;
    const s = defaultFeatures(cfg);
    s.groups = { g1: L.g1, g2: L.g2, g3: L.g3, g4: false };
    if (!L.g1 && !L.g2 && !L.g3) s.groups.g1 = true;
    s.tilings.g1 = L.tilingsG1;
    s.tilesPerDim = L.tilesPerDim;
    return s;
  };

  // --- state -------------------------------------------------------------
  let env: Env;
  let agent: ContinuingAgent;
  let steps = 0;
  let ref: Reference | null = null;
  let optDecide: ((c: number, g: number, len: number) => number) | null = null;
  let mcRef: { mean: number; half: number } | null = null;
  let roll = rollingMean(5000);
  let rewardSeries = recorder();
  let rbarSeries = recorder();
  let exactPoints = recorder(300);
  let lastScore = { steps: -1, avg: NaN, agree: NaN };
  let lastScoreTime = 0;
  let lastHeavy = 0;
  let visits = new Int32Array((cfg.K + 1) * 4 * lengths.length);
  let rhoScale = 1;

  const lenBin = (len: number) => Math.max(0, Math.min(lengths.length - 1, Math.round((Math.log(len / 2) / Math.log(60)) * (lengths.length - 1))));

  const reset = () => {
    env = createEnv(cfg, mulberry32(cfg.seed));
    const feat = createFeaturizer(cfg, env.buckets, spec());
    agent = createContinuingAgent(feat, { kind: 'differential', alpha: L.alpha, beta: L.beta, eps: L.eps, n: L.n, rule: L.rule }, mulberry32(cfg.seed + 7919));
    continuingBegin(agent, observe(env));
    steps = 0;
    roll = rollingMean(5000);
    rewardSeries = recorder();
    rbarSeries = recorder();
    exactPoints = recorder(300);
    lastScore = { steps: -1, avg: NaN, agree: NaN };
    visits = new Int32Array((cfg.K + 1) * 4 * lengths.length);
  };
  reset();

  const unit = () => {
    const o = observe(env);
    visits[(o.g * 4 + o.c) * lengths.length + lenBin(o.len)]! += 1;
    const r = step(env, agent.action).reward;
    continuingStep(agent, r, observe(env));
    steps += 1;
    roll.push(r);
    if (steps % 250 === 0) { rewardSeries.push(steps, roll.value()); rbarSeries.push(steps, agent.avg.rbar); }
  };

  // --- layout ------------------------------------------------------------
  const k = {
    steps: kpi('Steps'), rbar: kpi('R̄ (learned)'), roll: kpi('Avg reward, last 5k'), ref: kpi('Reference'),
    exact: kpi('Exact r(greedy)'), agree: kpi('Agreement with optimum', '—', 'μ*-weighted'),
  };
  const status = el('p', { class: 'status' });
  const runner = createRunner({ unit, render: () => render(), perFrame: () => L.speed, onState: (p, ff) => { rc.sync(p, ff); if (!p && ff === 0) { share(); render(true); } } });
  ctx.onDispose(() => runner.destroy());
  const rc = runnerControls(runner, { ff: [{ label: '10k steps', units: 1e4 }, { label: '100k steps', units: 1e5 }, { label: '500k steps', units: 5e5 }, { label: '2M steps', units: 2e6 }] });

  const share = () => {
    if (steps === 0) return;
    store.labAgent = { agent, cfg, steps, label: `M13 agent after ${steps.toLocaleString()} steps (${L.features === 'tabular' ? 'tabular' : `${agent.feat.active} active tiles`})` };
  };

  const restart = () => { runner.pause(); reset(); render(true); };

  const presets = el('div', { class: 'buttons' },
    button('Book Example 10.2', () => {
      Object.assign(L, { features: 'tabular', alpha: 0.01, beta: 0.01, eps: 0.1, n: 1, rule: 'td' });
      store.set({ ...BOOK_CONFIG, seed: cfg.seed });
      if (configMatches(BOOK_CONFIG)) { syncControls(); restart(); }
    }),
    button('Continuous jobs', () => {
      Object.assign(L, { ...DEFAULTS });
      store.set({ ...DEFAULT_CONFIG, seed: cfg.seed });
      if (configMatches(DEFAULT_CONFIG)) { syncControls(); restart(); }
    }),
    button('Diurnal', () => {
      Object.assign(L, { ...DEFAULTS, g3: true });
      store.set({ ...DEFAULT_CONFIG, amp: 0.8, seed: cfg.seed });
    }));
  const configMatches = (c: typeof DEFAULT_CONFIG) => c.K === cfg.K && c.median === cfg.median && c.sigma === cfg.sigma && c.amp === cfg.amp && c.kappa === cfg.kappa;

  const featSel = select('Features', [{ value: 'tiles', label: 'Tile coding' }, { value: 'tabular', label: 'Tabular (g, c) — σ = 0 only' }], L.features, v => {
    if (v === 'tabular' && cfg.sigma !== 0) { featSel.set('tiles'); status.textContent = 'Tabular features need σ = 0 (every job the same length). Use the Book preset or set σ = 0 in the settings.'; return; }
    L.features = v; syncControls(); restart();
  });
  const g1 = check('G1 · g × log ℓ', L.g1, v => { L.g1 = v; restart(); });
  const g2 = check('G2 · g × ρ', L.g2, v => { L.g2 = v; restart(); });
  const g3 = check('G3 · g × time of day (wrap)', L.g3, v => { L.g3 = v; restart(); });
  const tilings = range('G1 tilings', { min: 1, max: 16, step: 1, value: L.tilingsG1 }, v => { L.tilingsG1 = v; restart(); });
  const tpd = range('Tiles per dimension', { min: 2, max: 8, step: 1, value: L.tilesPerDim }, v => { L.tilesPerDim = v; restart(); });
  const tileBox = el('div', { class: 'controls' }, g1.el, g2.el, g3.el, tilings.el, tpd.el);
  const alpha = range('α × active features', { min: 0.005, max: 0.5, step: 0.005, value: L.alpha, format: v => fmt(v, 3) }, v => { L.alpha = v; agent.opts.alpha = v; });
  const beta = range('β (R̄ step size)', { min: 0.001, max: 0.1, step: 0.001, value: L.beta, format: v => fmt(v, 3) }, v => { L.beta = v; agent.avg.beta = v; });
  const eps = range('ε', { min: 0, max: 0.5, step: 0.01, value: L.eps, format: v => fmt(v, 2) }, v => { L.eps = v; agent.opts.eps = v; });
  const nCtl = range('n (Section 10.5)', { min: 1, max: 16, step: 1, value: L.n }, v => { L.n = v; restart(); });
  const rule = select('R̄ rule', (Object.keys(RULE_LABELS) as AverageRule[]).map(r => ({ value: r, label: RULE_LABELS[r] })), L.rule, v => { L.rule = v; restart(); });
  const speed = range('Steps per frame', { min: 1, max: 5000, step: 1, value: L.speed }, v => { L.speed = v; });
  const syncControls = () => {
    featSel.set(L.features);
    tileBox.hidden = L.features === 'tabular';
    g1.set(L.g1); g2.set(L.g2); g3.set(L.g3); tilings.set(L.tilingsG1); tpd.set(L.tilesPerDim);
    alpha.set(L.alpha); beta.set(L.beta); eps.set(L.eps); nCtl.set(L.n); rule.set(L.rule);
  };

  const t0 = theme();
  const curve = canvasPanel('Average reward', 220, canvas => {
    const t = theme();
    const rStar = ref && cfg.amp === 0 ? ref.rStar : mcRef?.mean;
    const xMax = Math.max(steps, 1);
    drawXY(canvas, [
      { color: t.series[0]!, points: rewardSeries.points() },
      { color: t.series[1]!, points: rbarSeries.points() },
      { color: t.series[2]!, points: exactPoints.points(), dots: true, r: 2.5 },
    ], {
      xMin: 0, xMax, xLabel: 'steps', formatX: v => (v >= 1e6 ? `${fmt(v / 1e6, 1)}M` : v >= 1e3 ? `${Math.round(v / 1e3)}k` : String(Math.round(v))),
      refs: rStar !== undefined ? [{ value: rStar, color: t.ref, label: cfg.amp > 0 ? 'MC ref' : 'r*', band: cfg.amp > 0 ? mcRef?.half : undefined }] : [],
      textColor: t.text, gridColor: t.grid,
    });
  }, { legend: legend([{ label: 'rolling average reward (5k steps)', color: t0.series[0]! }, { label: 'R̄', color: t0.series[1]! }, { label: 'exact r(greedy) at checkpoints', color: t0.series[2]! }, { label: 'reference', color: t0.ref, dash: true }]) });

  const decide = (c: number) => (g: number, li: number): number | null => {
    if (g === 0) return 0;
    const o: Observation = { g, c, len: lengths[li]!, rho: typicalRho(cfg, env.buckets.meanQ, g) * rhoScale, phase: 0.5, togo: Infinity, bucket: 0 };
    return greedyAction(agent, o);
  };
  const weight = (c: number) => (g: number, li: number) => Math.min(1, visits[(g * 4 + c) * lengths.length + li]! / 30);
  const learnedMaps = TIER_NAMES.map((name, c) => decisionMap(`Learned · ${name}`, () => cfg.K, lengths, decide(c), {
    weight: weight(c), height: 150, readout: (g, li) => `visits ${visits[(g * 4 + c) * lengths.length + li]}`,
  }));
  const optMaps = TIER_NAMES.map((name, c) => decisionMap(`Optimum (bucket bands) · ${name}`, () => cfg.K, lengths, (g, li) => (optDecide ? optDecide(c, g, lengths[li]!) : null), { height: 150 }));
  const rho = range('ρ for the maps (× typical value for each g)', { min: 0.25, max: 2, step: 0.05, value: 1, format: v => `${fmt(v, 2)}×` }, v => { rhoScale = v; learnedMaps.forEach(m => m.redraw()); valuePanel.redraw(); });

  const valuePanel = canvasPanel('Differential value of the best action vs. free GPUs (median ℓ, typical ρ)', 220, canvas => {
    const t = theme();
    const series = TIER_NAMES.map((_, c) => ({
      color: t.tier[c]!, dots: true,
      points: Array.from({ length: cfg.K + 1 }, (_, g) => {
        const o: Observation = { g, c, len: cfg.median, rho: typicalRho(cfg, env.buckets.meanQ, g) * rhoScale, phase: 0.5, togo: Infinity, bucket: 0 };
        return [g, g === 0 ? qOf(agent, o, REJECT) : Math.max(qOf(agent, o, ACCEPT), qOf(agent, o, REJECT))];
      }),
    }));
    drawXY(canvas, series, { xLabel: 'free GPUs g', textColor: t.text, gridColor: t.grid });
  }, { legend: tierLegend() });

  const refNote = el('p', { class: 'hint' });
  root.append(
    moduleHeader('m13', 'The main lab: differential semi-gradient Sarsa', 'The agent learns admission control on the <em>continuous</em> cluster from observations (g, c, ℓ, ρ, φ). At checkpoints its greedy policy is mapped onto the bucketed model and <strong>scored exactly</strong>, next to the bucketed optimum from relative value iteration.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Presets', presets, el('p', { class: 'hint', text: 'Book: σ = 0, tabular, α = β = 0.01, ε = 0.1; fast-forward 2M steps. Diurnal: A = 0.8 with G3; the reference is then Monte Carlo.' })),
        card('Run', rc.el, speed.el),
        card('Agent', featSel.el, tileBox, alpha.el, beta.el, eps.el, nCtl.el, rule.el, button('Reset agent', restart)),
      ),
      el('div', { class: 'panels' },
        el('div', { class: 'kpis' }, k.steps.el, k.rbar.el, k.roll.el, k.ref.el, k.exact.el, k.agree.el),
        refNote, status, curve.el,
        card('Learned decision boundary vs. bucketed optimum', mapLegend(), rho.el, el('p', { class: 'hint', text: `Faded cells were visited fewer than 30 times: their values are still close to the initial value. The optimum is drawn per length bucket, with busy jobs at their most likely mix.${cfg.sigma === 0 ? ` σ = 0: every job has ℓ = ${fmt(cfg.median, 1)}, so only that column is ever visited; the rest of each map just repeats it.` : ''}` })),
        el('div', { class: 'grid4' }, ...learnedMaps.map(m => m.el)),
        el('div', { class: 'grid4' }, ...optMaps.map(m => m.el)),
        valuePanel.el,
      ),
    ),
    observeBox([
      'Press <strong>Book Example 10.2</strong> and fast-forward 2M steps: the learned maps reproduce the threshold shape of Figure 10.5 (the better the tier, the fewer free GPUs it needs), and the exact score approaches r*.',
      'With continuous job lengths the boundary over ℓ moves with congestion: when few GPUs are free, only short or well-paid jobs are worth admitting. Tile coding gets there by generalizing across nearby lengths.',
      'Faded cells are rarely visited states (for example, many free GPUs in a busy cluster). Their values stay near the initial value, so their decisions are close to arbitrary.',
      'R̄ tracks the reward of the ε-greedy behaviour policy, so it sits below the exact score of the greedy policy.',
    ]),
  );
  syncControls();

  // --- references --------------------------------------------------------
  ctx.when(store.reference(cfg), r => {
    ref = r;
    optDecide = optimumDecider(model, { ...cfg, amp: 0 }, r);
    if (cfg.amp === 0) {
      k.ref.set(fmt(r.rStar), `exact, ${r.B}-bucket model`);
      refNote.textContent = `Reference: ${referenceLabel(r, cfg)}. Accept-all scores ${fmt(r.acceptAll)} on the same model.`;
    }
    optMaps.forEach(m => m.redraw());
    render(true);
  });
  if (cfg.amp > 0) {
    k.ref.set('…', 'Monte Carlo');
    refNote.textContent = 'A > 0: the bucketed model has no clock, so there is no exact r*. Reference: the A = 0 optimum run in the diurnal simulator (Monte Carlo, 95% CI). Exact scoring of the learned policy is unavailable.';
    ctx.when(store.reference(cfg), r => {
      ctx.when(store.longRun(cfg, { kind: 'model', policy: r.policy }, 'opt0'), iv => {
        mcRef = iv;
        k.ref.set(`${fmt(iv.mean)} ± ${fmt(iv.half)}`, 'MC, A=0 optimum in diurnal sim');
        curve.redraw();
      });
    });
  }

  // --- render ------------------------------------------------------------
  function render(force = false) {
    k.steps.set(steps.toLocaleString());
    k.rbar.set(fmt(agent.avg.rbar));
    k.roll.set(fmt(roll.value()));
    const now = performance.now();
    if (cfg.amp === 0 && (force || (steps !== lastScore.steps && now - lastScoreTime > 400))) {
      lastScoreTime = now;
      const pol = greedyOnModel(model, agent);
      const avg = evaluatePolicy(model, pol).avg;
      lastScore = { steps, avg, agree: ref ? agreement(pol, ref.policy, ref.mu) : NaN };
      if (steps > 0) exactPoints.push(steps, avg);
      k.exact.set(fmt(avg), ref ? `${fmt((100 * avg) / ref.rStar, 1)}% of r*` : '');
      k.agree.set(Number.isFinite(lastScore.agree) ? `${fmt(100 * lastScore.agree, 1)}%` : '—');
    }
    if (cfg.amp > 0) k.exact.set('n/a', 'A > 0');
    // Charts and maps are the expensive part: at most ~4 redraws per second while running.
    if (force || now - lastHeavy > 250) {
      lastHeavy = now;
      curve.redraw();
      learnedMaps.forEach(m => m.redraw());
      valuePanel.redraw();
    }
  }
  render(true);

  return () => {
    runner.destroy();
    share();
    learnedMaps.forEach(m => m.destroy());
    optMaps.forEach(m => m.destroy());
    curve.destroy();
    valuePanel.destroy();
  };
}
