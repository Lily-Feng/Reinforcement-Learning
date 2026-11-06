/**
 * M01 · Playground: you are the policy. Accept or reject requests (keys A / R)
 * or hand the cluster to an autopilot. Every decision is priced with the
 * exact optimal differential action values of the bucketed model.
 */

import { mulberry32 } from '../../shared/rng';
import { ACCEPT, PAY, REJECT, TIER_NAMES } from '../core/config.ts';
import { bucketCounts, createEnv, observe, step, type Env } from '../core/env.ts';
import { afterstate, countKey, reward as modelReward, stateIndex } from '../core/model.ts';
import { applyT } from '../core/exact.ts';
import { acceptAll, enterpriseOnly, handTuned, modelPolicyAutopilot, thresholdPolicy, type Policy, type Threshold } from '../core/policies.ts';
import type { Reference } from '../core/jobs.ts';
import type { ModuleContext } from '../app/context';
import { referenceLabel } from '../app/store';
import { button, card, el, fmt, moduleHeader, observeBox, range, select } from '../app/ui';
import { kpi } from './common';

type Pilot = 'off' | 'accept-all' | 'enterprise' | 'threshold' | 'optimum';

interface Local {
  pilot: Pilot;
  speed: number;
  th: Threshold;
}

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = ctx.cfg;
  const store = ctx.store;
  const L = store.local<Local>('m01', { pilot: 'off', speed: 4, th: handTuned(cfg.K) });
  if (L.th.theta.length !== 4 || Math.max(...L.th.theta) > cfg.K + 1) L.th = handTuned(cfg.K);
  const model = store.model(cfg);
  const index = new Map(model.counts.map((n, i) => [countKey(n), i]));
  let env: Env = createEnv(cfg, mulberry32(cfg.seed));
  let total = 0;
  let steps = 0;
  let regret = 0;
  let ref: Reference | null = null;
  let TU: Float64Array | null = null;
  const log: HTMLElement[] = [];

  // --- views ---------------------------------------------------------------
  const gpus = el('div', { class: 'gpu-grid', attrs: { role: 'list', 'aria-label': 'GPUs' } });
  const request = el('div', { class: 'request', attrs: { 'aria-live': 'polite' } });
  const clock = el('div', { class: 'clock' });
  const k = { avg: kpi('Your average reward'), steps: kpi('Decisions'), all: kpi('Accept-all (exact)'), ref: kpi('Reference'), regret: kpi('Cost of your decisions', '—', 'Σ q*(best) − q*(chosen)') };
  const logBox = el('ol', { class: 'decision-log' });
  const acceptBtn = button('Accept (A)', () => act(ACCEPT), 'primary big');
  const rejectBtn = button('Reject (R)', () => act(REJECT), 'big');

  const optimalCost = (a: number): { best: number; cost: number } | null => {
    if (!ref || !TU || cfg.amp > 0) return null;
    const ci = index.get(countKey(bucketCounts(env)));
    if (ci === undefined) return null;
    const s = stateIndex(model, ci, env.head.c, env.head.bucket);
    const q = (x: number) => modelReward(model, s, x) + TU![afterstate(model, s, x)]!;
    const best = observe(env).g > 0 && q(ACCEPT) > q(REJECT) + 1e-9 ? ACCEPT : REJECT;
    return { best, cost: Math.max(0, q(best) - q(a)) };
  };

  function act(a: number, auto = false): void {
    const o = observe(env);
    const judged = optimalCost(a);
    const res = step(env, a);
    total += res.reward;
    steps += 1;
    if (judged) regret += judged.cost;
    if (!auto || steps % 5 === 0) {
      const flag = res.accepted && o.g <= 2 && o.c <= 1 && o.len > cfg.median * 1.5 ? ' ⚠ long, cheap job into a nearly full cluster' : '';
      const item = el('li', {},
        el('span', { class: 'pill', attrs: { style: `border-color: var(--tier-${o.c})` }, text: TIER_NAMES[o.c]! }),
        ` ℓ=${o.len.toFixed(1)}, g=${o.g}: ${a === ACCEPT ? (res.accepted ? 'accepted' : 'no GPU free → rejected') : 'rejected'}, r = ${fmt(res.reward, 2)}`,
        judged ? el('span', { class: judged.cost > 0.05 ? 'cost bad' : 'cost', text: judged.cost > 1e-6 ? ` · cost ${fmt(judged.cost, 2)} (optimum: ${judged.best ? 'accept' : 'reject'})` : ' · optimal' }) : null,
        flag ? el('strong', { class: 'warn', text: flag }) : null);
      log.unshift(item);
      if (log.length > 12) log.pop();
      logBox.replaceChildren(...log);
    }
    render();
  }

  function render(): void {
    const o = observe(env);
    gpus.replaceChildren(...Array.from({ length: cfg.K }, (_, i) => {
      const busy = env.q[i]! > 0;
      const left = busy ? 1 / env.q[i]! : 0;
      const frac = Math.min(1, left / 60);
      const tile = el('div', { class: busy ? 'gpu busy' : 'gpu', attrs: { role: 'listitem', title: busy ? `${TIER_NAMES[env.tier[i]!]} job, ℓ = ${env.len[i]!.toFixed(1)}, expected ${left.toFixed(1)} steps left` : 'free' } },
        el('span', { class: 'ring', attrs: { style: busy ? `background: conic-gradient(var(--tier-${env.tier[i]}) ${Math.round(frac * 360)}deg, var(--grid) 0)` : '' } }),
        el('span', { class: 'gpu-label', text: busy ? `~${left.toFixed(0)}` : 'free' }));
      return tile;
    }));
    const density = PAY[o.c]! / o.len;
    request.replaceChildren(
      el('span', { class: 'hint', text: 'Head of the queue' }),
      el('strong', { class: 'req-tier', attrs: { style: `color: var(--tier-${o.c})` }, text: `${TIER_NAMES[o.c]} · $${PAY[o.c]}` }),
      el('span', { text: `job length ℓ = ${o.len.toFixed(1)} steps` }),
      el('span', { text: `revenue density pay/ℓ = ${fmt(density, 3)}` }),
      el('span', { class: 'hint', text: `${o.g} of ${cfg.K} GPUs free · ρ = ${fmt(o.rho, 2)} expected completions next step` }));
    clock.hidden = cfg.amp === 0;
    if (cfg.amp > 0) {
      const minutes = Math.round(o.phase * 24 * 60);
      clock.replaceChildren(el('span', { class: 'dial', attrs: { style: `--angle: ${o.phase * 360}deg` } }), el('span', { text: `${String(Math.floor(minutes / 60)).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}` }));
    }
    acceptBtn.disabled = o.g === 0;
    k.avg.set(steps ? fmt(total / steps) : '—', `${steps} steps`);
    k.steps.set(String(steps));
    k.regret.set(ref && cfg.amp === 0 ? fmt(regret, 2) : 'n/a');
  }

  // --- autopilot -------------------------------------------------------------
  const pilotPolicy = (): Policy | null => {
    switch (L.pilot) {
      case 'accept-all': return acceptAll;
      case 'enterprise': return enterpriseOnly;
      case 'threshold': return thresholdPolicy(L.th);
      case 'optimum': return ref ? modelPolicyAutopilot(model, ref.policy) : null;
      default: return null;
    }
  };
  let raf = 0;
  let last = 0;
  let acc = 0;
  let running = false;
  const loop = (now: number) => {
    raf = 0;
    if (!running) return;
    const pol = pilotPolicy();
    if (pol) {
      acc += ((now - (last || now)) / 1000) * L.speed;
      last = now;
      let n = 0;
      const t0 = performance.now();
      while (acc >= 1 && performance.now() - t0 < 8) { act(pol(observe(env), env), true); acc -= 1; n += 1; }
      if (acc > 50) acc = 0;
    }
    raf = requestAnimationFrame(loop);
  };
  const playBtn = button('Run autopilot', () => setRunning(!running));
  const setRunning = (v: boolean) => {
    running = v && L.pilot !== 'off';
    playBtn.textContent = running ? 'Pause autopilot' : 'Run autopilot';
    playBtn.setAttribute('aria-pressed', String(running));
    last = 0;
    if (running && !raf) raf = requestAnimationFrame(loop);
  };
  const thBox = el('div', { class: 'controls' });
  const buildTh = () => {
    thBox.replaceChildren(
      ...TIER_NAMES.map((name, c) => range(`${name}: accept if g ≥`, { min: 1, max: cfg.K + 1, step: 1, value: L.th.theta[c]!, format: v => (v > cfg.K ? 'never' : String(v)) }, v => { L.th.theta[c] = v; }).el),
      ...TIER_NAMES.map((name, c) => range(`${name}: and ℓ ≤`, { min: 2, max: 121, step: 1, value: Math.min(121, L.th.lambda[c]!), format: v => (v > 120 ? '∞' : String(v)) }, v => { L.th.lambda[c] = v > 120 ? Infinity : v; }).el));
    thBox.hidden = L.pilot !== 'threshold';
  };
  buildTh();
  const pilotSel = select<Pilot>('Autopilot', [
    { value: 'off', label: 'Off (you decide)' }, { value: 'accept-all', label: 'Accept all' }, { value: 'enterprise', label: 'Enterprise only' },
    { value: 'threshold', label: 'Threshold (sliders)' }, { value: 'optimum', label: 'Bucketed optimum (reads true state)' },
  ], L.pilot, v => { L.pilot = v; thBox.hidden = v !== 'threshold'; if (v === 'off') setRunning(false); });
  const speed = range('Autopilot speed (steps / s)', { min: 1, max: 200, step: 1, value: L.speed }, v => { L.speed = v; });

  const onKey = (e: KeyboardEvent) => {
    const tag = (e.target as HTMLElement | null)?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'a' || e.key === 'A') { e.preventDefault(); if (observe(env).g > 0) act(ACCEPT); }
    if (e.key === 'r' || e.key === 'R') { e.preventDefault(); act(REJECT); }
  };
  document.addEventListener('keydown', onKey);

  const refNote = el('p', { class: 'hint' });
  root.append(
    moduleHeader('m01', 'Playground: you are the policy', 'Each step one request reaches the head of the queue. Accepting takes a free GPU and pays the tier price; the GPU stays busy for a random time whose mean is the job length ℓ. Rejecting pays nothing. Your goal is the highest <strong>long-run reward per step</strong>.'),
    el('style', { text: `
      .gpu-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(64px, 1fr)); gap: .4rem; }
      .gpu { position: relative; aspect-ratio: 1; border: 1px solid var(--border); border-radius: 10px; background: var(--inset); display: grid; place-items: center; }
      .gpu .ring { position: absolute; inset: 8px; border-radius: 50%; }
      .gpu .ring::after { content: ''; position: absolute; inset: 6px; border-radius: 50%; background: var(--surface); }
      .gpu:not(.busy) .ring { display: none; }
      .gpu-label { position: relative; font-size: .75rem; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--muted); }
      .request { display: flex; flex-direction: column; gap: .15rem; }
      .req-tier { font-size: 1.3rem; }
      .decision-log { margin: 0; padding-left: 1.2rem; font-size: .78rem; display: flex; flex-direction: column; gap: .15rem; }
      .cost { color: var(--muted); } .cost.bad { color: var(--danger); font-weight: 700; } .warn { color: var(--danger); }
      .clock { display: flex; align-items: center; gap: .5rem; font-variant-numeric: tabular-nums; }
      .clock[hidden] { display: none; }
      .dial { width: 32px; height: 32px; border-radius: 50%; border: 2px solid var(--border); position: relative; }
      .dial::after { content: ''; position: absolute; left: 50%; top: 3px; width: 2px; height: 12px; background: var(--accent); transform-origin: 50% 12px; transform: translateX(-50%) rotate(var(--angle)); }
    ` }),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Your move', request, clock, el('div', { class: 'buttons' }, acceptBtn, rejectBtn), button('New cluster (same seed)', () => { env = createEnv(cfg, mulberry32(cfg.seed)); total = 0; steps = 0; regret = 0; log.length = 0; logBox.replaceChildren(); render(); }, 'link')),
        card('Autopilot', pilotSel.el, speed.el, playBtn, thBox),
        card('Cluster', el('p', { class: 'hint', text: 'K, m, σ, κ, A and the seed live in the shared settings.' }), button('Open cluster settings', ctx.openSettings)),
      ),
      el('div', { class: 'panels' },
        el('div', { class: 'kpis' }, k.avg.el, k.steps.el, k.all.el, k.ref.el, k.regret.el),
        refNote,
        card(`GPUs (ring = expected time left, 1/q)`, gpus),
        card('Recent decisions', el('p', { class: 'hint', text: 'Cost is how much long-run value a decision gave up, from the optimal differential action values of the bucketed model (0 = the optimal choice).' }), logBox),
      ),
    ),
    observeBox([
      'Run the <strong>Accept all</strong> autopilot for a while: its average falls well below the reference. A full cluster turns away Enterprise requests.',
      'The expensive mistake is accepting a <em>long, cheap</em> job when only one or two GPUs are free: it pays $1–2 and blocks a GPU that an Enterprise job would have paid $8 for.',
      'Short cheap jobs are often fine. They free their GPU quickly, so their revenue density pay/ℓ is high. Watch the cost column stay near 0 for them.',
    ]),
  );
  render();

  ctx.when(store.reference(cfg), r => {
    ref = r;
    TU = applyT(model, r.U);
    k.all.set(fmt(r.acceptAll), `exact, ${r.B}-bucket model`);
    if (cfg.amp === 0) k.ref.set(fmt(r.rStar), `exact, ${r.B}-bucket model`);
    else k.ref.set('n/a', 'A > 0');
    refNote.textContent = cfg.amp === 0
      ? `${referenceLabel(r, cfg)}. Your average uses the continuous simulator, so expect noise of a few hundredths over ~1,000 steps.`
      : 'A > 0: the bucketed model has no clock. The reference and decision costs are unavailable; the optimum autopilot uses the A = 0 policy.';
    render();
  });

  return () => {
    running = false;
    if (raf) cancelAnimationFrame(raf);
    document.removeEventListener('keydown', onKey);
  };
}
