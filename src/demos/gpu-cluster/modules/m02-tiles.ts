/**
 * M02 · How the agent sees the cluster: tile coding of (g, ℓ), (g, ρ) and
 * (g, φ), one click-to-teach update with visible generalization, and two
 * different true states with the same observation (aliasing).
 */

import { attachHeatmapHover, diverging, drawHeatmap, type HeatmapLayout } from '../../shared/heatmap';
import { ACCEPT, TIER_NAMES } from '../core/config.ts';
import type { Observation } from '../core/env.ts';
import { createFeaturizer, defaultFeatures, GROUP_LABELS, rhoMax, type FeatureConfig, type GroupId } from '../core/features.ts';
import { LEN_MAX, LEN_MIN, makeBuckets } from '../core/buckets.ts';
import { qValue } from '../core/agents.ts';
import type { ModuleContext } from '../app/context';
import { button, canvasPanel, card, check, el, fmt, moduleHeader, observeBox, range, select, theme } from '../app/ui';

interface Local {
  group: GroupId;
  tilings: number;
  tiles: number;
  wrap: boolean;
  alpha: number;
  target: number;
  tier: number;
}

const RES = 64;

export function mount(root: HTMLElement, ctx: ModuleContext): () => void {
  const cfg = ctx.cfg;
  const L = ctx.store.local<Local>('m02', { group: 'g1', tilings: 8, tiles: 4, wrap: true, alpha: 0.5, target: 10, tier: 0 });
  const buckets = makeBuckets(cfg, cfg.B);
  const rmax = rhoMax(cfg, buckets);

  let spec: FeatureConfig;
  let feat: ReturnType<typeof createFeaturizer>;
  let w: Float64Array;
  let clicked: [number, number] | null = null;
  let lastUpdate = '';

  const build = () => {
    spec = defaultFeatures(cfg);
    spec.groups = { g1: false, g2: false, g3: false, g4: false };
    spec.groups[L.group] = true;
    spec.tilings[L.group] = L.tilings;
    spec.tilesPerDim = L.tiles;
    spec.wrap = L.wrap;
    feat = createFeaturizer(cfg, buckets, spec);
    w = new Float64Array(feat.size);
    clicked = null;
    lastUpdate = '';
  };
  build();

  /** Observation at scaled grid point (x, y) ∈ [0, T)². */
  const obsAt = (x: number, y: number): Observation => {
    const T = L.tiles;
    const g = (x / T) * cfg.K;
    const base: Observation = { g, c: L.tier, len: cfg.median, rho: (cfg.K - g) * buckets.meanQ, phase: 0.5, togo: Infinity, bucket: 0 };
    const u = y / T;
    if (L.group === 'g1') base.len = Math.exp(Math.log(LEN_MIN) + u * (Math.log(LEN_MAX) - Math.log(LEN_MIN)));
    else if (L.group === 'g2') base.rho = u * rmax;
    else base.phase = u;
    return base;
  };
  const yName = () => (L.group === 'g1' ? 'normalized log job length' : L.group === 'g2' ? 'ρ / ρ_max' : 'time of day φ');

  let layout: HeatmapLayout = { cellAt: () => null };
  const readout = el('p', { class: 'readout' });
  const grid = canvasPanel(null, 360, canvas => {
    const t = theme();
    const T = L.tiles;
    const vals: number[] = [];
    for (let r = 0; r < RES; r += 1) for (let c = 0; c < RES; c += 1) vals.push(qValue(w, feat.indices(obsAt(((c + 0.5) / RES) * T, ((RES - r - 0.5) / RES) * T), ACCEPT)));
    const m = Math.max(1e-9, ...vals.map(Math.abs));
    const ramp = diverging(t.divLow, t.divMid, t.divHigh);
    layout = drawHeatmap(canvas, {
      rows: RES, cols: RES,
      color: (r, c) => ramp(0.5 + vals[r * RES + c]! / (2 * m)),
      rowLabel: r => (r % 16 === 0 ? fmt(((RES - r) / RES), 2) : ''),
      colLabel: c => (c % 16 === 0 ? fmt((c / RES) * cfg.K, 0) : ''), colLabelEvery: 1,
      xLabel: 'free GPUs g', yLabel: yName(),
      legend: { min: fmt(-m, 1), max: fmt(m, 1), ramp },
      textColor: t.text,
      overlay: (ctx2, cell) => {
        // Tile boundaries of tiling 0, dashed.
        const x0 = cell(0, 0).x;
        const y0 = cell(0, 0).y;
        const W = cell(0, RES - 1).x + cell(0, 0).w - x0;
        const Hh = cell(RES - 1, 0).y + cell(0, 0).h - y0;
        const toPx = (x: number, y: number) => [x0 + (x / T) * W, y0 + (1 - y / T) * Hh] as const;
        ctx2.strokeStyle = t.text;
        ctx2.globalAlpha = 0.35;
        ctx2.setLineDash([3, 3]);
        for (let k = 1; k < T; k += 1) {
          const [px] = toPx(k, 0);
          const [, py] = toPx(0, k);
          ctx2.beginPath(); ctx2.moveTo(px, y0); ctx2.lineTo(px, y0 + Hh); ctx2.stroke();
          ctx2.beginPath(); ctx2.moveTo(x0, py); ctx2.lineTo(x0 + W, py); ctx2.stroke();
        }
        ctx2.setLineDash([]);
        ctx2.globalAlpha = 1;
        if (!clicked) return;
        // The active tile in each tiling, following tiles3's offsets: tiling i shifts dim d by i·(1 + 2d)/n.
        const n = L.tilings;
        for (let i = 0; i < n; i += 1) {
          const bx = i % n;
          const by = (3 * i) % n;
          const cx = Math.floor((Math.floor(clicked[0] * n) + bx) / n);
          const cy = Math.floor((Math.floor(clicked[1] * n) + by) / n);
          const xa = cx - bx / n;
          const ya = cy - by / n;
          const [p0x, p0y] = toPx(xa, ya + 1);
          const [p1x, p1y] = toPx(xa + 1, ya);
          ctx2.strokeStyle = t.series[i % 6]!;
          ctx2.lineWidth = 2;
          ctx2.strokeRect(p0x, p0y, p1x - p0x, p1y - p0y);
          if (L.group === 'g3' && L.wrap && (ya < 0 || ya + 1 > T)) {
            // The same tile seen across midnight.
            const [q0x, q0y] = toPx(xa, ya + 1 + (ya < 0 ? T : -T));
            const [q1x, q1y] = toPx(xa + 1, ya + (ya < 0 ? T : -T));
            ctx2.setLineDash([2, 2]);
            ctx2.strokeRect(q0x, q0y, q1x - q0x, q1y - q0y);
            ctx2.setLineDash([]);
          }
        }
        const [mx, my] = toPx(clicked[0], clicked[1]);
        ctx2.fillStyle = t.ink;
        ctx2.beginPath(); ctx2.arc(mx, my, 4, 0, Math.PI * 2); ctx2.fill();
      },
    });
  }, { note: readout });

  const offHover = attachHeatmapHover(grid.canvas, () => layout, cell => {
    if (!cell) { readout.textContent = lastUpdate; return; }
    const x = ((cell.col + 0.5) / RES) * L.tiles;
    const y = ((RES - cell.row - 0.5) / RES) * L.tiles;
    const o = obsAt(x, y);
    readout.textContent = `g ≈ ${fmt(o.g, 1)}, ${L.group === 'g1' ? `ℓ ≈ ${fmt(o.len, 1)}` : L.group === 'g2' ? `ρ ≈ ${fmt(o.rho, 2)}` : `φ ≈ ${fmt(o.phase, 3)} (${Math.floor(o.phase * 24)}:${String(Math.floor((o.phase * 24 * 60) % 60)).padStart(2, '0')})`}: q̂(·, ${TIER_NAMES[L.tier]}, accept) = ${fmt(qValue(w, feat.indices(o, ACCEPT)), 3)}`;
  }, cell => {
    clicked = [((cell.col + 0.5) / RES) * L.tiles, ((RES - cell.row - 0.5) / RES) * L.tiles];
    lastUpdate = 'Point selected: its active tile in every tiling is outlined. Press "Teach this point".';
    grid.redraw();
  });

  const teach = () => {
    if (!clicked) { lastUpdate = 'Click a point first.'; readout.textContent = lastUpdate; return; }
    const o = obsAt(clicked[0], clicked[1]);
    const idx = feat.indices(o, ACCEPT);
    const before = qValue(w, idx);
    const dw = (L.alpha / feat.active) * (L.target - before);
    for (const i of idx) w[i]! += dw;
    const after = qValue(w, idx);
    lastUpdate = `Update at the clicked point: q̂ ${fmt(before, 3)} → ${fmt(after, 3)} (target ${L.target}; Δw = ${fmt(dw, 4)} on each of ${feat.active} active weights; α × active = ${L.alpha}).`;
    readout.textContent = lastUpdate;
    grid.redraw();
  };

  // --- aliasing --------------------------------------------------------------
  const aliasA = [16, 16, 16, 16];
  const aliasB = [6, 36, 36, 36];
  const rhoOf = (ls: number[]) => ls.reduce((s, l) => s + 1 / l, 0);
  const freeAfter = (ls: number[], k: number) => ls.reduce((s, l) => s + (1 - (1 - 1 / l) ** k), 0);
  const aliasRows = [aliasA, aliasB].map((ls, i) => el('tr', {},
    el('td', { class: 'l', text: i ? 'State B' : 'State A' }), el('td', { class: 'l', text: ls.map(l => `ℓ=${l}`).join(', ') }),
    el('td', { text: String(cfg.K - ls.length) }), el('td', { text: fmt(rhoOf(ls), 4) }),
    el('td', { text: fmt(freeAfter(ls, 10), 3) }), el('td', { text: fmt(freeAfter(ls, 30), 3) })));

  const restart = () => { build(); syncCtl(); grid.redraw(); };
  const groupSel = select<GroupId>('Feature group', (['g1', 'g2', 'g3'] as GroupId[]).map(g => ({ value: g, label: GROUP_LABELS[g] })), L.group, v => { L.group = v; restart(); });
  const wrap = check('Wrap time of day (G3)', L.wrap, v => { L.wrap = v; restart(); });
  const syncCtl = () => { wrap.el.hidden = L.group !== 'g3'; };

  root.append(
    moduleHeader('m02', 'How the agent sees the cluster (tile coding)', 'The agent never sees the per-GPU job list. It sees o = (g, c, ℓ, ρ, φ) and turns each feature group into overlapping tilings. q̂(o, a) is the sum of one weight per active tile. Inputs are scaled to [0, tiles per dimension); the input that is not plotted is held at a typical value.'),
    el('div', { class: 'mod-grid' },
      el('div', { class: 'controls' },
        card('Tiling', groupSel.el,
          range('Tilings', { min: 1, max: 16, step: 1, value: L.tilings }, v => { L.tilings = v; restart(); }).el,
          range('Tiles per dimension', { min: 2, max: 8, step: 1, value: L.tiles }, v => { L.tiles = v; restart(); }).el,
          wrap.el,
          select('Tier for q̂(·, c, accept)', TIER_NAMES.map((n, c) => ({ value: String(c), label: n })), String(L.tier), v => { L.tier = Number(v); grid.redraw(); }).el),
        card('Teach this point',
          range('α × active features', { min: 0.05, max: 1, step: 0.05, value: L.alpha, format: v => fmt(v, 2) }, v => { L.alpha = v; }).el,
          range('Target value', { min: -20, max: 20, step: 1, value: L.target }, v => { L.target = v; }).el,
          el('div', { class: 'buttons' }, button('Teach this point', teach, 'primary'), button('Clear weights', () => { w.fill(0); lastUpdate = ''; grid.redraw(); }))),
      ),
      el('div', { class: 'panels' },
        card('q̂(o, c, accept) over the chosen feature group: click a point, then teach it', grid.el),
        card('Aliasing: two true states, one observation',
          el('p', { class: 'hint', text: 'Both clusters have 4 busy GPUs and the same ρ = Σ 1/ℓ, so with the same head request they produce exactly the same observation. Their futures differ.' }),
          el('div', { class: 'table-scroll' }, el('table', {},
            el('thead', {}, el('tr', {}, ...['', 'busy jobs', 'g', 'ρ', 'E[freed in 10 steps]', 'E[freed in 30 steps]'].map(h => el('th', { text: h })))),
            el('tbody', {}, ...aliasRows)))),
      ),
    ),
    observeBox([
      'Click a point and teach it: q̂ changes in a <em>neighbourhood</em>, the union of its active tiles, not just at the point. That is generalization.',
      'More tilings at the same tiles per dimension give finer resolution (more distinct regions) at the same generalization width. More tiles per dimension narrow the width.',
      'Pick G3 with wrap on and teach a point near midnight: the change also appears just after 00:00. Turn wrap off and it stays on one side.',
      `The observation is a lossy summary. States A and B look identical to the agent but free GPUs at different rates. This is Section 10.4's situation: features that cannot distinguish states.`,
    ]),
  );
  syncCtl();
  return () => { offHover(); grid.destroy(); };
}
