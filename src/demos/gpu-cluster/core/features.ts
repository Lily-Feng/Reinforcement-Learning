/**
 * Feature vectors x(o, a) for q̂(o, a, w) = w·x(o, a). All features are binary,
 * so q̂ is the sum of the active weights.
 *
 * Tile-coded groups, each with integer inputs [c, a]:
 *   G1  g/K × normalized log ℓ_head       8 tilings
 *   G2  g/K × ρ/ρ_max                     4 tilings
 *   G3  g/K × φ (wraps at midnight)       4 tilings, meant for A > 0
 *   G4  g/K × time-to-go/H                4 tilings, episodic (one shift) only
 * Each input is scaled to [0, tilesPerDim).
 *
 * Tabular one-hot over (g, c, a) is only faithful when σ = 0, where every job
 * has the same length and (g, c) is the whole state.
 */

import { LEN_MAX, LEN_MIN, type Buckets } from './buckets.ts';
import type { ClusterConfig } from './config.ts';
import type { Observation } from './env.ts';
import { createIHT, tiles, tileswrap, type IHT } from './tiles.ts';

export type GroupId = 'g1' | 'g2' | 'g3' | 'g4';

export interface FeatureConfig {
  kind: 'tiles' | 'tabular';
  groups: Record<GroupId, boolean>;
  tilings: Record<GroupId, number>;
  tilesPerDim: number;
  /** Wrap the time-of-day dimension of G3. */
  wrap: boolean;
  ihtSize: number;
}

export const GROUP_LABELS: Record<GroupId, string> = {
  g1: 'G1 · free GPUs × log job length',
  g2: 'G2 · free GPUs × ρ',
  g3: 'G3 · free GPUs × time of day',
  g4: 'G4 · free GPUs × time to go',
};

export function defaultFeatures(cfg: ClusterConfig, episodic = false): FeatureConfig {
  return {
    kind: 'tiles',
    groups: { g1: true, g2: true, g3: cfg.amp > 0, g4: episodic },
    tilings: { g1: 8, g2: 4, g3: 4, g4: 4 },
    tilesPerDim: 4,
    wrap: true,
    ihtSize: 8192,
  };
}

export const TABULAR: FeatureConfig = {
  kind: 'tabular', groups: { g1: false, g2: false, g3: false, g4: false },
  tilings: { g1: 0, g2: 0, g3: 0, g4: 0 }, tilesPerDim: 1, wrap: false, ihtSize: 0,
};

export interface Featurizer {
  spec: FeatureConfig;
  /** Length of the weight vector. */
  size: number;
  /** Active (nonzero) features per (o, a): step sizes are reported as α × this. */
  active: number;
  iht: IHT | null;
  indices: (o: Observation, a: number) => number[];
  /** Per-group indices, for displaying which tiles are active. */
  groupIndices: (o: Observation, a: number) => { group: GroupId; indices: number[] }[];
  /** The scaled 2-D input of a group, for drawing the grid. */
  inputs: (group: GroupId, o: Observation) => [number, number];
}

/** Log job length mapped onto [0, 1]. */
export const normLogLength = (len: number) => (Math.log(len) - Math.log(LEN_MIN)) / (Math.log(LEN_MAX) - Math.log(LEN_MIN));

export function rhoMax(cfg: ClusterConfig, buckets: Buckets): number {
  return 1.5 * cfg.K * buckets.meanQ;
}

export const ORDER: GroupId[] = ['g1', 'g2', 'g3', 'g4'];

export function createFeaturizer(cfg: ClusterConfig, buckets: Buckets, spec: FeatureConfig): Featurizer {
  if (spec.kind === 'tabular') {
    return {
      spec, size: (cfg.K + 1) * 8, active: 1, iht: null,
      indices: (o, a) => [(o.g * 4 + o.c) * 2 + a],
      groupIndices: (o, a) => [{ group: 'g1', indices: [(o.g * 4 + o.c) * 2 + a] }],
      inputs: (_g, o) => [o.g, 0],
    };
  }
  const iht = createIHT(spec.ihtSize);
  const T = spec.tilesPerDim;
  const rmax = rhoMax(cfg, buckets);
  const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
  const inputs = (group: GroupId, o: Observation): [number, number] => {
    const x = clamp01(o.g / cfg.K) * T;
    switch (group) {
      case 'g1': return [x, clamp01(normLogLength(o.len)) * T];
      case 'g2': return [x, clamp01(o.rho / rmax) * T];
      case 'g3': return [x, (((o.phase % 1) + 1) % 1) * T];
      default: return [x, clamp01(Number.isFinite(o.togo) ? o.togo / cfg.H : 1) * T];
    }
  };
  const enabled = ORDER.filter(g => spec.groups[g]);
  const groupIndices = (o: Observation, a: number) => enabled.map((group, gi) => {
    const ints = [gi, o.c, a];
    const indices = group === 'g3' && spec.wrap
      ? tileswrap(iht, spec.tilings.g3, inputs(group, o), [null, T], ints)
      : tiles(iht, spec.tilings[group], inputs(group, o), ints);
    return { group, indices };
  });
  return {
    spec, size: spec.ihtSize, iht,
    active: enabled.reduce((s, g) => s + spec.tilings[g], 0),
    indices: (o, a) => groupIndices(o, a).flatMap(r => r.indices),
    groupIndices, inputs,
  };
}
