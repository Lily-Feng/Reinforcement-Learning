/**
 * Shared lab state: one cluster configuration edited from the settings
 * drawer, cached references keyed by a hash of the config, the agent trained
 * in M13 (reused by M11, M15, M17), and per-module local settings that
 * survive a remount.
 */

import type { ContinuingAgent } from '../core/agents.ts';
import { configKey, DEFAULT_CONFIG, type ClusterConfig } from '../core/config.ts';
import type { PolicySpec, Reference, ShiftReference } from '../core/jobs.ts';
import { buildModel, type Model } from '../core/model.ts';
import type { Interval } from '../core/stats.ts';
import { runJob } from './jobs';

const STORAGE_KEY = 'gpu-lab:v1';

export interface LabAgent {
  agent: ContinuingAgent;
  cfg: ClusterConfig;
  steps: number;
  label: string;
}

type Listener = (cfg: ClusterConfig) => void;

export interface Store {
  readonly config: ClusterConfig;
  set: (patch: Partial<ClusterConfig>) => void;
  reset: () => void;
  subscribe: (fn: Listener) => () => void;
  model: (cfg?: ClusterConfig) => Model;
  reference: (cfg?: ClusterConfig) => Promise<Reference>;
  shift: (cfg?: ClusterConfig) => Promise<ShiftReference>;
  longRun: (cfg: ClusterConfig, spec: PolicySpec, specKey: string, steps?: number) => Promise<Interval>;
  labAgent: LabAgent | null;
  local: <T extends object>(id: string, defaults: T) => T;
  lastModule: string | null;
  rememberModule: (id: string) => void;
}

function load(): { config?: Partial<ClusterConfig>; module?: string } {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function save(data: { config: ClusterConfig; module: string | null }): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {
    /* storage blocked: the lab still works, it just forgets */
  }
}

/** Keeps a config inside the ranges the reference solvers support. */
export function sanitize(cfg: ClusterConfig): ClusterConfig {
  const clamp = (v: number, lo: number, hi: number, d: number) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d);
  const base: number[] = Array.isArray(cfg.base) && cfg.base.length === 4 ? cfg.base.map(v => clamp(v, 0, 1, 0.25)) : [...DEFAULT_CONFIG.base];
  if (!base.some(v => v > 0)) base.fill(0.25);
  const sigma = clamp(cfg.sigma, 0, 1.2, DEFAULT_CONFIG.sigma);
  return {
    K: Math.round(clamp(cfg.K, 4, 16, 10)),
    median: clamp(cfg.median, 4, 40, 16),
    sigma,
    kappa: clamp(cfg.kappa, 0, 0.5, 0),
    amp: clamp(cfg.amp, 0, 1, 0),
    base,
    B: sigma === 0 ? 1 : Math.round(clamp(cfg.B, 1, 4, 3)),
    H: Math.round(clamp(cfg.H, 24, 96, 48)),
    seed: Math.round(clamp(cfg.seed, 0, 2 ** 31, 1)),
  };
}

export function createStore(): Store {
  const saved = load();
  let config = sanitize({ ...DEFAULT_CONFIG, ...(saved.config ?? {}) });
  let lastModule: string | null = saved.module ?? null;
  const listeners = new Set<Listener>();
  const models = new Map<string, Model>();
  const refs = new Map<string, Promise<Reference>>();
  const shifts = new Map<string, Promise<ShiftReference>>();
  const runs = new Map<string, Promise<Interval>>();
  const locals = new Map<string, object>();

  const cached = <T>(map: Map<string, Promise<T>>, key: string, make: () => Promise<T>): Promise<T> => {
    const hit = map.get(key);
    if (hit) return hit;
    const p = make();
    map.set(key, p);
    p.catch(() => map.delete(key));
    return p;
  };

  const store: Store = {
    get config() { return config; },
    set(patch) {
      const next = sanitize({ ...config, ...patch });
      if (configKey(next) === configKey(config) && next.seed === config.seed) return;
      config = next;
      save({ config, module: lastModule });
      listeners.forEach(fn => fn(config));
    },
    reset() { store.set({ ...DEFAULT_CONFIG }); },
    subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    model(cfg = config) {
      const key = configKey(cfg);
      let m = models.get(key);
      if (!m) {
        m = buildModel(cfg);
        if (models.size > 6) models.clear();
        models.set(key, m);
      }
      return m;
    },
    reference(cfg = config) {
      return cached(refs, configKey({ ...cfg, amp: 0, H: 0 }), () => runJob<Reference>('reference', { cfg: { ...cfg, amp: 0 } }).promise);
    },
    shift(cfg = config) {
      return cached(shifts, configKey(cfg), () => runJob<ShiftReference>('shift', { cfg }).promise);
    },
    longRun(cfg, spec, specKey, steps = 400000) {
      return cached(runs, `${configKey(cfg)}#${specKey}#${steps}#${cfg.seed}`, () => runJob<Interval>('longRun', { cfg, policy: spec, steps, seed: cfg.seed }).promise);
    },
    labAgent: null,
    local<T extends object>(id: string, defaults: T): T {
      let v = locals.get(id) as T | undefined;
      if (!v) { v = { ...defaults }; locals.set(id, v); }
      return v;
    },
    get lastModule() { return lastModule; },
    rememberModule(id) { lastModule = id; save({ config, module: id }); },
  };
  return store;
}

/** How a reference r* is labelled, so the reader always knows where it came from. */
export function referenceLabel(ref: Reference, cfg: ClusterConfig): string {
  if (cfg.amp > 0) return `no exact r* when A > 0 (the bucketed model has no clock)`;
  return `r* ≈ ${ref.rStar.toFixed(3)} (exact on ${ref.B}-bucket model)`;
}
