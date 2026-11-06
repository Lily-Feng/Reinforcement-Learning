/// <reference lib="webworker" />
/**
 * The lab's one module Worker. Each message names a job from core/jobs.ts;
 * progress and the result are posted back with the job id. Cancellation is
 * handled by the client terminating this worker.
 */

import {
  computeChatter, computeCompare, computeFamily, computeLongRun, computeReference, computeReplicas,
  computeShift, computeSweep, computeTargets,
} from './core/jobs.ts';

const jobs = {
  reference: (p: any, progress: (f: number, n?: string) => void) => computeReference(p.cfg, progress),
  shift: (p: any) => computeShift(p.cfg),
  longRun: (p: any, progress: (f: number, n?: string) => void) => computeLongRun(p.cfg, p.policy, p.steps, p.seed, progress),
  sweep: (p: any, progress: (f: number, n?: string) => void) => computeSweep(p.cfg, p.ns, p.alphas, p.runs, p.shifts, p.seed, progress, p.q0, p.eps),
  compare: (p: any, progress: (f: number, n?: string) => void) => computeCompare(p.cfg, p.gammas, p.steps, p.alpha, p.seed, progress),
  chatter: (p: any, progress: (f: number, n?: string) => void) => computeChatter(p.cfg, p.kind, p.alpha, p.eps, p.steps, p.every, p.seed, progress),
  family: (p: any, progress: (f: number, n?: string) => void) => computeFamily(p.cfg, p.gammas, progress),
  targets: (p: any, progress: (f: number, n?: string) => void) => computeTargets(p.cfg, p.ns, p.samples, p.seed, progress),
  replicas: (p: any, progress: (f: number, n?: string) => void) => computeReplicas(p.cfg, p.replicas, p.T, p.seed, progress),
};

export type JobKind = keyof typeof jobs;

export type JobMessage =
  | { id: number; type: 'progress'; fraction: number; note?: string }
  | { id: number; type: 'result'; value: unknown }
  | { id: number; type: 'error'; error: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<{ id: number; kind: JobKind; payload: unknown }>) => {
  const { id, kind, payload } = e.data;
  let last = 0;
  const progress = (fraction: number, note?: string) => {
    const now = performance.now();
    if (now - last < 80 && fraction < 1) return;
    last = now;
    ctx.postMessage({ id, type: 'progress', fraction, note } satisfies JobMessage);
  };
  try {
    const value = jobs[kind](payload, progress);
    ctx.postMessage({ id, type: 'result', value } satisfies JobMessage);
  } catch (err) {
    ctx.postMessage({ id, type: 'error', error: err instanceof Error ? err.message : String(err) } satisfies JobMessage);
  }
};
