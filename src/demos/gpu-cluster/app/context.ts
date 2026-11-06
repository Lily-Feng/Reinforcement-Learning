/**
 * What a module receives when mounted. Jobs started and subscriptions made
 * through the context are cancelled automatically on unmount, on top of the
 * module's own cleanup.
 */

import type { ClusterConfig } from '../core/config.ts';
import type { JobKind } from '../worker';
import { isCancelled, runJob, type Job } from './jobs';
import type { Store } from './store';
import { disposer, prefersReducedMotion } from './ui';

export interface ModuleContext {
  store: Store;
  readonly cfg: ClusterConfig;
  embed: boolean;
  reducedMotion: boolean;
  go: (id: string) => void;
  openSettings: () => void;
  /** Runs a worker job owned by this module (cancelled on unmount). */
  job: <T>(kind: JobKind, payload: unknown, onProgress?: (f: number, note?: string) => void) => Job<T>;
  /** Resolves a promise unless the module has unmounted; ignores cancellations. */
  when: <T>(p: Promise<T>, fn: (v: T) => void, onError?: (e: unknown) => void) => void;
  onDispose: (fn: () => void) => void;
  alive: () => boolean;
}

export function createContext(store: Store, opts: { embed: boolean; go: (id: string) => void; openSettings: () => void }) {
  const d = disposer();
  let alive = true;
  const ctx: ModuleContext = {
    store,
    get cfg() { return store.config; },
    embed: opts.embed,
    reducedMotion: prefersReducedMotion(),
    go: opts.go,
    openSettings: opts.openSettings,
    job: (kind, payload, onProgress) => {
      const j = runJob(kind, payload, onProgress);
      d.add(j.cancel);
      return j as never;
    },
    when: (p, fn, onError) => {
      p.then(v => { if (alive) fn(v); }, e => { if (alive && !isCancelled(e)) (onError ?? console.error)(e); });
    },
    onDispose: fn => d.add(fn),
    alive: () => alive,
  };
  return { ctx, dispose: () => { alive = false; d.run(); } };
}
