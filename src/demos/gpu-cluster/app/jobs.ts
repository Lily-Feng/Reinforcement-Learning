/**
 * Client for the lab's one shared module Worker. Jobs run one at a time in a
 * queue; cancelling a queued job drops it, cancelling the running job
 * terminates the worker (the next job starts a fresh one), so no cancelled
 * computation keeps burning CPU after a module unmounts.
 */

import type { JobKind, JobMessage } from '../worker';

export interface Job<T> {
  promise: Promise<T>;
  cancel: () => void;
}

interface Pending {
  id: number;
  kind: JobKind;
  payload: unknown;
  resolve: (v: unknown) => void;
  reject: (e: unknown) => void;
  onProgress?: (fraction: number, note?: string) => void;
}

let worker: Worker | null = null;
let running: Pending | null = null;
const queue: Pending[] = [];
let nextId = 1;

export class CancelledError extends Error {
  constructor() {
    super('cancelled');
    this.name = 'CancelledError';
  }
}

function ensureWorker(): Worker {
  if (worker) return worker;
  const w = new Worker(new URL('../worker.ts', import.meta.url), { type: 'module' });
  w.onmessage = (e: MessageEvent<JobMessage>) => {
    const msg = e.data;
    if (!running || msg.id !== running.id) return;
    if (msg.type === 'progress') { running.onProgress?.(msg.fraction, msg.note); return; }
    const job = running;
    running = null;
    if (msg.type === 'result') job.resolve(msg.value);
    else job.reject(new Error(msg.error));
    pump();
  };
  w.onerror = e => {
    const job = running;
    running = null;
    worker?.terminate();
    worker = null;
    job?.reject(new Error(e.message || 'worker error'));
    pump();
  };
  worker = w;
  return w;
}

function pump(): void {
  if (running || queue.length === 0) return;
  running = queue.shift()!;
  ensureWorker().postMessage({ id: running.id, kind: running.kind, payload: running.payload });
}

export function runJob<T>(kind: JobKind, payload: unknown, onProgress?: (fraction: number, note?: string) => void): Job<T> {
  const id = nextId++;
  let entry!: Pending;
  const promise = new Promise<T>((resolve, reject) => {
    entry = { id, kind, payload, resolve: resolve as (v: unknown) => void, reject, onProgress };
  });
  queue.push(entry);
  pump();
  const cancel = () => {
    const qi = queue.indexOf(entry);
    if (qi >= 0) {
      queue.splice(qi, 1);
      entry.reject(new CancelledError());
      return;
    }
    if (running === entry) {
      running = null;
      worker?.terminate();
      worker = null;
      entry.reject(new CancelledError());
      pump();
    }
  };
  return { promise, cancel };
}

export const isCancelled = (e: unknown) => e instanceof CancelledError;
