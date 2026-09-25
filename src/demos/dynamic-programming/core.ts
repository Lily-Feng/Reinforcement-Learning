/** Pure, deterministic DP engine. Action order and in-place sweeps match cs394r-pa2. */
export const ACTIONS = ['Left', 'Up', 'Right', 'Down'];
export const ARROWS = ['←', '↑', '→', '↓'];
export type Mode = 'value' | 'evaluation' | 'policy';
export interface Config { size: number; gamma: number; theta: number; mode: Mode }
export interface Transition { prob: number; next: number; reward: number; done: boolean }
export interface Backup { state: number; old: number; value: number; q: number[]; future: number[]; weights: number[] }
export interface Snapshot {
  v: number[]; q: number[][]; policy: number[][]; cursor: number; sweeps: number;
  delta: number; lastDelta: number | null; phase: 'evaluate' | 'improve' | 'done';
  improvements: number; changes: number | null; trace: Backup | null; history: number[];
}
export function transition(size: number, state: number, action: number): Transition {
  const terminal = size * size - 1;
  if (state === terminal) return { prob: 1, next: state, reward: 0, done: true };
  const row = Math.floor(state / size), col = state % size;
  const dr = [0, -1, 0, 1][action], dc = [-1, 0, 1, 0][action];
  const r = row + dr, c = col + dc;
  const next = r < 0 || c < 0 || r >= size || c >= size ? state : r * size + c;
  return { prob: 1, next, reward: -1, done: next === terminal };
}
export function initial(config: Config): Snapshot {
  if (![2, 4].includes(config.size) || config.gamma < 0 || config.gamma > 1 || !Number.isFinite(config.gamma) || !(config.theta > 0)) throw new Error('Invalid configuration');
  const n = config.size ** 2;
  return { v: Array(n).fill(0), q: Array.from({ length: n }, () => [0, 0, 0, 0]),
    policy: Array.from({ length: n }, () => [0.25, 0.25, 0.25, 0.25]),
    cursor: 0, sweeps: 0, delta: 0, lastDelta: null, phase: 'evaluate',
    improvements: 0, changes: null, trace: null, history: [] };
}
export function greedy(q: number[]): number { return q.indexOf(Math.max(...q)); }
export function backup(config: Config, snapshot: Snapshot, state: number): Backup {
  const future = ACTIONS.map((_, a) => {
    const t = transition(config.size, state, a);
    return t.done ? 0 : snapshot.v[t.next];
  });
  const q = ACTIONS.map((_, a) => transition(config.size, state, a).reward + config.gamma * future[a]);
  const weights = config.mode === 'value' ? q.map((_, a) => Number(a === greedy(q))) : [...snapshot.policy[state]];
  return { state, old: snapshot.v[state], value: q.reduce((sum, v, a) => sum + v * weights[a], 0), q, future, weights };
}
/** One state backup, or one separate greedy policy-improvement phase. Never mutates input. */
export function advance(config: Config, current: Snapshot): Snapshot {
  if (current.phase === 'done') return current;
  const next = structuredClone(current);
  if (next.phase === 'improve') {
    let changes = 0;
    for (let s = 0; s < next.v.length; s++) {
      next.q[s] = backup(config, next, s).q;
      const a = greedy(next.q[s]);
      const improved = ACTIONS.map((_, i) => Number(i === a));
      if (s !== next.v.length - 1 && improved.some((p, i) => p !== next.policy[s][i])) changes++;
      next.policy[s] = improved;
    }
    next.changes = changes;
    next.improvements++;
    next.trace = null;
    next.phase = changes === 0 ? 'done' : 'evaluate';
    return next;
  }
  const trace = backup(config, next, next.cursor);
  next.trace = trace;
  next.v[trace.state] = trace.value;
  next.q[trace.state] = trace.q;
  next.delta = Math.max(next.delta, Math.abs(trace.old - trace.value));
  next.cursor++;
  if (next.cursor === next.v.length) {
    next.cursor = 0;
    next.sweeps++;
    next.lastDelta = next.delta;
    next.history.push(next.delta);
    if (next.delta < config.theta) {
      next.phase = config.mode === 'policy' ? 'improve' : 'done';
      // value_prediction recomputes Q from converged V; value_iteration returns its last sweep's Q.
      if (config.mode === 'evaluation') next.q = next.v.map((_, s) => backup(config, next, s).q);
    }
    next.delta = 0;
  }
  return next;
}
