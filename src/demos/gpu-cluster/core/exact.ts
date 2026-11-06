/**
 * Exact references on the bucketed model.
 *
 * Because the head request is drawn fresh every step, a decision-time state
 * (n, c, b) is a count vector n plus an independent head. Every solver here
 * therefore works on the count-vector chain (286 states at K = 10, B = 3) and
 * expands to full states only at the end:
 *
 *   μ_π(n, c, b) = ν(n)·w_c/B,    h(n, c, b) = r(s, π(s)) − r(π) + Σ T(after, n′) H̄(n′).
 *
 * Small chains are solved directly (dense Gaussian elimination); large ones
 * fall back to iteration on the lazy chain ½(I + P), which has the same
 * stationary distribution and converges even when P is periodic.
 */

import { ACCEPT, PAY, REJECT, STEPS_PER_DAY, TIERS, tierMix, type ClusterConfig } from './config.ts';
import { afterstate, decodeState, reward, type Model, type ModelPolicy, type SparseRow } from './model.ts';

/** Count chains up to this size are solved directly. */
export const DENSE_LIMIT = 1500;

/** Solves A x = b in place (A is n×n row-major), Gaussian elimination with partial pivoting. */
export function solveDense(A: Float64Array, b: Float64Array, n: number): Float64Array {
  for (let col = 0; col < n; col += 1) {
    let piv = col;
    let best = Math.abs(A[col * n + col]!);
    for (let r = col + 1; r < n; r += 1) {
      const v = Math.abs(A[r * n + col]!);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-300) throw new Error('singular system');
    if (piv !== col) {
      for (let k = 0; k < n; k += 1) {
        const t = A[col * n + k]!; A[col * n + k] = A[piv * n + k]!; A[piv * n + k] = t;
      }
      const t = b[col]!; b[col] = b[piv]!; b[piv] = t;
    }
    const d = A[col * n + col]!;
    for (let r = col + 1; r < n; r += 1) {
      const f = A[r * n + col]! / d;
      if (f === 0) continue;
      for (let k = col; k < n; k += 1) A[r * n + k]! -= f * A[col * n + k]!;
      b[r]! -= f * b[col]!;
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r -= 1) {
    let s = b[r]!;
    for (let k = r + 1; k < n; k += 1) s -= A[r * n + k]! * x[k]!;
    x[r] = s / A[r * n + r]!;
  }
  return x;
}

/** Count-level chain under a policy: P(n → n′) and expected reward r̄(n). */
export interface CountChain {
  rows: SparseRow[];
  rbar: Float64Array;
}

/** out[n] = Σ_{n′} T(n → n′) U(n′) for every afterstate n. */
export function applyT(model: Model, U: Float64Array, out = new Float64Array(model.nCounts)): Float64Array {
  for (let n = 0; n < model.nCounts; n += 1) {
    const { next, prob } = model.trans[n]!;
    let s = 0;
    for (let k = 0; k < next.length; k += 1) s += prob[k]! * U[next[k]!]!;
    out[n] = s;
  }
  return out;
}

export function countChain(model: Model, policy: ModelPolicy, mix = model.mix): CountChain {
  const N = model.nCounts;
  const scratch = new Float64Array(N);
  const touched: number[] = [];
  const rows: SparseRow[] = [];
  const rbar = new Float64Array(N);
  for (let ci = 0; ci < N; ci += 1) {
    for (let c = 0; c < TIERS; c += 1) {
      for (let b = 0; b < model.B; b += 1) {
        const s = (ci * TIERS + c) * model.B + b;
        const weight = mix[c]! / model.B;
        if (weight === 0) continue;
        const a = policy[s]!;
        rbar[ci]! += weight * reward(model, s, a);
        const { next, prob } = model.trans[afterstate(model, s, a)]!;
        for (let k = 0; k < next.length; k += 1) {
          const j = next[k]!;
          if (scratch[j] === 0) touched.push(j);
          scratch[j]! += weight * prob[k]!;
        }
      }
    }
    touched.sort((x, y) => x - y);
    rows.push({ next: Int32Array.from(touched), prob: Float64Array.from(touched, j => scratch[j]!) });
    for (const j of touched) scratch[j] = 0;
    touched.length = 0;
  }
  return { rows, rbar };
}

function rowDot(row: SparseRow, U: Float64Array): number {
  let s = 0;
  for (let k = 0; k < row.next.length; k += 1) s += row.prob[k]! * U[row.next[k]!]!;
  return s;
}

/** ν P for a count-level distribution ν. */
export function pushForward(chain: CountChain, nu: Float64Array): Float64Array {
  const out = new Float64Array(nu.length);
  chain.rows.forEach((row, i) => {
    const m = nu[i]!;
    if (m === 0) return;
    for (let k = 0; k < row.next.length; k += 1) out[row.next[k]!]! += m * row.prob[k]!;
  });
  return out;
}

/** Stationary distribution ν of the count chain. */
export function stationaryCounts(chain: CountChain, tol = 1e-14): Float64Array {
  const N = chain.rows.length;
  if (N <= DENSE_LIMIT) {
    // (I − P)ᵀ ν = 0 with the last equation replaced by Σν = 1.
    const A = new Float64Array(N * N);
    chain.rows.forEach((row, i) => {
      A[i * N + i]! += 1;
      for (let k = 0; k < row.next.length; k += 1) A[row.next[k]! * N + i]! -= row.prob[k]!;
    });
    for (let i = 0; i < N; i += 1) A[(N - 1) * N + i] = 1;
    const b = new Float64Array(N);
    b[N - 1] = 1;
    const nu = solveDense(A, b, N);
    for (let i = 0; i < N; i += 1) if (nu[i]! < 0 && nu[i]! > -1e-15) nu[i] = 0;
    return nu;
  }
  let nu: Float64Array = new Float64Array(N).fill(1 / N);
  for (let it = 0; it < 200000; it += 1) {
    const next = pushForward(chain, nu);
    let diff = 0;
    for (let i = 0; i < N; i += 1) {
      diff = Math.max(diff, Math.abs(next[i]! - nu[i]!));
      next[i] = 0.5 * (next[i]! + nu[i]!);
    }
    nu = next;
    if (diff < tol) break;
  }
  return nu;
}

/** Expands a count-level quantity to full states: μ(n, c, b) = ν(n)·w_c/B. */
export function expandDistribution(model: Model, nu: Float64Array, mix = model.mix): Float64Array {
  const mu = new Float64Array(model.nStates);
  for (let s = 0; s < model.nStates; s += 1) {
    const { ci, c } = decodeState(model, s);
    mu[s] = (nu[ci]! * mix[c]!) / model.B;
  }
  return mu;
}

/** One step of a full-state distribution under the policy: μ ↦ μ P_π. */
export function stepFull(model: Model, policy: ModelPolicy, mu: Float64Array, mix = model.mix): Float64Array {
  const nu = new Float64Array(model.nCounts);
  for (let s = 0; s < model.nStates; s += 1) {
    const m = mu[s]!;
    if (m === 0) continue;
    const { next, prob } = model.trans[afterstate(model, s, policy[s]!)]!;
    for (let k = 0; k < next.length; k += 1) nu[next[k]!]! += m * prob[k]!;
  }
  return expandDistribution(model, nu, mix);
}

export interface Evaluation {
  nu: Float64Array;
  mu: Float64Array;
  /** Long-run average reward r(π) = Σ μ(s) E[r | s, π]. */
  avg: number;
  chain: CountChain;
}

export function evaluatePolicy(model: Model, policy: ModelPolicy): Evaluation {
  const chain = countChain(model, policy);
  const nu = stationaryCounts(chain);
  let avg = 0;
  for (let i = 0; i < nu.length; i += 1) avg += nu[i]! * chain.rbar[i]!;
  return { nu, mu: expandDistribution(model, nu), avg, chain };
}

export interface DiscountedValues {
  /** Count-level V̄(n) = Σ_{c,b} w_c/B v(n, c, b). */
  vbar: Float64Array;
  v: Float64Array;
  /** J_γ(π) = Σ μ_π(s) v^γ_π(s). */
  J: number;
}

export function discountedValues(model: Model, policy: ModelPolicy, gamma: number, ev = evaluatePolicy(model, policy)): DiscountedValues {
  const { chain } = ev;
  const N = model.nCounts;
  let vbar: Float64Array;
  if (N <= DENSE_LIMIT) {
    const A = new Float64Array(N * N);
    chain.rows.forEach((row, i) => {
      A[i * N + i]! += 1;
      for (let k = 0; k < row.next.length; k += 1) A[i * N + row.next[k]!]! -= gamma * row.prob[k]!;
    });
    vbar = solveDense(A, Float64Array.from(chain.rbar), N);
  } else {
    vbar = new Float64Array(N);
    for (let it = 0; it < 1e6; it += 1) {
      let diff = 0;
      for (let i = 0; i < N; i += 1) {
        const v = chain.rbar[i]! + gamma * rowDot(chain.rows[i]!, vbar);
        diff = Math.max(diff, Math.abs(v - vbar[i]!));
        vbar[i] = v;
      }
      if (diff < 1e-12) break;
    }
  }
  const TV = applyT(model, vbar);
  const v = new Float64Array(model.nStates);
  let J = 0;
  for (let s = 0; s < model.nStates; s += 1) {
    const a = policy[s]!;
    v[s] = reward(model, s, a) + gamma * TV[afterstate(model, s, a)]!;
    J += ev.mu[s]! * v[s]!;
  }
  return { vbar, v, J };
}

export interface DifferentialValues {
  /** Count-level H̄(n), normalized so Σ ν H̄ = 0. */
  hbar: Float64Array;
  /** h_π(s), normalized so Σ μ_π h_π = 0. */
  h: Float64Array;
  /** Differential action values q_π(s, a). */
  q: (s: number, a: number) => number;
}

export function differentialValues(model: Model, policy: ModelPolicy, ev = evaluatePolicy(model, policy)): DifferentialValues {
  const { chain, nu, avg } = ev;
  const N = model.nCounts;
  let hbar: Float64Array;
  if (N <= DENSE_LIMIT) {
    // (I − P + 1νᵀ) H̄ = r̄ − r(π): nonsingular for a unichain, and forces ν·H̄ = 0.
    const A = new Float64Array(N * N);
    for (let i = 0; i < N; i += 1) {
      A[i * N + i]! += 1;
      for (let j = 0; j < N; j += 1) A[i * N + j]! += nu[j]!;
      const row = chain.rows[i]!;
      for (let k = 0; k < row.next.length; k += 1) A[i * N + row.next[k]!]! -= row.prob[k]!;
    }
    hbar = solveDense(A, Float64Array.from(chain.rbar, r => r - avg), N);
  } else {
    hbar = new Float64Array(N);
    for (let it = 0; it < 1e6; it += 1) {
      const next = new Float64Array(N);
      let diff = 0;
      let center = 0;
      for (let i = 0; i < N; i += 1) {
        next[i] = 0.5 * (hbar[i]! + chain.rbar[i]! - avg + rowDot(chain.rows[i]!, hbar));
        center += nu[i]! * next[i]!;
      }
      for (let i = 0; i < N; i += 1) {
        next[i]! -= center;
        diff = Math.max(diff, Math.abs(next[i]! - hbar[i]!));
      }
      hbar = next;
      if (diff < 1e-12) break;
    }
  }
  const TH = applyT(model, hbar);
  const q = (s: number, a: number) => reward(model, s, a) - avg + TH[afterstate(model, s, a)]!;
  const h = Float64Array.from({ length: model.nStates }, (_, s) => q(s, policy[s]!));
  return { hbar, h, q };
}

export interface RviResult {
  /** Optimal average reward r*. */
  rStar: number;
  policy: ModelPolicy;
  /** Count-level bias U(n) of the optimal policy, normalized so ν*·U = 0. */
  U: Float64Array;
  /** Optimal differential action values. */
  q: (s: number, a: number) => number;
  iterations: number;
  /** span(LU − U) with the untransformed operator at the end. */
  residualSpan: number;
}

export interface RviOptions {
  eta?: number;
  tol?: number;
  maxIter?: number;
  onProgress?: (iteration: number, span: number) => void;
}

/** Pre-head Bellman operator: (LU)(n) = Σ_{c,b} w_c/B max_a [r + TU(after)]. */
function bellman(model: Model, TU: Float64Array, ci: number, mix: readonly number[]): number {
  const busy = model.K - model.free[ci]!;
  const rej = -model.kappa * busy + TU[ci]!;
  let total = 0;
  for (let c = 0; c < TIERS; c += 1) {
    for (let b = 0; b < model.B; b += 1) {
      let best = rej;
      if (model.free[ci]! > 0) {
        const acc = PAY[c]! - model.kappa * (busy + 1) + TU[model.inc[ci * model.B + b]!]!;
        if (acc > best) best = acc;
      }
      total += (mix[c]! / model.B) * best;
    }
  }
  return total;
}

/**
 * Relative value iteration with the aperiodicity transform
 * p̃ = (1 − η)I + ηp, r̃ = ηr. The transformed problem has the same optimal
 * policy and bias; its gain is η·r*.
 */
export function relativeValueIteration(model: Model, options: RviOptions = {}): RviResult {
  const eta = options.eta ?? 0.9;
  const tol = options.tol ?? 1e-10;
  const maxIter = options.maxIter ?? 500000;
  const N = model.nCounts;
  let U = new Float64Array(N);
  const TU = new Float64Array(N);
  let gain = 0;
  let iterations = 0;
  for (; iterations < maxIter; iterations += 1) {
    applyT(model, U, TU);
    const next = new Float64Array(N);
    let lo = Infinity;
    let hi = -Infinity;
    for (let n = 0; n < N; n += 1) {
      next[n] = (1 - eta) * U[n]! + eta * bellman(model, TU, n, model.mix);
      const d = next[n]! - U[n]!;
      if (d < lo) lo = d;
      if (d > hi) hi = d;
    }
    const ref = next[model.empty]!;
    for (let n = 0; n < N; n += 1) next[n]! -= ref;
    U = next;
    gain = (lo + hi) / 2;
    if (options.onProgress && iterations % 200 === 0) options.onProgress(iterations, hi - lo);
    if (hi - lo < tol) break;
  }
  const rStar = gain / eta;
  applyT(model, U, TU);
  const policy = new Uint8Array(model.nStates);
  for (let s = 0; s < model.nStates; s += 1) {
    const { ci, c, b } = decodeState(model, s);
    if (model.free[ci]! === 0) continue;
    const gainFromAccepting = PAY[c]! - model.kappa + TU[model.inc[ci * model.B + b]!]! - TU[ci]!;
    policy[s] = gainFromAccepting > 1e-9 ? ACCEPT : REJECT;
  }
  // Re-center the bias on the optimal policy's own distribution.
  const ev = evaluatePolicy(model, policy);
  let center = 0;
  for (let n = 0; n < N; n += 1) center += ev.nu[n]! * U[n]!;
  for (let n = 0; n < N; n += 1) U[n]! -= center;
  applyT(model, U, TU);
  let lo = Infinity;
  let hi = -Infinity;
  for (let n = 0; n < N; n += 1) {
    const d = bellman(model, TU, n, model.mix) - U[n]!;
    lo = Math.min(lo, d);
    hi = Math.max(hi, d);
  }
  const TUfinal = Float64Array.from(TU);
  const q = (s: number, a: number) => reward(model, s, a) - rStar + TUfinal[afterstate(model, s, a)]!;
  return { rStar, policy, U, q, iterations, residualSpan: hi - lo };
}

export interface ShiftSolution {
  /** U[k][n]: expected revenue of the last k steps of the shift from count n, before the head is drawn. */
  U: Float64Array[];
  /** decision[k][s]: the action taken with k steps to go. */
  decision: Uint8Array[];
  /** Expected revenue of a whole shift from an empty cluster. */
  value: number;
}

/**
 * Finite-horizon DP over one shift of H steps starting at τ = 0. The head
 * at clock τ is drawn from w(φ(τ)), so the diurnal mix is handled exactly.
 * Without `fixed`, it returns the optimal shift policy; with it, the value of
 * that fixed policy (k = steps to go).
 */
export function solveShift(model: Model, cfg: ClusterConfig, fixed?: (k: number, s: number) => number): ShiftSolution {
  const N = model.nCounts;
  const H = cfg.H;
  const U: Float64Array[] = [new Float64Array(N)];
  const decision: Uint8Array[] = [new Uint8Array(model.nStates)];
  const TU = new Float64Array(N);
  for (let k = 1; k <= H; k += 1) {
    const tau = H - k;
    const mix = tierMix(cfg, (tau % STEPS_PER_DAY) / STEPS_PER_DAY);
    applyT(model, U[k - 1]!, TU);
    const Uk = new Float64Array(N);
    const dk = new Uint8Array(model.nStates);
    for (let ci = 0; ci < N; ci += 1) {
      const busy = model.K - model.free[ci]!;
      const rej = -model.kappa * busy + TU[ci]!;
      for (let c = 0; c < TIERS; c += 1) {
        for (let b = 0; b < model.B; b += 1) {
          const s = (ci * TIERS + c) * model.B + b;
          const acc = model.free[ci]! > 0 ? PAY[c]! - model.kappa * (busy + 1) + TU[model.inc[ci * model.B + b]!]! : -Infinity;
          const a = model.free[ci]! === 0 ? REJECT : fixed ? fixed(k, s) : acc > rej + 1e-9 ? ACCEPT : REJECT;
          dk[s] = a;
          Uk[ci]! += (mix[c]! / model.B) * (a === ACCEPT ? acc : rej);
        }
      }
    }
    U.push(Uk);
    decision.push(dk);
  }
  return { U, decision, value: U[H]![model.empty]! };
}
