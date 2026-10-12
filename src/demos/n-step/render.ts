import { drawLineChart, type Series } from '../shared/chart';
import { toSeed } from '../shared/rng';
import { alphaSweep, nLabel, runStream, trueValues, type StreamConfig, type StreamResult } from './core';

const N_OPTIONS = [1, 2, 4, 8, 16, 32, Infinity];
const ALPHAS = Array.from({ length: 21 }, (_, i) => i / 20);
const SWEEP_EPISODES = 10;
const COLORS = { a: '#60a5fa', b: '#fb923c', truth: '#e2e8f0', other: '#475569' };

type Key = 'a' | 'b';

export function initNStepComparison() {
  const root = document.querySelector<HTMLElement>('[data-nstep]');
  if (!root) return;
  const $ = <T extends Element>(selector: string) => root.querySelector<T>(selector)!;

  const streams: Record<Key, StreamConfig> = {
    a: { n: 1, alpha: 0.4 },
    b: { n: 4, alpha: 0.4 },
  };
  const episodesInput = $<HTMLInputElement>('[data-episodes]');
  const episodesLabel = $<HTMLOutputElement>('[data-episodes-value]');
  const runsInput = $<HTMLSelectElement>('[data-runs]');
  const seedInput = $<HTMLInputElement>('[data-seed]');
  const status = $<HTMLElement>('[data-status]');
  const curveCanvas = $<HTMLCanvasElement>('[data-curve]');
  const valueCanvas = $<HTMLCanvasElement>('[data-values]');
  const sweepCanvas = $<HTMLCanvasElement>('[data-sweep]');

  const format = (value: number) => value.toFixed(3);
  let sweepKey = '';
  let sweep = new Map<number, number[]>();

  for (const key of ['a', 'b'] as const) {
    const panel = $<HTMLElement>(`[data-stream="${key}"]`);
    const group = panel.querySelector<HTMLElement>('[data-n-group]')!;
    for (const n of N_OPTIONS) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = Number.isFinite(n) ? String(n) : '∞';
      button.title = nLabel(n);
      button.dataset.n = String(n);
      button.addEventListener('click', () => { streams[key].n = n; render(); });
      group.append(button);
    }
    const alpha = panel.querySelector<HTMLInputElement>('[data-alpha]')!;
    alpha.value = String(streams[key].alpha);
    alpha.addEventListener('input', () => { streams[key].alpha = Number(alpha.value); render(); });
  }

  function sweepFor(runs: number, seed: number): Map<number, number[]> {
    // The sweep is the slow part and does not depend on either stream, so it
    // is recomputed only when the shared experience changes.
    const key = `${Math.min(runs, 50)}:${seed}`;
    if (key !== sweepKey) {
      sweepKey = key;
      sweep = new Map(N_OPTIONS.map(n => [n, alphaSweep(n, ALPHAS, SWEEP_EPISODES, Math.min(runs, 50), seed)]));
    }
    return sweep;
  }

  function describe(key: Key, result: StreamResult) {
    const panel = $<HTMLElement>(`[data-stream="${key}"]`);
    const { n, alpha } = streams[key];
    for (const button of panel.querySelectorAll<HTMLButtonElement>('[data-n]')) {
      button.setAttribute('aria-pressed', String(Number(button.dataset.n) === n));
    }
    panel.querySelector<HTMLOutputElement>('[data-alpha-value]')!.value = alpha.toFixed(2);
    panel.querySelector<HTMLElement>('[data-label]')!.textContent = nLabel(n);
    panel.querySelector<HTMLElement>('[data-average]')!.textContent = format(result.averageError);
    panel.querySelector<HTMLElement>('[data-final]')!.textContent = format(result.curve.at(-1)!);
    panel.querySelector<HTMLElement>('[data-reach]')!.textContent = Number.isFinite(n)
      ? `Each update uses ${n} real reward${n === 1 ? '' : 's'}, then bootstraps from V(S${n === 1 ? 't+1' : `t+${n}`}).`
      : 'Each update waits for the episode to end and uses the full return. No bootstrapping.';
  }

  function render() {
    const episodes = Number(episodesInput.value);
    const runs = Number(runsInput.value);
    const seed = toSeed(seedInput.value);
    episodesLabel.value = String(episodes);

    const results = {
      a: runStream(streams.a, episodes, runs, seed),
      b: runStream(streams.b, episodes, runs, seed),
    };
    describe('a', results.a);
    describe('b', results.b);

    const gap = results.a.averageError - results.b.averageError;
    const same = streams.a.n === streams.b.n && streams.a.alpha === streams.b.alpha;
    status.textContent = same
      ? 'Both streams have identical settings, so their curves overlap exactly.'
      : Math.abs(gap) < 0.005
        ? `Both streams average about ${format(results.a.averageError)} RMS error over ${episodes} episodes.`
        : `Stream ${gap > 0 ? 'B' : 'A'} averages ${format(Math.abs(gap))} lower RMS error over ${episodes} episodes, from the same ${runs === 1 ? 'episodes' : `${runs} runs of episodes`}.`;

    drawLineChart(curveCanvas, [
      { label: 'A', color: COLORS.a, points: results.a.curve, xEnd: episodes },
      { label: 'B', color: COLORS.b, points: results.b.curve, xEnd: episodes },
    ], { yMin: 0, yMax: 0.6, xMax: episodes, xLabel: 'Episodes', formatY: v => v.toFixed(2) });

    drawLineChart(valueCanvas, [
      { label: 'True', color: COLORS.truth, points: trueValues },
      { label: 'A', color: COLORS.a, points: results.a.values },
      { label: 'B', color: COLORS.b, points: results.b.values },
    ], { yMin: -1, yMax: 1, formatY: v => v.toFixed(1) });

    const curves = sweepFor(runs, seed);
    const selected = new Set([streams.a.n, streams.b.n]);
    const series: Series[] = N_OPTIONS.filter(n => !selected.has(n))
      .map(n => ({ label: nLabel(n), color: COLORS.other, points: curves.get(n)!, xEnd: 1 }));
    // Draw B before A so that when both pick the same n, A's colour is on top.
    if (streams.b.n !== streams.a.n) series.push({ label: 'B', color: COLORS.b, points: curves.get(streams.b.n)!, xEnd: 1 });
    series.push({ label: 'A', color: COLORS.a, points: curves.get(streams.a.n)!, xEnd: 1 });
    drawLineChart(sweepCanvas, series, { yMin: 0.2, yMax: 0.7, xMax: 1, xLabel: 'Step size α', formatY: v => v.toFixed(2) });
    const best = $<HTMLElement>('[data-sweep-best]');
    best.textContent = 'Best per n — ' + N_OPTIONS.map(n => {
      const errors = curves.get(n)!;
      const best = errors.indexOf(Math.min(...errors));
      return `${Number.isFinite(n) ? n : '∞'}: ${format(errors[best])} at α=${ALPHAS[best].toFixed(2)}`;
    }).join(' · ');
    best.title = best.textContent;
  }

  episodesInput.addEventListener('input', render);
  runsInput.addEventListener('change', render);
  seedInput.addEventListener('change', render);
  $<HTMLButtonElement>('[data-swap]').addEventListener('click', () => {
    [streams.a, streams.b] = [streams.b, streams.a];
    $<HTMLInputElement>('[data-stream="a"] [data-alpha]').value = String(streams.a.alpha);
    $<HTMLInputElement>('[data-stream="b"] [data-alpha]').value = String(streams.b.alpha);
    render();
  });
  let resizeFrame = 0;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(render);
  });
  render();
}
