/**
 * DOM layer for the epsilon-greedy demo.
 *
 * All simulation lives in ./core. This file only reads state and writes pixels,
 * which is what keeps the core runnable headlessly for the comparison sweep.
 */

import { BanditRun, compareEpsilons, smooth, type StepRecord } from './core';
import { downsample, drawLineChart, clearCanvas, type Series } from '../shared/chart';
import { randomSeed, toSeed } from '../shared/rng';

const COLORS = {
  reward: '#10b981',
  optimal: '#818cf8',
  grid: '#334155',
  text: '#94a3b8',
  sweep: ['#f87171', '#fbbf24', '#34d399', '#60a5fa', '#c084fc'],
};

const AUTOPLAY_MS = 200;
const COMPARE_EPSILONS = [0, 0.01, 0.1, 0.3];
const COMPARE_STEPS = 2000;
const COMPARE_RUNS = 150;

function must<T extends Element>(root: ParentNode, selector: string): T {
  const el = root.querySelector<T>(selector);
  if (!el) throw new Error(`epsilon-greedy demo: missing element ${selector}`);
  return el;
}

/** Per-arm DOM handles, built once and mutated in place. */
interface ArmView {
  root: HTMLElement;
  estimate: HTMLElement;
  pulls: HTMLElement;
  wins: HTMLElement;
  truthBar: HTMLElement;
  truthValue: HTMLElement;
  crown: HTMLElement;
}

function buildArm(index: number): ArmView {
  const root = document.createElement('div');
  root.className = 'arm';
  root.id = `arm-${index}`;

  const crown = document.createElement('span');
  crown.className = 'arm__crown';
  crown.textContent = 'Best known';
  crown.hidden = true;

  const title = document.createElement('h3');
  title.className = 'arm__title';
  title.textContent = `Arm ${index + 1}`;

  const estimateBox = document.createElement('div');
  estimateBox.className = 'arm__estimate';
  const estimateLabel = document.createElement('span');
  estimateLabel.className = 'arm__label';
  estimateLabel.textContent = 'Estimated value Q';
  const estimate = document.createElement('span');
  estimate.className = 'arm__value';
  estimate.textContent = '—';
  estimateBox.append(estimateLabel, estimate);

  const counts = document.createElement('div');
  counts.className = 'arm__counts';
  const pulls = document.createElement('span');
  const wins = document.createElement('span');
  counts.append(pulls, wins);

  const truth = document.createElement('div');
  truth.className = 'arm__truth';
  const truthLabel = document.createElement('span');
  truthLabel.className = 'arm__label';
  truthLabel.textContent = 'True win rate';
  const track = document.createElement('div');
  track.className = 'arm__track';
  const truthBar = document.createElement('div');
  truthBar.className = 'arm__bar';
  track.append(truthBar);
  const truthValue = document.createElement('span');
  truthValue.className = 'arm__truth-value';
  truth.append(truthLabel, track, truthValue);

  root.append(crown, title, estimateBox, counts, truth);
  return { root, estimate, pulls, wins, truthBar, truthValue, crown };
}

export function initEpsilonGreedyDemo(root: ParentNode = document): void {
  const armsContainer = must<HTMLElement>(root, '[data-arms]');
  const epsilonSlider = must<HTMLInputElement>(root, '[data-epsilon]');
  const epsilonValue = must<HTMLElement>(root, '[data-epsilon-value]');
  const seedInput = must<HTMLInputElement>(root, '[data-seed]');
  const btnStep = must<HTMLButtonElement>(root, '[data-step]');
  const btnAuto = must<HTMLButtonElement>(root, '[data-auto]');
  const btnReset = must<HTMLButtonElement>(root, '[data-reset]');
  const btnNewEnv = must<HTMLButtonElement>(root, '[data-new-env]');
  const toggleTruth = must<HTMLInputElement>(root, '[data-truth]');
  const log = must<HTMLElement>(root, '[data-log]');
  const statPulls = must<HTMLElement>(root, '[data-stat-pulls]');
  const statReward = must<HTMLElement>(root, '[data-stat-reward]');
  const statOptimal = must<HTMLElement>(root, '[data-stat-optimal]');
  const statRegret = must<HTMLElement>(root, '[data-stat-regret]');
  const chart = must<HTMLCanvasElement>(root, '[data-chart]');
  const chartHint = must<HTMLElement>(root, '[data-chart-hint]');
  const compareChart = must<HTMLCanvasElement>(root, '[data-compare-chart]');
  const btnCompare = must<HTMLButtonElement>(root, '[data-compare]');
  const compareStatus = must<HTMLElement>(root, '[data-compare-status]');
  const compareLegend = must<HTMLElement>(root, '[data-compare-legend]');

  const armViews: ArmView[] = [];
  let run: BanditRun;
  let autoTimer: number | null = null;

  function currentEpsilon(): number {
    return Number(epsilonSlider.value);
  }

  function renderStats(): void {
    statPulls.textContent = String(run.steps);
    statReward.textContent = run.steps === 0 ? '—' : run.avgReward.toFixed(3);
    statOptimal.textContent = run.steps === 0 ? '—' : `${(run.optimalRate * 100).toFixed(1)}%`;
    statRegret.textContent = run.cumulativeRegret.toFixed(2);
  }

  function renderArms(): void {
    const showTruth = toggleTruth.checked;
    const greedy = run.steps === 0 ? [] : run.greedyArms();

    run.arms.forEach((arm, i) => {
      const view = armViews[i]!;
      view.estimate.textContent = arm.pulls === 0 ? '—' : arm.estimate.toFixed(3);
      view.estimate.classList.toggle('arm__value--unknown', arm.pulls === 0);
      view.pulls.textContent = `Pulls ${arm.pulls}`;
      view.wins.textContent = `Wins ${arm.wins}`;
      view.crown.hidden = !(greedy.length > 0 && greedy.includes(i));
      view.truthBar.style.width = `${(arm.trueProb * 100).toFixed(1)}%`;
      view.truthValue.textContent = `${(arm.trueProb * 100).toFixed(1)}%`;
      view.root.classList.toggle('arm--revealed', showTruth);
      // Truth values are hidden from assistive tech too while concealed —
      // otherwise a screen-reader user is told the answer the demo is hiding.
      view.truthBar.parentElement!.parentElement!.setAttribute('aria-hidden', String(!showTruth));
    });
  }

  function renderChart(): void {
    const empty = run.history.length < 2;
    chartHint.hidden = !empty;
    if (empty) {
      clearCanvas(chart);
      return;
    }
    const rewards = downsample(smooth(run.history.map((r) => r.reward), 25), 400);
    const optimal = downsample(smooth(run.history.map((r) => (r.optimal ? 1 : 0)), 25), 400);

    drawLineChart(
      chart,
      [
        { label: 'Average reward', color: COLORS.reward, points: rewards, xEnd: run.steps, fill: true },
        { label: '% optimal action', color: COLORS.optimal, points: optimal, xEnd: run.steps },
      ],
      {
        yMin: 0,
        yMax: 1,
        xMax: Math.max(100, run.steps),
        xLabel: 'Steps',
        gridColor: COLORS.grid,
        textColor: COLORS.text,
      },
    );
  }

  function showLogPlaceholder(): void {
    const empty = document.createElement('li');
    empty.className = 'log__empty';
    empty.textContent = 'Step once to see why each arm was chosen.';
    log.replaceChildren(empty);
  }

  function appendLog(record: StepRecord): void {
    log.querySelector('.log__empty')?.remove();

    const entry = document.createElement('li');
    entry.className = `log__entry log__entry--${record.explored ? 'explore' : 'exploit'}`;

    const head = document.createElement('div');
    head.className = 'log__head';
    const kind = document.createElement('span');
    kind.className = 'log__kind';
    kind.textContent = record.explored ? 'EXPLORE' : 'EXPLOIT';
    const arm = document.createElement('span');
    arm.className = 'log__arm';
    arm.textContent = `Arm ${record.arm + 1} → ${record.reward === 1 ? 'win' : 'loss'}`;
    head.append(kind, arm);

    const why = document.createElement('p');
    why.className = 'log__why';
    why.textContent = record.explored
      ? `draw ${record.draw.toFixed(3)} < ε ${record.epsilon.toFixed(2)} — picked at random`
      : `draw ${record.draw.toFixed(3)} ≥ ε ${record.epsilon.toFixed(2)} — picked highest Q`;

    entry.append(head, why);
    log.prepend(entry);

    while (log.children.length > 40) log.lastElementChild?.remove();
  }

  function stopAuto(): void {
    if (autoTimer !== null) {
      window.clearInterval(autoTimer);
      autoTimer = null;
    }
    btnAuto.textContent = 'Auto-play';
    btnAuto.setAttribute('aria-pressed', 'false');
  }

  function toggleAuto(): void {
    if (autoTimer !== null) {
      stopAuto();
      return;
    }
    btnAuto.textContent = 'Pause';
    btnAuto.setAttribute('aria-pressed', 'true');
    autoTimer = window.setInterval(takeStep, AUTOPLAY_MS);
  }

  function flashArm(record: StepRecord): void {
    const view = armViews[record.arm]!;
    const cls = record.explored ? 'arm--explore' : 'arm--exploit';
    const outcome = record.reward === 1 ? 'arm--win' : 'arm--loss';
    view.root.classList.add(cls, outcome);
    window.setTimeout(() => view.root.classList.remove(cls, outcome), 450);
  }

  function takeStep(): void {
    run.epsilon = currentEpsilon();
    const record = run.step();
    renderArms();
    renderStats();
    renderChart();
    appendLog(record);
    flashArm(record);
  }

  /** Rebuilds the run. `seed` unchanged means the same arms come back. */
  function reset(seed: number): void {
    stopAuto();
    run = new BanditRun({ epsilon: currentEpsilon(), seed });
    seedInput.value = String(seed);

    if (armViews.length !== run.numArms) {
      armsContainer.replaceChildren();
      armViews.length = 0;
      for (let i = 0; i < run.numArms; i += 1) {
        const view = buildArm(i);
        armViews.push(view);
        armsContainer.append(view.root);
      }
    }

    showLogPlaceholder();
    renderArms();
    renderStats();
    renderChart();
  }

  function runComparison(): void {
    btnCompare.disabled = true;
    compareStatus.textContent = `Running ${COMPARE_RUNS} runs × ${COMPARE_STEPS} steps for each ε…`;

    // Yield first so the status text actually paints before the main thread
    // is tied up by the sweep.
    window.setTimeout(() => {
      const seed = toSeed(seedInput.value);
      const results = compareEpsilons(COMPARE_EPSILONS, {
        steps: COMPARE_STEPS,
        runs: COMPARE_RUNS,
        seed,
      });

      const series: Series[] = results.map((result, i) => ({
        label: `ε = ${result.epsilon}`,
        color: COLORS.sweep[i % COLORS.sweep.length]!,
        points: downsample(smooth(result.avgRewardCurve, 20), 400),
        xEnd: COMPARE_STEPS,
      }));

      drawLineChart(compareChart, series, {
        // Zoomed to the region the curves actually occupy — at 0-1 the
        // crossover this panel exists to show is a few pixels tall. The axis
        // labels state the floor, so the truncation is visible rather than hidden.
        yMin: 0.4,
        yMax: 1,
        yTicks: 6,
        xMax: COMPARE_STEPS,
        xLabel: `Steps (averaged over ${COMPARE_RUNS} runs)`,
        gridColor: COLORS.grid,
        textColor: COLORS.text,
      });

      compareLegend.replaceChildren();
      results.forEach((result, i) => {
        const item = document.createElement('li');
        item.className = 'legend__item';

        const swatch = document.createElement('span');
        swatch.className = 'legend__swatch';
        swatch.style.background = COLORS.sweep[i % COLORS.sweep.length]!;

        const label = document.createElement('span');
        label.textContent = `ε = ${result.epsilon.toFixed(2)}`;

        const detail = document.createElement('span');
        detail.className = 'legend__detail';
        detail.textContent =
          `${(result.optimalRate * 100).toFixed(1)}% optimal · regret ${result.cumulativeRegret.toFixed(1)}`;

        item.append(swatch, label, detail);
        compareLegend.append(item);
      });

      compareStatus.textContent = `Seed ${seed}. Every ε faced the same ${COMPARE_RUNS} environments.`;
      btnCompare.disabled = false;
    }, 16);
  }

  epsilonSlider.addEventListener('input', () => {
    epsilonValue.textContent = currentEpsilon().toFixed(2);
    if (run) run.epsilon = currentEpsilon();
  });

  btnStep.addEventListener('click', () => {
    stopAuto();
    takeStep();
  });
  btnAuto.addEventListener('click', toggleAuto);
  btnReset.addEventListener('click', () => reset(run.seed));
  btnNewEnv.addEventListener('click', () => reset(randomSeed()));
  seedInput.addEventListener('change', () => reset(toSeed(seedInput.value)));
  toggleTruth.addEventListener('change', renderArms);
  btnCompare.addEventListener('click', runComparison);

  window.addEventListener('resize', () => {
    renderChart();
  });

  epsilonValue.textContent = currentEpsilon().toFixed(2);
  reset(toSeed(seedInput.value || String(randomSeed())));
}
