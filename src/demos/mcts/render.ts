import { initialCounts, lastStep, snapshot } from './core';

export function initMCTS() {
  const get = <T extends Element = HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
  const next = get<HTMLButtonElement>('[data-next]');
  const back = get<HTMLButtonElement>('[data-back]');
  const play = get<HTMLButtonElement>('[data-play]');
  const outcome = get<HTMLSelectElement>('#outcome');
  let step = 0;
  let timer: ReturnType<typeof setInterval> | undefined;

  function stop() {
    clearInterval(timer);
    timer = undefined;
    play.textContent = 'Play';
  }

  function render() {
    const success = outcome.value === 'success';
    const result = success ? 'success (+1)' : 'failure (0)';
    const increment = success ? 1 : 0;
    const { counts, child } = snapshot(step, success);
    const stages = [
      ['Start with the given tree', 'The root value is 2/4 = 0.50. Compare its children before choosing an action.'],
      ['1 · Select node 3', 'The child values are V(2) = 0, V(3) = 1, and V(4) = 0.5. Greedy selection follows 1 → 3, the highest estimated success probability.'],
      ['2 · Expand an untried action', 'Replace the black dot below node 3 with a new child. Its counts start at 0/0; its value is undefined until a rollout has been backed up.'],
      ['3 · Simulate from the new child', `A rollout from the new state ends in ${result}. We supply this outcome for the example; counts have not changed yet. Deterministic transitions do not require all rollout action choices to be the same.`],
      ['4 · Back up to the new child', `Record one rollout: successes += ${increment}, visits += 1. The new child now has ${increment}/1. Continue up the selected path.`],
      ['5 · Back up to node 3', `Node 3 changes from 1/1 to ${1 + increment}/2. Its value is ${((1 + increment) / 2).toFixed(2)}. Back up the rollout result itself, not the new child’s average.`],
      ['6 · Back up to the root', `The root changes from 2/4 to ${2 + increment}/5. Its value is ${((2 + increment) / 5).toFixed(2)}. Nodes 2, 4, and 5 were outside this path and remain unchanged.`],
    ];
    get('[data-title]').textContent = stages[step][0];
    get('[data-explanation]').textContent = stages[step][1];
    get('[data-progress]').textContent = `Step ${step} / ${lastStep}`;
    const phase = step === 0 ? 0 : Math.min(step, 4);
    document.querySelectorAll<HTMLElement>('[data-phase]').forEach(el => {
      const active = Number(el.dataset.phase) === phase;
      el.classList.toggle('active', active);
      if (active) el.setAttribute('aria-current', 'step');
      else el.removeAttribute('aria-current');
    });
    counts.forEach(([wins, visits], i) => {
      const id = i + 1;
      const node = get(`[data-node="${id}"]`);
      const updated = visits !== initialCounts[i][1];
      node.querySelector('[data-count]')!.textContent = `${wins}/${visits}`;
      node.querySelector('[data-value]')!.textContent = `V = ${(wins / visits).toFixed(2)}`;
      node.classList.toggle('selected', step >= 1 && (id === 1 || id === 3));
      node.classList.toggle('updated', updated);
      const row = get(`[data-row="${id}"]`);
      row.querySelector('[data-current]')!.textContent = `${wins}/${visits}`;
      row.querySelector('[data-ratio]')!.textContent = (wins / visits).toFixed(2);
      row.querySelector('[data-change]')!.textContent = updated ? `+${increment} success, +1 visit` : 'Unchanged';
      row.classList.toggle('updated', updated);
    });
    get('[data-dot]').toggleAttribute('data-hidden', step >= 2);
    get('[data-new-node]').toggleAttribute('data-hidden', step < 2);
    get('[data-new-node]').classList.toggle('updated', step >= 4);
    if (child) {
      get('[data-new-count]').textContent = `${child[0]}/${child[1]}`;
      get('[data-new-value]').textContent = child[1] ? `V = ${child[0].toFixed(2)}` : 'V undefined';
    }
    get('[data-edge="select"]').classList.toggle('active', step >= 1);
    get('[data-edge="expand"]').classList.toggle('active', step >= 2);
    get('[data-summary]').textContent = step === lastStep
      ? `Final values (nodes 1–5): ${(counts.map(([w, n]) => `${w}/${n} = ${(w / n).toFixed(2)}`)).join('; ')}.`
      : 'Predict the final values, then step through the iteration to check your answer.';
    next.disabled = step === lastStep;
    back.disabled = step === 0;
    if (step === lastStep) stop();
  }

  next.addEventListener('click', () => { stop(); step = Math.min(lastStep, step + 1); render(); });
  back.addEventListener('click', () => { stop(); step = Math.max(0, step - 1); render(); });
  function reset() { stop(); step = 0; render(); }
  get('[data-reset]').addEventListener('click', reset);
  outcome.addEventListener('change', reset);
  play.addEventListener('click', () => {
    if (timer !== undefined) { stop(); return; }
    if (step === lastStep) step = 0;
    step++;
    render();
    if (step < lastStep) {
      play.textContent = 'Pause';
      timer = setInterval(() => { step++; render(); }, 1800);
    }
  });
  window.addEventListener('pagehide', stop);
  render();
}
