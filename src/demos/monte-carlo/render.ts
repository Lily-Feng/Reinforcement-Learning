import { compareVisits, episodeReturns, episodes } from './core';

export function initReturnExplorer() {
  const root = document.querySelector<HTMLElement>('[data-mc-explorer]');
  if (!root) return;
  const gammaInput = root.querySelector<HTMLInputElement>('[data-gamma]')!;
  const gammaLabel = root.querySelector<HTMLOutputElement>('[data-gamma-value]')!;
  const next = root.querySelector<HTMLButtonElement>('[data-next]')!;
  const reset = root.querySelector<HTMLButtonElement>('[data-reset]')!;
  const status = root.querySelector<HTMLElement>('[data-status]')!;
  const path = root.querySelector<HTMLElement>('[data-path]')!;
  const rows = root.querySelector<HTMLTableSectionElement>('[data-visits]')!;
  const estimates = root.querySelector<HTMLElement>('[data-estimates]')!;
  let count = 0;
  const format = (value: number) => String(Number(value.toFixed(3)));

  function render() {
    const gamma = Number(gammaInput.value);
    gammaLabel.value = gamma.toFixed(2);
    next.disabled = count === episodes.length;
    next.textContent = count === episodes.length ? 'All 3 episodes included' : `Include episode ${count + 1}`;
    reset.disabled = count === 0;
    status.textContent = `${count} of ${episodes.length} episodes included. Discount γ = ${gamma.toFixed(2)}. ${count ? 'Both methods use the same completed episodes.' : 'Include the first episode to see the returns.'}`;
    rows.replaceChildren();
    estimates.replaceChildren();
    if (!count) {
      path.textContent = 'Waiting for a completed episode.';
      const row = rows.insertRow();
      const cell = row.insertCell();
      cell.colSpan = 5;
      cell.textContent = 'No visits included yet.';
      const empty = document.createElement('p');
      empty.textContent = 'Values are unknown until a state contributes a return.';
      estimates.append(empty);
      return;
    }
    const current = episodes[count - 1];
    path.textContent = `Episode ${count}: ` + current.map(step => `${step.state} → (${step.reward >= 0 ? '+' : ''}${step.reward}) → `).join('') + 'terminal';
    for (const visit of episodeReturns(current, gamma)) {
      const row = rows.insertRow();
      for (const value of [String(visit.time), visit.state, String(visit.reward), format(visit.value), visit.first ? 'Both methods' : 'Every-visit only']) {
        row.insertCell().textContent = value;
      }
      if (!visit.first) row.classList.add('repeat');
    }
    for (const entry of compareVisits(episodes.slice(0, count), gamma)) {
      const card = document.createElement('section');
      card.className = 'estimate';
      const heading = document.createElement('h3');
      heading.textContent = `State ${entry.state}`;
      card.append(heading);
      for (const [label, samples, mean] of [
        ['First-visit', entry.first, entry.firstMean],
        ['Every-visit', entry.every, entry.everyMean],
      ] as const) {
        const block = document.createElement('div');
        const title = document.createElement('h4');
        title.textContent = `${label}: V = ${mean === null ? 'unknown' : format(mean)}`;
        const detail = document.createElement('p');
        detail.textContent = `${samples.length} sample${samples.length === 1 ? '' : 's'}: [${samples.map(format).join(', ')}]`;
        block.append(title, detail);
        card.append(block);
      }
      estimates.append(card);
    }
  }
  next.addEventListener('click', () => { count = Math.min(count + 1, episodes.length); render(); });
  reset.addEventListener('click', () => { count = 0; render(); });
  gammaInput.addEventListener('input', render);
  render();
}
