import { ACTIONS, ARROWS, advance, backup, greedy, initial, transition, type Config, type Mode, type Snapshot } from './core';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (v: number) => (Math.abs(v) < 0.0000005 ? 0 : v).toFixed(3);
const small = (v: number) => v === 0 ? '0' : v < 0.001 ? v.toExponential(2) : v.toFixed(4);
let config: Config = { size: 2, gamma: 1, theta: 0.0001, mode: 'value' };
let state = initial(config);
let selected = 0;
let timer: ReturnType<typeof setInterval> | undefined;
let past: { state: Snapshot; selected: number }[] = [];
const descriptions: Record<Mode, string> = {
  value: 'Choose the best action at every backup. Values approach the optimal return.',
  evaluation: 'Keep a uniform random policy fixed: each action has probability ¼. Estimate its return.',
  policy: 'Learning extension: evaluate a policy, then make it greedy. Repeat until stable.',
};
function stop() { if (timer !== undefined) clearInterval(timer); timer = undefined; $('play').textContent = '▶ Run'; }
function save() { past.push({ state, selected }); if (past.length > 500) past.shift(); }
function reset() {
  stop();
  config = { size: Number($<HTMLSelectElement>('world').value), gamma: Number($<HTMLInputElement>('gamma').value), theta: Number($<HTMLSelectElement>('theta').value), mode: $<HTMLSelectElement>('algorithm').value as Mode };
  state = initial(config); selected = 0; past = []; render();
}
function act(sweep: boolean) {
  if (state.phase === 'done') return;
  save();
  const count = state.sweeps;
  const phase = state.phase;
  do { state = advance(config, state); }
  while (sweep && phase === 'evaluate' && state.phase === 'evaluate' && state.sweeps === count);
  if (state.trace) selected = state.trace.state;
  // Keep the final nonterminal backup visible after a full sweep.
  if (sweep) selected = Math.min(selected, state.v.length - 2);
  if (state.phase === 'done' || state.sweeps >= 10000) stop();
  render();
  if (state.sweeps >= 10000 && state.phase !== 'done') $('status').textContent = 'Paused at 10,000 sweeps. Try a smaller discount or a larger threshold.';
}
function render() {
  $('gamma-value').textContent = config.gamma.toFixed(2);
  $('algorithm-help').textContent = descriptions[config.mode];
  $('sweeps').textContent = String(state.sweeps);
  $('delta').textContent = state.lastDelta === null ? '—' : small(state.lastDelta);
  $('improvements').textContent = String(state.improvements);
  $('phase').textContent = state.phase === 'done' ? (config.mode === 'policy' ? 'Policy stable' : 'Converged') : state.phase === 'improve' ? 'Improve policy' : config.mode === 'value' ? 'Value update' : 'Evaluate policy';
  $<HTMLButtonElement>('step').textContent = state.phase === 'improve' ? 'Improve policy →' : 'Step state →';
  $<HTMLButtonElement>('sweep').textContent = state.phase === 'improve' ? 'Improve policy' : 'Full sweep';
  for (const id of ['step', 'sweep', 'play']) $<HTMLButtonElement>(id).disabled = state.phase === 'done';
  $<HTMLButtonElement>('undo').disabled = !past.length;
  $('arrow-label').textContent = config.mode === 'value' ? 'Arrows: greedy from stored Q' : 'Arrows: current policy';
  const focus = document.activeElement instanceof HTMLElement ? document.activeElement.dataset.state : undefined;
  const grid = $('grid');
  grid.classList.toggle('large', config.size === 4);
  grid.style.gridTemplateColumns = `repeat(${config.size}, 1fr)`;
  grid.replaceChildren();
  const scale = Math.max(1, ...state.v.map(Math.abs));
  state.v.forEach((v, s) => {
    const cell = document.createElement('button');
    const terminal = s === state.v.length - 1;
    const policy = config.mode === 'value' ? ARROWS[greedy(state.q[s])] : state.policy[s].map((p, a) => p > 0 ? ARROWS[a] : '').join('');
    cell.className = `cell${terminal ? ' terminal' : ''}`;
    cell.dataset.state = String(s);
    cell.setAttribute('aria-pressed', String(s === selected));
    cell.setAttribute('aria-label', `State ${s}${terminal ? ', terminal' : ''}, value ${fmt(v)}${terminal ? '' : `, policy ${policy}`}`);
    if (!terminal) cell.style.background = `hsl(177 22% ${26 - Math.abs(v) / scale * 12}%)`;
    cell.innerHTML = `<span class="cell-index">s${s}</span><span class="cell-value">${terminal ? '★' : fmt(v)}</span><span class="cell-arrow">${terminal ? '0 · GOAL' : policy}</span>`;
    cell.addEventListener('click', () => { selected = s; render(); });
    grid.append(cell);
  });
  if (focus !== undefined) grid.querySelector<HTMLElement>(`[data-state="${focus}"]`)?.focus({ preventScroll: true });
  const trace = state.trace?.state === selected ? state.trace : backup(config, state, selected);
  const actual = trace === state.trace;
  $('state-label').textContent = `s${selected}`;
  $('backup-title').textContent = actual ? 'Last executed backup' : 'Next backup preview';
  $('backup-caption').textContent = actual ? 'These are the values read at the instant this state was updated.' : 'Using the current V table; inspecting this state does not update it.';
  $('formula').textContent = config.mode === 'value' ? 'V(s) ← maxₐ Q(s,a)' : 'V(s) ← Σₐ π(a|s) Q(s,a)';
  $('q-table').innerHTML = trace.q.map((q, a) => {
    const t = transition(config.size, selected, a);
    return `<tr class="${trace.weights[a] > 0 ? 'chosen' : ''}"><td>${ARROWS[a]} ${ACTIONS[a]}</td><td>s${t.next}${t.done ? ' ★' : ''}</td><td>${fmt(q)}</td><td>${trace.weights[a].toFixed(2)}</td></tr>`;
  }).join('');
  const a = greedy(trace.q);
  const terms = trace.q.map((q, i) => `${trace.weights[i].toFixed(2)} × (${fmt(q)})`).filter((_, i) => trace.weights[i] > 0);
  $('calculation').innerHTML = `<span>${config.mode === 'value' ? `max(${trace.q.map(fmt).join(', ')})` : terms.join(' + ')}</span><br /><strong>V(s${selected}): ${fmt(trace.old)} → ${fmt(trace.value)}</strong><br /><span>|change| = ${small(Math.abs(trace.value - trace.old))}</span>`;
  $('tie-note').textContent = config.mode === 'value' ? `Greedy choice: ${ACTIONS[a]}. Ties take the first maximum in [left, up, right, down].` : 'Weights come from the fixed policy being evaluated. An average evaluates the policy; a maximum optimizes the value.';
  $('transition-code').textContent = ACTIONS.map((name, a) => {
    const t = transition(config.size, selected, a);
    return `# ${a}: ${name}; future = ${fmt(trace.future[a])}\nP[${selected}][${a}] = [(1.0, ${t.next}, ${t.reward}, ${t.done ? 'True' : 'False'})]`;
  }).join('\n');
  const status = state.phase === 'done'
    ? config.mode === 'policy' ? `Policy stable after ${state.improvements} improvements and ${state.sweeps} evaluation sweeps.` : `Converged: last sweep Δ = ${small(state.lastDelta ?? 0)} < θ = ${config.theta}.`
    : state.phase === 'improve' ? 'Evaluation reached tolerance. Next: improve the policy in all states.'
    : `Next update: s${state.cursor} · sweep ${state.sweeps + 1}.${state.changes !== null ? ` Last improvement changed ${state.changes} nonterminal states.` : ''}`;
  $('status').textContent = status;
  const values = state.history;
  const max = Math.max(0.01, ...values.map(v => Math.log10(1 + v)));
  const points = values.map((v, i) => `${20 + i / Math.max(1, values.length - 1) * 560},${90 - Math.log10(1 + v) / max * 70}`).join(' ');
  $('chart').innerHTML = `<path d="M20 12 V90 H580" fill="none" stroke="#40515a"/><text x="20" y="106" fill="#a4b5ba" font-size="9">1</text><text x="560" y="106" fill="#a4b5ba" font-size="9">${values.length || ''}</text>${values.length ? `<polyline points="${points}" fill="none" stroke="#c6ef85" stroke-width="2"/>${values.length === 1 ? `<circle cx="20" cy="${90 - Math.log10(1 + values[0]) / max * 70}" r="3" fill="#c6ef85"/>` : ''}` : '<text x="200" y="52" fill="#a4b5ba" font-size="12">Waiting for the first sweep</text>'}`;
  $('chart-description').textContent = values.length ? `${values.length} sweeps · first Δ ${small(values[0])} → latest Δ ${small(values.at(-1)!)}. ${config.mode === 'policy' ? 'A policy improvement can make Δ rise again.' : 'Smaller Δ means values are changing less.'}` : 'Run a sweep to see how much the values change.';
}
$('step').addEventListener('click', () => { stop(); act(false); });
$('sweep').addEventListener('click', () => { stop(); act(true); });
$('play').addEventListener('click', () => {
  if (timer !== undefined) { stop(); return; }
  $('play').textContent = 'Ⅱ Pause';
  timer = setInterval(() => act(true), 350);
});
$('reset').addEventListener('click', reset);
$('undo').addEventListener('click', () => { stop(); const previous = past.pop(); if (previous) { state = previous.state; selected = previous.selected; render(); } });
for (const id of ['algorithm', 'world', 'gamma', 'theta']) $(id).addEventListener('input', reset);
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
window.addEventListener('pagehide', stop);
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-view]')) {
  button.addEventListener('click', () => {
    stop();
    for (const b of document.querySelectorAll<HTMLButtonElement>('[data-view]')) b.setAttribute('aria-pressed', String(b === button));
    for (const panel of document.querySelectorAll<HTMLElement>('.view')) panel.hidden = panel.id !== button.dataset.view;
  });
}
const sources = import.meta.glob('./python/*.py', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;
const files = [
  ['assignments/dp.py', 'Value prediction and value iteration. Both use in-place sweeps, explicit terminal handling, and a maximum-change stopping criterion.'],
  ['assignments/policy_deterministic_greedy.py', 'The deterministic greedy policy stores Q, chooses np.argmax, and returns action probabilities of 0 or 1.'],
  ['interfaces/policy.py', 'The policy contract: action(state) and action_prob(state, action). DP depends on this interface, not a specific policy class.'],
  ['interfaces/random_policy.py', 'A fixed action distribution. The evaluation demo uses its default uniform probabilities.'],
  ['lib/envs/grid_world.py', 'The environment base class defines state coordinates, legal moves, and the model-based environment interface.'],
  ['lib/envs/grid_world_2x2.py', 'The exact 2 × 2 transition model used in the lab: deterministic moves, −1 per move, state 3 terminal, and wall self-loops.'],
  ['lib/envs/wrapped_gridworld.py', 'Grid wrappers and visualizers. Actions are ordered LEFT=0, UP=1, RIGHT=2, DOWN=3; the browser uses the same order.'],
  ['run.py', 'The project entry point connects environments, algorithms, and visualization. The website replaces the Python visualizer with HTML.'],
  ['tests/dp.py', 'Original Python tests, including the expected optimal 2 × 2 values [−2, −1, −1, 0]. Other environments in this file are outside this demo’s scope.'],
];
let currentSource = '';
let currentPath = '';
let group = '';
for (const [path, description] of files) {
  const folder = path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : 'project root';
  if (folder !== group) { const label = document.createElement('p'); label.className = 'file-group'; label.textContent = `${folder}/`; $('file-tree').append(label); group = folder; }
  const button = document.createElement('button'); button.className = 'file-button'; button.textContent = `└ ${path.split('/').at(-1)}`;
  button.setAttribute('aria-pressed', 'false');
  button.addEventListener('click', () => {
    for (const b of document.querySelectorAll('.file-button')) b.setAttribute('aria-pressed', String(b === button));
    $('file-name').textContent = path; $('file-description').textContent = description;
    currentSource = sources[`./python/${path.replaceAll('/', '__')}`]; currentPath = path;
    $('python-code').replaceChildren();
    for (const line of currentSource.split('\n')) {
      const span = document.createElement('span');
      span.className = `code-line${line.trim().startsWith('#') ? ' comment' : /^\s*(class|def) /.test(line) ? ' definition' : ''}`;
      span.textContent = line || ' '; $('python-code').append(span);
    }
  });
  $('file-tree').append(button);
}
document.querySelector<HTMLButtonElement>('.file-button')!.click();
$('download-source').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([currentSource], { type: 'text/x-python' }));
  const a = document.createElement('a'); a.href = url; a.download = currentPath.split('/').at(-1)!; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
const embedDialog = $<HTMLDialogElement>('embed-dialog');
if (new URLSearchParams(location.search).get('embed') === '1') document.body.classList.add('embedded');
$('embed-open').addEventListener('click', () => {
  const url = new URL(location.href); url.search = '?embed=1'; url.hash = '';
  $<HTMLTextAreaElement>('embed-code').value = `<iframe\n  src="${url.href}"\n  title="Dynamic programming learning lab"\n  width="100%" height="1100" loading="lazy"\n  style="border:0; border-radius:12px">\n</iframe>`;
  $('embed-feedback').textContent = location.hostname === 'localhost' || location.hostname === '127.0.0.1' ? 'This is a local preview URL. Open the hosted demo to copy a public embed URL.' : 'The iframe is responsive; adjust its height to fit your page.';
  embedDialog.showModal();
});
$('embed-close').addEventListener('click', () => embedDialog.close());
$('copy-embed').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($<HTMLTextAreaElement>('embed-code').value); $('embed-feedback').textContent = 'Embed code copied.'; }
  catch { $<HTMLTextAreaElement>('embed-code').select(); $('embed-feedback').textContent = 'Copy the selected embed code using your keyboard.'; }
});
render();
