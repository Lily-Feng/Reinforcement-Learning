import { buildSearch, children, mean, positions, terminals, type Frame } from './search';

export function initSearchLab() {
  const get = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector<T>(s)!;
  const next = get<HTMLButtonElement>('[data-next]'), back = get<HTMLButtonElement>('[data-back]');
  const play = get<HTMLButtonElement>('[data-play]'), finish = get<HTMLButtonElement>('[data-finish]');
  const policy = get<HTMLSelectElement>('[data-policy]'), seed = get<HTMLInputElement>('[data-seed]');
  const speed = get<HTMLSelectElement>('[data-speed]'), timeline = get<HTMLInputElement>('[data-timeline]');
  const inspector = get<HTMLSelectElement>('[data-inspect]');
  let frames = buildSearch(), index = 0, inspected = 1;
  let timer: ReturnType<typeof setInterval> | undefined;
  const fixed = (v: number | null) => v === null ? '—' : v.toFixed(3);
  function stop() { clearInterval(timer); timer = undefined; play.textContent = 'Play'; }
  function showInspector(frame: Frame) {
    const node = frame.tree.find(n => n.id === inspected) ?? frame.tree[0];
    inspected = node.id; inspector.value = String(inspected);
    const untried = children(inspected).filter(id => !frame.tree.some(n => n.id === id));
    get('[data-inspector]').textContent = `State ${inspected}: W = ${node.wins}, N = ${node.visits}, V = ${node.visits ? `${node.wins}/${node.visits} = ${fixed(mean(node))}` : 'undefined (0 visits)'}. ${children(inspected).length ? `Untried successors: ${untried.join(', ') || 'none'}.` : `Terminal payoff: ${terminals[inspected]}. No outgoing actions.`}`;
    document.querySelectorAll<HTMLElement>('[data-stat-row]').forEach(row => row.classList.toggle('inspect', Number(row.dataset.statRow) === inspected));
  }
  function draw(frame: Frame) {
    const ids = new Set(frame.tree.map(n => n.id));
    const showRollout = ['rollout', 'return', 'backup'].includes(frame.phase);
    const ghosts = showRollout ? frame.rollout.filter(id => !ids.has(id)) : [];
    const visible = new Set([...ids, ...ghosts]);
    const edges: string[] = [], nodes: string[] = [];
    for (const id of visible) {
      const p = positions(id);
      if (id !== 1) {
        const parent = Math.floor(id / 2), from = positions(parent);
        const isRollout = ghosts.includes(id);
        const selected = frame.path.includes(id) && frame.path.includes(parent);
        const backup = frame.phase === 'backup' && frame.updated === parent && selected;
        edges.push(`<line class="edge ${isRollout ? 'rollout' : selected ? 'selected' : ''} ${backup ? 'backup' : ''}" x1="${from.x}" y1="${from.y}" x2="${p.x}" y2="${p.y}" />`);
      }
      const node = frame.tree.find(n => n.id === id);
      const ghost = !node;
      nodes.push(`<g class="node ${ghost ? 'ghost' : ''} ${frame.active === id ? 'active' : ''} ${frame.updated === id ? 'updated' : ''}" transform="translate(${p.x} ${p.y})" ${ghost ? '' : `data-inspect-node="${id}" role="button" tabindex="0" aria-label="Inspect state ${id}"`}><title>State ${id}: ${node ? `${node.wins} successes / ${node.visits} visits` : 'rollout only, not stored'}</title><text class="label" y="-40">State ${id}${id >= 8 ? ` · r=${terminals[id]}` : ''}</text><circle r="29"/><text class="count" y="6">${node ? `${node.wins}/${node.visits}` : id >= 8 ? `r = ${terminals[id]}` : 'rollout'}</text><text class="value" y="49">${node ? `V ${fixed(mean(node))}` : 'not stored'}</text></g>`);
      if (node) for (const child of children(id)) if (!visible.has(child)) {
        const dot = positions(child);
        edges.push(`<line class="edge" x1="${p.x}" y1="${p.y + 30}" x2="${dot.x}" y2="${dot.y}" stroke-dasharray="4 5"/>`);
        nodes.push(`<g><circle class="dot" cx="${dot.x}" cy="${dot.y}" r="6"/><text class="dot-label" x="${dot.x}" y="${dot.y + 24}">untried → ${child}</text></g>`);
      }
    }
    get('[data-edges]').innerHTML = edges.join(''); get('[data-nodes]').innerHTML = nodes.join('');
  }
  function render() {
    const f = frames[index];
    get('[data-title]').textContent = f.title; get('[data-detail]').textContent = f.detail;
    get('[data-formula]').textContent = f.formula;
    get('[data-iteration]').textContent = f.phase === 'start' ? 'Before search' : `Iteration ${f.iteration} of 18 · ${f.phase}`;
    get('[data-tree-size]').textContent = `${f.tree.length} stored ${f.tree.length === 1 ? 'node' : 'nodes'}`;
    get('[data-progress]').textContent = `${index} / ${frames.length - 1}`;
    timeline.max = String(frames.length - 1); timeline.value = String(index);
    get('[data-path]').textContent = f.path.join(' → ');
    get('[data-rollout]').textContent = f.rollout.length ? `${f.rollout.join(' → ')}${f.rollout.length === 1 ? (f.result === null ? ' · no rollout actions yet' : ' · reached terminal directly') : ''}` : 'Not started';
    get('[data-return]').textContent = f.result === null ? (f.phase === 'choose' ? 'Search finished · execute the recommended root action' : 'G not observed yet') : `Observed return G = ${f.result}${f.result ? ' · success' : ' · failure'}`;
    const phase = f.phase === 'return' ? 'rollout' : f.phase;
    document.querySelectorAll<HTMLElement>('[data-phase]').forEach(el => {
      const active = el.dataset.phase === phase; el.classList.toggle('active', active);
      if (active) el.setAttribute('aria-current', 'step'); else el.removeAttribute('aria-current');
    });
    get('[data-scores]').hidden = !f.choices.length;
    get('[data-choices]').innerHTML = f.choices.map(c => `<tr${c.id === f.active ? ' class="updated"' : ''}><td>${c.id}${c.id === f.active ? ' ✓' : ''}</td><td>${fixed(c.mean)}</td><td>${fixed(c.bonus)}</td><td>${fixed(c.score)}</td></tr>`).join('');
    const previous = frames[Math.max(0, index - 1)];
    get('[data-statistics]').innerHTML = f.tree.slice().sort((a,b) => a.id-b.id).map(n => {
      const old = previous.tree.find(p => p.id === n.id);
      const change = f.updated === n.id ? `W ${old?.wins ?? 0} → ${n.wins}; N ${old?.visits ?? 0} → ${n.visits}` : !old ? 'Added · value undefined' : '—';
      return `<tr data-stat-row="${n.id}" class="${f.updated === n.id ? 'updated' : ''}"><th scope="row">${n.id}</th><td>${n.wins}</td><td>${n.visits}</td><td>${fixed(mean(n))}</td><td>${children(n.id).filter(id => !f.tree.some(p => p.id === id)).join(', ') || (n.id >= 8 ? 'terminal' : 'none')}</td><td>${change}</td></tr>`;
    }).join('');
    inspector.innerHTML = f.tree.slice().sort((a,b) => a.id-b.id).map(n => `<option value="${n.id}">State ${n.id}</option>`).join('');
    showInspector(f); draw(f);
    get<HTMLSelectElement>('[data-iteration-jump]').value = String(f.iteration);
    next.disabled = index === frames.length - 1; back.disabled = index === 0; finish.disabled = next.disabled;
    if (next.disabled) stop();
  }
  function go(to: number) { stop(); index = Math.max(0, Math.min(frames.length - 1, to)); render(); }
  function reset() {
    stop(); index = 0; inspected = 1;
    const parsed = Number(seed.value);
    const safeSeed = Number.isFinite(parsed) ? Math.max(0, Math.min(999999, Math.floor(parsed))) : 7;
    seed.value = String(safeSeed);
    frames = buildSearch(policy.value === 'greedy' ? 'greedy' : 'uct', safeSeed);
    get<HTMLSelectElement>('[data-iteration-jump]').innerHTML = '<option value="0">Before search</option>' + Array.from({length:18}, (_,i) => `<option value="${i+1}">Iteration ${i+1}</option>`).join('');
    get<HTMLSelectElement>('[data-operation-jump]').value = '';
    render();
  }
  next.addEventListener('click', () => go(index + 1)); back.addEventListener('click', () => go(index - 1));
  finish.addEventListener('click', () => { const nextEnd = frames.findIndex((f, i) => i > index && (f.phase === 'complete' || f.phase === 'choose')); go(nextEnd < 0 ? frames.length - 1 : nextEnd); });
  play.addEventListener('click', () => {
    if (timer !== undefined) { stop(); return; }
    if (index === frames.length - 1) index = 0;
    index++; render();
    if (index < frames.length - 1) { play.textContent = 'Pause'; timer = setInterval(() => { index++; render(); }, Number(speed.value)); }
  });
  speed.addEventListener('change', stop); policy.addEventListener('change', reset); seed.addEventListener('change', reset);
  get('[data-reset]').addEventListener('click', reset);
  timeline.addEventListener('input', () => go(Number(timeline.value)));
  get<HTMLSelectElement>('[data-iteration-jump]').addEventListener('change', e => { const n = Number((e.target as HTMLSelectElement).value); go(frames.findIndex(f => f.iteration === n)); });
  get<HTMLSelectElement>('[data-operation-jump]').addEventListener('change', e => {
    const select = e.target as HTMLSelectElement;
    const match = (f: Frame) => select.value === 'backup-average' ? f.phase === 'backup' && f.tree.some(n => n.id === f.updated && n.visits > 1 && n.wins / n.visits !== (n.wins - Number(f.result)) / (n.visits - 1)) : select.value === 'select-score' ? !!f.choices.length : select.value === 'return-success' ? f.phase === 'return' && f.result === 1 : select.value === 'return-failure' ? f.phase === 'return' && f.result === 0 : select.value === 'terminal' ? f.phase === 'return' && f.rollout.length === 1 : select.value === 'rollout' ? f.phase === 'rollout' && f.rollout.length > 1 : f.phase === select.value;
    const found = frames.findIndex(match); if (found >= 0) go(found);
    select.value = '';
  });
  inspector.addEventListener('change', () => { inspected = Number(inspector.value); showInspector(frames[index]); });
  function inspectEvent(event: Event) {
    const el = (event.target as Element).closest<SVGElement>('[data-inspect-node]');
    if (el) { inspected = Number(el.dataset.inspectNode); showInspector(frames[index]); }
  }
  get('[data-tree]').addEventListener('click', inspectEvent);
  get('[data-tree]').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); inspectEvent(e); } });
  window.addEventListener('pagehide', stop);
  reset();
}
