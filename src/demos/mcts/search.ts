/** A finite deterministic environment; only rollout action choices are sampled. */
export interface SearchNode { id: number; wins: number; visits: number }
export interface Choice { id: number; mean: number | null; bonus: number | null; score: number | null }
export interface Frame {
  phase: 'start' | 'select' | 'expand' | 'rollout' | 'return' | 'backup' | 'complete' | 'choose';
  iteration: number; title: string; detail: string; formula: string;
  tree: SearchNode[]; path: number[]; rollout: number[]; active: number;
  result: number | null; choices: Choice[]; updated: number | null;
}
export const terminals: Record<number, number> = { 8: 0, 9: 1, 10: 1, 11: 0, 12: 1, 13: 0, 14: 0, 15: 1 };
export const children = (id: number): number[] => id < 8 ? [id * 2, id * 2 + 1] : [];
export const mean = (node: SearchNode) => node.visits ? node.wins / node.visits : null;
export const positions = (id: number) => {
  const depth = Math.floor(Math.log2(id));
  return { x: (id - 2 ** depth + .5) * 1000 / 2 ** depth, y: 65 + depth * 135 };
};

export function buildSearch(policy: 'uct' | 'greedy' = 'uct', seed = 7, budget = 18): Frame[] {
  const tree = new Map<number, SearchNode>([[1, { id: 1, wins: 0, visits: 0 }]]);
  const frames: Frame[] = [];
  let randomState = seed >>> 0;
  const random = () => { randomState = (1664525 * randomState + 1013904223) >>> 0; return randomState / 4294967296; };
  let path = [1], rollout: number[] = [], result: number | null = null, iteration = 0;
  const emit = (phase: Frame['phase'], active: number, title: string, detail: string, formula = '', choices: Choice[] = [], updated: number | null = null) => {
    frames.push({ phase, active, title, detail, formula, choices, updated, iteration, result,
      tree: [...tree.values()].map(n => ({ ...n })), path: [...path], rollout: [...rollout] });
  };
  emit('start', 1, 'Start with an empty search tree', 'Only state 1 is stored. The model has two actions per nonterminal state. Expand one untried action, then simulate to a terminal. All intermediate rewards are 0; a terminal pays 0 or 1.');
  for (iteration = 1; iteration <= budget; iteration++) {
    path = [1]; rollout = []; result = null;
    let current = 1;
    emit('select', current, `Iteration ${iteration}: start at the root`, 'Follow the stored tree. If a state has an untried action, stop selection there and expand it.');
    while (children(current).length) {
      const next = children(current);
      const untried = next.find(id => !tree.has(id));
      if (untried !== undefined) {
        tree.set(untried, { id: untried, wins: 0, visits: 0 });
        const parent = current;
        current = untried; path.push(current);
        emit('expand', current, `Add state ${current} to the tree`, `Take the first untried action at state ${parent}. Create exactly one stored node with W = 0 and N = 0. Its value is undefined until backup.`, `Tree: ${tree.size - 1} → ${tree.size} nodes; V(${current}) = 0/0 is undefined`);
        break;
      }
      const choices: Choice[] = next.map(id => {
        const node = tree.get(id)!;
        const value = mean(node);
        const bonus = policy === 'uct' ? Math.SQRT2 * Math.sqrt(Math.log(Math.max(1, tree.get(current)!.visits)) / node.visits) : 0;
        return { id, mean: value, bonus, score: value === null ? null : value + bonus };
      });
      const selected = choices.reduce((a, b) => (b.score ?? Infinity) > (a.score ?? Infinity) ? b : a);
      const parent = current;
      current = selected.id; path.push(current);
      emit('select', current, `Select ${parent} → ${current}`, policy === 'uct' ? 'Compare mean return plus exploration bonus. Choose the largest UCT score; ties go left. The exploration bonus is used only for selection.' : 'Choose the largest mean return. Ties go left. Untried actions are expanded before comparing visited children.', policy === 'uct' ? `UCT = W/N + √2 × √(ln N(parent) / N(child))` : 'Greedy score = W/N', choices);
    }
    rollout = [current];
    if (children(current).length) {
      emit('rollout', current, 'Begin a rollout from the new node', 'The rollout policy chooses left or right with equal probability using the seed. It uses the environment model. Temporary rollout states are not added to the stored search tree.');
      while (children(current).length) {
        const parent = current;
        const next = children(current);
        current = next[random() < .5 ? 0 : 1];
        rollout.push(current);
        const terminal = !children(current).length;
        emit('rollout', current, `Rollout: ${parent} → ${current}`, `The rollout policy chose ${current === parent * 2 ? 'left' : 'right'}. The deterministic model returns state ${current} and reward ${terminal ? terminals[current] : 0}.${terminal ? ' This state is terminal.' : ' Continue simulating.'}`, `Model(${parent}, ${current === parent * 2 ? 'left' : 'right'}) → (state ${current}, reward ${terminal ? terminals[current] : 0})`);
      }
    }
    result = terminals[current];
    emit('return', current, result ? 'Terminal success: calculate the return' : 'Terminal failure: calculate the return', rollout.length === 1 ? 'Selection or expansion reached a terminal directly. No further rollout actions are needed. Back up its observed payoff.' : 'The rollout has reached a terminal. With γ = 1 and zero intermediate rewards, the return is the terminal payoff. Rollout-only states keep no tree statistics.', `G = ${rollout.slice(1).map((id, i, arr) => i === arr.length - 1 ? String(terminals[id]) : '0').join(' + ') || String(result)} = ${result}`);
    for (const id of [...path].reverse()) {
      const node = tree.get(id)!;
      const before = { ...node };
      node.wins += result; node.visits++;
      emit('backup', id, `Back up G = ${result} to state ${id}`, `Add this same sampled return once to each stored node on the selected path. ${result ? 'Success increases W and N.' : 'Failure increases N but leaves W unchanged.'} Siblings and rollout-only states receive no update.`, `W: ${before.wins} + ${result} = ${node.wins}; N: ${before.visits} + 1 = ${node.visits}; V: ${before.visits ? (before.wins / before.visits).toFixed(3) : 'undefined'} → ${node.wins}/${node.visits} = ${(node.wins / node.visits).toFixed(3)}`, [], id);
    }
    emit('complete', 1, `Iteration ${iteration} complete`, 'The updated tree persists into the next iteration. Temporary rollout states disappear; future iterations may add them through expansion.', `Root: ${tree.get(1)!.wins}/${tree.get(1)!.visits} = ${mean(tree.get(1)!)!.toFixed(3)}`);
  }
  iteration = budget;
  const recommendation = children(1).map(id => tree.get(id)!).filter(Boolean).reduce((a, b) => b.visits > a.visits ? b : a);
  path = [1, recommendation.id]; rollout = []; result = null;
  emit('choose', recommendation.id, `Choose root action 1 → ${recommendation.id}`, 'The search budget is exhausted. Choose the most visited root action (ties go left). This is the real action recommendation; search simulations did not move the real agent.', `State ${recommendation.id}: N = ${recommendation.visits}, mean return = ${mean(recommendation)!.toFixed(3)}`);
  return frames;
}
