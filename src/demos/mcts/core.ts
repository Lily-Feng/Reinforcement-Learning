/** One illustrative MCTS iteration: select, expand, simulate, then back up. */
export const initialCounts = [[2, 4], [0, 1], [1, 1], [1, 2], [0, 1]] as const;
export const lastStep = 6;

export function greedyChild(): number {
  const children = [2, 3, 4];
  return children.reduce((best, id) => {
    const [wins, visits] = initialCounts[id - 1];
    const [bestWins, bestVisits] = initialCounts[best - 1];
    return wins / visits > bestWins / bestVisits ? id : best;
  });
}

export function snapshot(step: number, success = true) {
  const counts: [number, number][] = initialCounts.map(([wins, visits]) => [wins, visits]);
  // A return increments visits once at every node on this iteration's path.
  if (step >= 5) {
    counts[2][0] += Number(success);
    counts[2][1]++;
  }
  if (step >= 6) {
    counts[0][0] += Number(success);
    counts[0][1]++;
  }
  return { counts, child: step >= 2 ? (step >= 4 ? [Number(success), 1] : [0, 0]) : null };
}
