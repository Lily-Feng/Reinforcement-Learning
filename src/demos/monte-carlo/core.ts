/** Fixed, complete teaching episodes. No model or random sampling is inferred. */
export interface Transition {
  state: string;
  reward: number;
}

export const episodes: readonly (readonly Transition[])[] = [
  [{ state: 'A', reward: 2 }, { state: 'B', reward: -1 }, { state: 'A', reward: 3 }],
  [{ state: 'A', reward: 0 }, { state: 'B', reward: 2 }],
  [{ state: 'B', reward: -2 }, { state: 'A', reward: 1 }, { state: 'B', reward: 4 }],
];

export interface Visit extends Transition {
  time: number;
  value: number;
  first: boolean;
}

export function episodeReturns(episode: readonly Transition[], gamma: number): Visit[] {
  if (!Number.isFinite(gamma) || gamma < 0 || gamma > 1) {
    throw new RangeError('Discount must be between 0 and 1.');
  }
  const values = new Array<number>(episode.length);
  let value = 0;
  for (let time = episode.length - 1; time >= 0; time--) {
    value = episode[time].reward + gamma * value;
    values[time] = value;
  }
  const seen = new Set<string>();
  return episode.map((step, time) => {
    const first = !seen.has(step.state);
    seen.add(step.state);
    return { ...step, time, value: values[time], first };
  });
}

export interface Estimate {
  state: string;
  first: number[];
  every: number[];
  firstMean: number | null;
  everyMean: number | null;
}

export function compareVisits(batch: readonly (readonly Transition[])[], gamma: number): Estimate[] {
  const samples = new Map<string, { first: number[]; every: number[] }>();
  for (const episode of batch) {
    for (const visit of episodeReturns(episode, gamma)) {
      const entry = samples.get(visit.state) ?? { first: [], every: [] };
      entry.every.push(visit.value);
      if (visit.first) entry.first.push(visit.value);
      samples.set(visit.state, entry);
    }
  }
  const mean = (values: number[]) => values.length
    ? values.reduce((total, value) => total + value, 0) / values.length
    : null;
  return [...samples.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([state, entry]) => ({
    state, ...entry, firstMean: mean(entry.first), everyMean: mean(entry.every),
  }));
}
