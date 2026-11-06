/**
 * Module metadata. Each module is code-split: `load()` is a dynamic import,
 * so opening one section downloads only that section's modules.
 */

import type { ModuleContext } from './context';

export interface LabModule {
  mount: (el: HTMLElement, ctx: ModuleContext) => () => void;
}

export interface ModuleInfo {
  id: string;
  section: SectionId;
  title: string;
  blurb: string;
  load: () => Promise<LabModule>;
}

export type SectionId = 'problem' | 'episodic' | 'n-step' | 'average-reward' | 'discounting' | 'review';

export const SECTIONS: { id: SectionId; title: string; book: string }[] = [
  { id: 'problem', title: 'The problem', book: 'Running example' },
  { id: 'episodic', title: 'Episodic semi-gradient control', book: '§10.1' },
  { id: 'n-step', title: 'n-step Sarsa', book: '§10.2' },
  { id: 'average-reward', title: 'Average reward', book: '§10.3, §10.5' },
  { id: 'discounting', title: 'Deprecating discounting', book: '§10.4' },
  { id: 'review', title: 'Review', book: 'Check yourself' },
];

export const MODULES: ModuleInfo[] = [
  { id: 'm01', section: 'problem', title: 'Playground: you are the policy', blurb: 'Accept or reject requests yourself and race accept-all and the reference.', load: () => import('../modules/m01-playground') },
  { id: 'm02', section: 'problem', title: 'How the agent sees the cluster', blurb: 'Tile coding over free GPUs, job length, ρ and time of day.', load: () => import('../modules/m02-tiles') },
  { id: 'm03', section: 'episodic', title: 'One semi-gradient Sarsa update', blurb: 'Every number in one update, and why it is only semi-gradient.', load: () => import('../modules/m03-update') },
  { id: 'm04', section: 'episodic', title: 'Training over shifts', blurb: 'Learning curves for three step sizes and the learned shift policy.', load: () => import('../modules/m04-shifts') },
  { id: 'm05', section: 'episodic', title: 'Optimism explores even with ε = 0', blurb: 'Initial values decide what a greedy agent ever tries.', load: () => import('../modules/m05-optimism') },
  { id: 'm06', section: 'n-step', title: 'Building G_{t:t+n}', blurb: 'Bias from bootstrapping vs. variance from summing rewards.', load: () => import('../modules/m06-targets') },
  { id: 'm07', section: 'n-step', title: 'Which n learns fastest?', blurb: 'A sweep over n and α, with error bars.', load: () => import('../modules/m07-sweep') },
  { id: 'm08', section: 'average-reward', title: 'Where does the cluster spend its time?', blurb: 'The on-policy distribution μ_π and r(π) = Σ μ r.', load: () => import('../modules/m08-distribution') },
  { id: 'm09', section: 'average-reward', title: 'When does the limit exist?', blurb: 'Ergodicity, periodic chains and time averages.', load: () => import('../modules/m09-ergodicity') },
  { id: 'm10', section: 'average-reward', title: 'Values when the sum oscillates', blurb: 'Abel limits, f_s(γ), and differential values.', load: () => import('../modules/m10-values') },
  { id: 'm11', section: 'average-reward', title: 'Values are defined up to a constant', blurb: 'Shift every value: no TD error changes.', load: () => import('../modules/m11-constant') },
  { id: 'm12', section: 'average-reward', title: 'How R̄ should be updated', blurb: 'TD error vs. reward error, and four R̄ rules.', load: () => import('../modules/m12-rbar') },
  { id: 'm13', section: 'average-reward', title: 'The main lab: differential Sarsa', blurb: 'Learn admission control and compare with the exact optimum.', load: () => import('../modules/m13-lab') },
  { id: 'm14', section: 'discounting', title: 'γ cannot change the ranking', blurb: 'J_γ(π) = r(π)/(1 − γ) across 20 policies.', load: () => import('../modules/m14-ranking') },
  { id: 'm15', section: 'discounting', title: 'γ as an algorithm parameter', blurb: 'Discounted Sarsa at four γ vs. differential Sarsa.', load: () => import('../modules/m15-gamma') },
  { id: 'm16', section: 'discounting', title: 'The derivation, with numbers', blurb: 'Each line of J = r(π)/(1 − γ), evaluated.', load: () => import('../modules/m16-derivation') },
  { id: 'm17', section: 'discounting', title: 'No policy improvement theorem', blurb: 'Greedy policies that chatter under shared features.', load: () => import('../modules/m17-chatter') },
  { id: 'm18', section: 'review', title: 'Check yourself', blurb: 'True/false and numeric questions with explanations.', load: () => import('../modules/m18-concept-check') },
];

export const moduleById = (id: string) => MODULES.find(m => m.id === id);
export const sectionModules = (s: SectionId) => MODULES.filter(m => m.section === s);
export const isSection = (s: string): s is SectionId => SECTIONS.some(x => x.id === s);
