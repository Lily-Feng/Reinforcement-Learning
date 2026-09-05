---
title: Multi-armed bandits
description: "Exploration versus exploitation in the simplest reinforcement learning problem — one state, several actions, immediate reward."
sidebar:
  order: 1
---

## Learning objective

Understand why exploration is necessary at all, and be able to state the
exploration–exploitation trade-off precisely enough to measure it.

## The setup

You face $k$ slot machines. Each machine $a$ pays out according to a fixed but
**unknown** distribution with mean $q_*(a)$:

$$
q_*(a) = \mathbb{E}[R_t \mid A_t = a]
$$

At each step you pick one machine and receive one reward. The distributions never
change. You get 1000 pulls. Maximise the total.

That is the whole problem. There is no state — the situation after 500 pulls is
identical to the situation at the start, apart from what you have learned. There
is no delayed consequence — the reward arrives immediately and affects nothing
else.

## Why strip it down this far

Removing state and delay leaves exactly one difficulty behind: **you cannot
learn about an action without giving up reward.**

If you knew every $q_*(a)$, the problem would be trivial — always pull the best
machine. You do not, so you hold estimates $Q_t(a)$. At any moment:

- **Exploiting** means picking $\arg\max_a Q_t(a)$: the best return given what you
  currently believe.
- **Exploring** means picking something else: worse in expectation right now,
  but it improves the estimate that all your future decisions depend on.

You cannot do both on the same pull. Every algorithm in this chapter is a
different answer to *when is a worse action worth taking?*

## Measuring how well you did

Total reward alone is a poor yardstick, because a lucky run on an easy set of
arms beats a smart run on a hard one. Two better measures:

**Percent optimal action** — the share of pulls that hit the truly best arm. It
goes to 100% for a method that eventually finds and keeps the best arm.

**Regret** — the reward you gave up by not always playing the best arm:

$$
\mathrm{Regret}_T = \sum_{t=1}^{T} \left( q_*(a^*) - q_*(A_t) \right)
$$

Regret only ever increases. What matters is *how fast*. A method that keeps
exploring at a fixed rate accrues regret linearly forever; a method that anneals
its exploration can get sublinear regret.

## Contents

- [Action-value methods](/Reinforcement-Learning/chapters/02-multi-armed-bandits/action-value-methods/) — how to estimate $q_*(a)$ from experience.
- [Epsilon-greedy](/Reinforcement-Learning/chapters/02-multi-armed-bandits/epsilon-greedy/) — the simplest exploration rule that works, with an interactive simulator.

## Personal takeaways

The bandit problem is worth taking seriously rather than treating as a warm-up.
Nearly every exploration idea in deep RL — optimistic initialisation, upper
confidence bounds, decaying $\varepsilon$ — is a bandit idea that got carried
into a larger setting mostly unchanged.

## References

- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., ch. 2.
