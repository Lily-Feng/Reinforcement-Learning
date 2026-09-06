---
title: Multi-armed bandits
description: "Exploration versus exploitation in the simplest reinforcement learning problem — one state, several actions, immediate reward."
sidebar:
  order: 1
---

A one-armed bandit is a slot machine. A *multi-armed* bandit is a row of them,
each paying out at its own unknown rate, and you only get so many pulls. Every
pull spends one of two things: reward, or information about which machine to
pull next. You cannot buy both with the same coin.

That trade is the whole subject of this chapter. It is reinforcement learning
with state and delayed consequences deleted, leaving the one difficulty that
neither supervised learning nor planning has to face — the learner has to act in
order to find out what acting is worth.

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

## A simple bandit algorithm

Here is a complete agent for the problem, in eight lines. Everything else in the
chapter is an explanation of one of them.

<div class="pseudocode">

**A simple bandit algorithm**

Initialise, for $a = 1$ to $k$:

$\quad Q(a) \leftarrow 0$

$\quad N(a) \leftarrow 0$

**Loop forever:**

$\quad A \leftarrow \begin{cases} \arg\max_a Q(a) & \text{with probability } 1-\varepsilon \quad \text{(breaking ties randomly)} \\ \text{a random action} & \text{with probability } \varepsilon \end{cases}$

$\quad R \leftarrow \mathrm{bandit}(A)$

$\quad N(A) \leftarrow N(A) + 1$

$\quad Q(A) \leftarrow Q(A) + \dfrac{1}{N(A)}\left[R - Q(A)\right]$

</div>

Three pieces are doing the work, and each gets a page of its own:

- $Q(a)$ and $N(a)$ — the estimate of $q_*(a)$ and the count it averages over.
  Why the running average is written incrementally rather than recomputed is
  [action-value methods](/Reinforcement-Learning/chapters/02-multi-armed-bandits/action-value-methods/).
- The choice of $A$ — the $\varepsilon$ coin flip that decides exploit or
  explore on this step, and how much that choice costs, is
  [epsilon-greedy](/Reinforcement-Learning/chapters/02-multi-armed-bandits/epsilon-greedy/).
- $\mathrm{bandit}(A)$ — the environment. It draws a reward from the unknown
  distribution behind arm $A$ and tells you nothing else. No transition, no next
  state; the loop returns to exactly where it started.

Two details in that box are easy to skim past and expensive to get wrong.
Ties must be broken **randomly** — with every $Q(a)$ starting at 0, the first
step is a $k$-way tie, and an `argmax` that returns the lowest index turns the
agent into one that only explores by accident. And the step size is
$1/N(A)$, the count for the arm actually taken, not the global step number.

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
- [Tracking a nonstationary problem](/Reinforcement-Learning/chapters/02-multi-armed-bandits/nonstationary-problems/) — what changes when $q_*(a)$ drifts, and why the step size is really a memory length.
- [Optimistic initial values](/Reinforcement-Learning/chapters/02-multi-armed-bandits/optimistic-initial-values/) — how a starting value alone can make a greedy agent explore, and why only once.
- [Upper-confidence-bound action selection](/Reinforcement-Learning/chapters/02-multi-armed-bandits/upper-confidence-bound/) — explore the arm you are least sure about, and why the learning curve spikes at step 11.
- [Gradient bandit algorithms](/Reinforcement-Learning/chapters/02-multi-armed-bandits/gradient-bandit/) — learn preferences rather than values, derived as stochastic gradient ascent, and what the baseline is really for.

## Personal takeaways

The bandit problem is worth taking seriously rather than treating as a warm-up.
Nearly every exploration idea in deep RL — optimistic initialisation, upper
confidence bounds, decaying $\varepsilon$ — is a bandit idea that got carried
into a larger setting mostly unchanged.

## References

- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., ch. 2.
