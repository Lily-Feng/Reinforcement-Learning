---
title: Tracking a nonstationary problem
description: "When the true action values drift, the sample average is the wrong estimator. Constant step sizes, exponential recency weighting, and what convergence costs."
sidebar:
  order: 4
---

## Learning objective

Be able to say precisely what makes a problem nonstationary, explain why the
sample average fails on one, and state what a constant step size buys and what
it gives up.

## Intuition

Everything so far has assumed the bandit is **stationary**: each $q_*(a)$ is
fixed for all time, so every reward you have ever collected is evidence about
the same world. Under that assumption averaging all of it is not just
reasonable, it is optimal — more data, better estimate, no downside.

A **nonstationary** problem is one where $q_*(a)$ itself changes as you play:

$$
q_*(a) \;\longrightarrow\; q_{*,t}(a)
$$

The best arm at pull 100 need not be the best arm at pull 1000 — not because
your estimate was wrong, but because the answer moved underneath it.

This is the common case, not the exotic one. Click-through rates shift as tastes
change. Server latencies move with load. An opponent adapts to you. A
recommender's users get bored. The standard demonstration in the literature
(Sutton & Barto, exercise 2.5) makes all ten arms take independent random walks:
every $q_*(a)$ gets a small Gaussian nudge on every step, so the identity of the
best arm slowly wanders.

## Why the sample average fails

Recall the incremental sample average from
[action-value methods](/Reinforcement-Learning/chapters/02-multi-armed-bandits/action-value-methods/):

$$
Q_{n+1} = Q_n + \frac{1}{n}\left(R_n - Q_n\right)
$$

Two properties that were virtues on a stationary problem become defects here.

**Every reward carries equal weight.** The reward from pull 1 counts exactly as
much as the reward from pull 999. When those two rewards were drawn from
different distributions, the estimate is an average over a world that no longer
exists.

**The step size shrinks toward zero.** By pull 1000 the update moves $Q$ by one
thousandth of the error. The estimator becomes *less* able to react to change
exactly as it accumulates more stale evidence. It converges — confidently, and
to the wrong number.

The second point is the sharper one. How fast the sample average can respond to
a change depends not on the change but on how much history preceded it. Suppose
an arm's true value jumps from 0 to 1. If the jump happens after 10 pulls, 100
further pulls bring the estimate to $100/110 \approx 0.91$. If the same jump
happens after 1000 pulls, the same 100 further pulls reach only
$100/1100 \approx 0.09$. Same change, same evidence, and the older agent has
barely noticed.

## Constant step size

Replace $1/n$ with a fixed $\alpha \in (0, 1]$:

$$
Q_{n+1} = Q_n + \alpha\left(R_n - Q_n\right)
$$

Expanding the recursion shows what changed:

$$
Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1-\alpha)^{n-i} R_i
$$

The weight on a reward decays geometrically with its age, and the weights sum to
1, so this is still a weighted average — an **exponential recency-weighted
average**. Old rewards fade out on their own, without anyone deciding when to
discard them.

The rate of forgetting is set by $\alpha$ alone. A reward's weight halves every

$$
\frac{\ln 2}{-\ln(1-\alpha)} \;\text{ steps}
$$

which is about 6.6 steps for $\alpha = 0.1$ and about 69 steps for
$\alpha = 0.01$. That is the number to reason with: $\alpha$ is a memory length
in disguise. Choose it to match how fast you think the problem drifts, not by
feel.

## Worked example

One arm, noiseless rewards so the arithmetic is checkable ($R = q_*$). Its true
value is 0 for the first ten pulls, then jumps to 1 and stays there. Both
estimators start at $Q_1 = 0$ and see identical rewards.

| Pull | $R$ | Sample average $1/n$ | Constant $\alpha = 0.1$ |
| --- | --- | --- | --- |
| 10 | 0 | 0.000 | 0.000 |
| 11 | 1 | 0.091 | 0.100 |
| 13 | 1 | 0.231 | 0.271 |
| 15 | 1 | 0.333 | 0.410 |
| 17 | 1 | 0.412 | 0.522 |
| 20 | 1 | 0.500 | 0.651 |

Both are wrong ten pulls after the change — the constant-$\alpha$ estimator is
simply wrong by less, and the gap widens. Push it further and they separate
completely: the constant-$\alpha$ estimate closes on 1.0 and stays there, while
the sample average keeps dragging ten zeros behind it forever, approaching 1.0
only as those ten are diluted by hundreds of ones.

Note also that the sample average would win outright if the value had *not*
changed. That is the trade, in one table.

## What convergence costs

The clean statement of the trade-off is the Robbins–Monro conditions. A sequence
of step sizes $\alpha_n(a)$ guarantees convergence to $q_*(a)$ with probability 1
when both hold:

$$
\sum_{n=1}^{\infty} \alpha_n(a) = \infty
\qquad\text{and}\qquad
\sum_{n=1}^{\infty} \alpha_n^2(a) < \infty
$$

The first says the steps stay large enough, in total, to overcome any starting
point and any run of bad luck. The second says they shrink fast enough that the
noise in the rewards eventually averages out instead of bouncing the estimate
around forever.

- $\alpha_n = 1/n$ satisfies **both**. It converges.
- Constant $\alpha$ satisfies the first and **fails the second**. It never
  converges: the estimate keeps fluctuating in response to the most recent
  rewards.

On a stationary problem that failure is a defect. On a nonstationary one it is
the entire point. An estimate that has stopped responding is an estimate that
cannot track anything, and convergence to a fixed number is the wrong goal when
there is no fixed number to converge to.

This is also why step-size sequences that satisfy both conditions are rare in
practice despite being the theoretically sound choice. They converge slowly, the
tuning is fiddly, and almost every problem anyone actually cares about is
nonstationary.

## Removing the initial bias

Constant $\alpha$ has one blemish. That leading $(1-\alpha)^n Q_1$ term decays
but never vanishes, so the estimate is permanently biased by whatever you
initialised it to — after 20 steps at $\alpha = 0.1$, about 12% of the estimate
is still $Q_1$. The sample average has no such bias after the first reward.

Exercise 2.7 gives a step size that keeps the recency weighting and drops the
bias. Track a running trace $\bar{o}_n$ and divide by it:

$$
\bar{o}_n = \bar{o}_{n-1} + \alpha\left(1 - \bar{o}_{n-1}\right), \quad \bar{o}_0 = 0
\qquad
\beta_n = \frac{\alpha}{\bar{o}_n}
$$

For $\alpha = 0.1$ this gives $\beta_1 = 1$, then 0.526, 0.369, 0.291, 0.244,
settling toward 0.1. The first update overwrites $Q_1$ completely — so no trace
of the initialisation survives — and later updates converge to ordinary constant
$\alpha$ behaviour.

## Common mistakes

**Assuming stationarity because nobody said otherwise.** The default in a
textbook is a fixed $q_*(a)$; the default in a deployed system is drift. If the
problem has users, competitors, or hardware in it, assume nonstationary.

**Choosing $\alpha$ by feel.** Convert it to a half-life first. "$\alpha = 0.1$"
means little; "this estimate forgets half of what it knows every seven steps"
can be checked against how fast the problem actually moves.

**Reading non-convergence as a bug.** A constant-$\alpha$ estimate that keeps
jittering is working as designed. Judge it by tracking error against the moving
$q_{*,t}(a)$, not by whether it settles.

**Fixing drift with more exploration alone.** Raising $\varepsilon$ makes the
agent re-sample arms, which is necessary — a stale arm never pulled is never
re-estimated — but a sample average will still fold those new rewards into an
average dominated by old ones. Exploration and step size solve different halves
of the problem, and a nonstationary bandit needs both.

## Personal takeaways

The step size is not a tuning knob for learning speed. It is a statement about
how long the past stays relevant, and $1/n$ is the specific claim that it stays
relevant forever.

The part worth carrying into later chapters: **full RL is nonstationary even
when the environment is not.** Once there is a policy being improved, the value
of an action depends on what the agent will do afterwards — and that keeps
changing as the policy changes. The target moves because the learner moved it.
This is why a constant $\alpha$ shows up nearly everywhere later in this book,
in TD learning and DQN and policy gradients alike, and not just in this one
corner of the bandit chapter.

## Where next

The initial value $Q_1$ has been a nuisance on this page — a bias to be removed.
[Optimistic initial
values](/Reinforcement-Learning/chapters/02-multi-armed-bandits/optimistic-initial-values/)
turns it into an exploration mechanism instead, and the exponential weighting
derived above is exactly what makes that work.

## References

- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., §2.5 and exercises 2.5, 2.7.
