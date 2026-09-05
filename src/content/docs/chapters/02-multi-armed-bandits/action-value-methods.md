---
title: Action-value methods
description: "Estimating the value of an action from experience, and updating that estimate incrementally."
sidebar:
  order: 2
---

## Learning objective

Be able to write down the sample-average estimate of $q_*(a)$, convert it to
incremental form, and explain why the incremental form is the one worth
memorising.

## Intuition

You cannot see $q_*(a)$. You can see rewards. So estimate the value of an action
by averaging the rewards you have actually received from it. Pull a machine ten
times, win four, and your estimate of its payout rate is 0.4.

The obvious method is the right one here. What matters is the *form* you write
it in.

## The sample average

$$
Q_t(a) = \frac{\text{sum of rewards taken from } a \text{ before } t}{\text{number of times } a \text{ was taken before } t}
    = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i = a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i = a}}
$$

By the law of large numbers $Q_t(a) \to q_*(a)$ as the denominator grows. If an
action has never been taken the estimate is undefined, so it is initialised to
some default $Q_1(a)$ — usually 0.

## The incremental form

Storing every reward to recompute the mean wastes memory that grows without
bound. Let $Q_n$ be the estimate after $n-1$ rewards. Then:

$$
Q_{n+1} = Q_n + \frac{1}{n}\left(R_n - Q_n\right)
$$

<div class="pseudocode">

**Derivation.** With $Q_{n+1} = \frac{1}{n}\sum_{i=1}^{n} R_i$,

$$
Q_{n+1} = \frac{1}{n}\left(R_n + \sum_{i=1}^{n-1} R_i\right)
        = \frac{1}{n}\left(R_n + (n-1) Q_n\right)
        = Q_n + \frac{1}{n}\left(R_n - Q_n\right)
$$

</div>

Constant memory, constant work per step, identical answer.

## The update pattern

That last line is worth reading closely, because the same shape appears in every
method in this book:

$$
\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \cdot \left[\text{Target} - \text{OldEstimate}\right]
$$

The bracketed quantity is the **error**. The update moves the estimate a fraction
of the way toward the target. TD learning, Q-learning, and gradient-based policy
updates are all this expression with a different target substituted in.

## Constant step size

Replacing $1/n$ with a constant $\alpha \in (0, 1]$ gives:

$$
Q_{n+1} = Q_n + \alpha\left(R_n - Q_n\right)
$$

Expanding the recursion shows what this does:

$$
Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1-\alpha)^{n-i} R_i
$$

The weight on a reward decays exponentially with how long ago it arrived. This is
an **exponential recency-weighted average**: it never fully converges, which is
exactly what you want when $q_*(a)$ drifts over time. For the stationary bandit
in this chapter, $1/n$ is the better choice; for anything non-stationary, a
constant $\alpha$ is.

## Common mistakes

**Recomputing the mean from stored rewards.** Correct, but the memory grows
forever, and the incremental form makes the connection to every later algorithm
visible.

**Using $1/n$ on a non-stationary problem.** The step size shrinks toward zero, so
the estimate eventually stops responding to change at all.

**Forgetting that ties need breaking.** With $Q_1(a) = 0$ for every $a$, the first
greedy choice is a $k$-way tie. Break it uniformly at random, or the first arm
wins by index order and the run looks broken.

## Personal takeaways

`error = target - estimate`, then step toward it. Writing the sample average in
that form once makes TD learning look like a small change of target rather than a
new idea.

## Where next

Estimates alone do not tell you when to act on them.
[Epsilon-greedy](/Reinforcement-Learning/chapters/02-multi-armed-bandits/epsilon-greedy/)
is the simplest rule for deciding when to trust $Q$ and when to look elsewhere.

## References

- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., §2.3–2.5.
