---
title: "Returns and Episodes"
description: "Calculate finite and discounted returns, including backward calculations and infinite sequences."
sidebar:
  order: 4
---

A **return** combines the rewards after a particular time into one score. An **episode** is a sequence of interactions that ends, such as one game or one trip to Goal.

## Episodic and continuing tasks

| Task type | What happens | Examples |
| --- | --- | --- |
| Episodic | The task reaches a terminal state. | A maze attempt, a game, balancing a pole until it falls |
| Continuing | The task has no natural ending. | Thermostat control, ongoing server scheduling |

For an episode ending at time $T$, $S_T$ is terminal and $R_T$ is the last reward. The undiscounted return is:

$$
G_t = R_{t+1} + R_{t+2} + \cdots + R_T.
$$

If rewards continue forever, simply adding them may not produce a finite number. Discounting provides one useful objective for continuing tasks and can also be used in episodic tasks.

## Discounted return

With discount factor $\gamma$:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots.
$$

- $\gamma=0$: only the next reward counts.
- $0<\gamma<1$: later rewards count, but receive progressively less weight.
- $\gamma=1$: all rewards count equally, provided the return is well-defined, such as an episode with bounded rewards and a bounded number of steps.

For an infinite discounted sum, bounded rewards and $0\leq\gamma<1$ ensure a finite return. The next section explains how the infinite-sum notation also handles episodes.

The timing is easy to misread: $R_{t+1}$ has weight 1, $R_{t+2}$ has weight $\gamma$, and $R_{t+3}$ has weight $\gamma^2$. With $\gamma=0.5$, the **third upcoming reward** receives weight $0.5^2=0.25$.

## The gridworld route

For Start → Hall → Goal, the rewards are −1 and +5. At $\gamma=0.9$:

$$
G_0 = -1 + 0.9(5) = 3.5.
$$

The first reward is −1, but the return is 3.5. A longer route, Start → B → C → D → Goal, has return:

$$
G_0 = -1 - 0.9 - 0.9^2 + 0.9^3(5) = 0.935.
$$

Both routes reach the same goal. The longer one pays more movement costs and receives the positive reward later.

## Compute returns backward

Separate the next reward from the remaining return:

$$
G_t = R_{t+1} + \gamma G_{t+1}.
$$

Suppose $\gamma=0.5$, the rewards are $R_1=-1$, $R_2=2$, $R_3=6$, $R_4=3$, and $R_5=2$, and the episode ends at $T=5$. Start with terminal return $G_5=0$ and work backward:

| Time $t$ | Calculation | Return $G_t$ |
| --- | --- | --- |
| 5 | No rewards remain. | $0$ |
| 4 | $2 + 0.5(0)$ | $2$ |
| 3 | $3 + 0.5(2)$ | $4$ |
| 2 | $6 + 0.5(4)$ | $8$ |
| 1 | $2 + 0.5(8)$ | $6$ |
| 0 | $-1 + 0.5(6)$ | $2$ |

This recursion avoids repeatedly summing the same future rewards. It also provides the idea behind the Bellman equations in section 3.5.

## An infinite reward sequence

Suppose $R_1=2$, every reward after that equals 7, and $\gamma=0.9$. The geometric-series identity $\sum_{k=0}^{\infty}x^k=1/(1-x)$ applies for $|x|<1$:

$$
G_0 = 2 + 0.9(7) + 0.9^2(7) + \cdots = 2 + \frac{7(0.9)}{1-0.9} = 65.
$$

One step later, the return includes only the sequence of sevens:

$$
G_1 = 7 + 0.9(7) + \cdots = \frac{7}{1-0.9} = 70.
$$

The recursion checks the result: $G_0=2+0.9(70)=65$.

## Try it

For the two-step gridworld route, change $\gamma$ to 0.5. Its return becomes $-1+0.5(5)=1.5$. Then set $\gamma=0$: the future goal reward disappears from the objective, leaving return −1.

## Reference

Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., chapter 3, section 3.3. The explanations and small worked examples here are written for these notes.

---

[Previous: 3.2 Goals and Rewards](/Reinforcement-Learning/chapters/03-markov-decision-processes/02-goals-and-rewards/) · [Next: 3.4 Unified Notation for Episodic and Continuing Tasks](/Reinforcement-Learning/chapters/03-markov-decision-processes/04-unified-notation/)
