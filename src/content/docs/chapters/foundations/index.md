---
title: Foundations
description: "The vocabulary of reinforcement learning — agent, environment, reward, policy, and return — before any equations that need solving."
sidebar:
  order: 1
---

## Learning objective

Be able to state, for any problem you are handed, what the agent is, what the
environment is, what an action is, and what the reward signal is — and to say
why the problem is reinforcement learning rather than supervised learning.

## Intuition

Supervised learning is given the right answer for every example. Reinforcement
learning is not. It is given a **number** after it acts, and that number does not
say what the best action would have been. It says only how good the action it
took turned out to be.

That single change causes almost every difficulty in the field:

- **You must explore to learn.** The only way to find out what an untried action
  is worth is to try it, which costs you whatever the known-good action would
  have paid.
- **Feedback can be delayed.** A move in chess is punished twenty moves later.
- **Your data depends on your policy.** Act differently and you see a different
  slice of the world, so the training distribution shifts as you learn.

## The loop

At each step $t$ the agent observes a state $S_t$, takes an action $A_t$, and
receives a reward $R_{t+1}$ along with the next state $S_{t+1}$.

$$
S_0 \xrightarrow{A_0} R_1, S_1 \xrightarrow{A_1} R_2, S_2 \xrightarrow{A_2} \cdots
$$

That is the entire interface. Everything else is machinery for choosing $A_t$
well.

## The pieces

**Agent** — the learner and decision maker. The only thing you control.

**Environment** — everything else, including parts of the "body" the agent
cannot change arbitrarily. The boundary is drawn at the limit of *absolute
control*, not at the skin.

**Action** $A_t \in \mathcal{A}$ — what the agent chooses.

**State** $S_t \in \mathcal{S}$ — what the agent knows when it chooses.

**Reward** $R_{t+1} \in \mathbb{R}$ — a single scalar, delivered by the
environment, that defines the goal. Not a hint about how to reach it.

**Policy** $\pi$ — the agent's behaviour: a mapping from states to actions, or to
a distribution over actions, written $\pi(a \mid s)$.

**Return** $G_t$ — the total reward from step $t$ onward. This, not the immediate
reward, is what the agent maximises:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

The **discount factor** $\gamma \in [0, 1]$ sets how much future reward is worth
now. At $\gamma = 0$ the agent is myopic and maximises only $R_{t+1}$. As
$\gamma \to 1$ it becomes farsighted. Discounting also keeps $G_t$ finite when
the task never ends.

## The reward hypothesis

> All of what we mean by goals and purposes can be well thought of as the
> maximisation of the expected value of the cumulative sum of a received scalar
> signal.

This is an assumption, not a theorem, and it is where applied RL usually goes
wrong. The reward must say **what** you want achieved, never **how**. Rewarding a
chess agent for capturing pieces produces an agent that captures pieces and
loses games.

## Common mistakes

**Rewarding subgoals.** Any bonus for intermediate progress is an invitation to
farm the bonus instead of finishing the task.

**Putting knowledge in the reward.** Prior knowledge belongs in the initial
policy or the initial value estimates, not in the reward function.

**Confusing reward with return.** The agent maximises $G_t$. A move with low
immediate reward and high return is a good move.

**Drawing the agent–environment boundary too wide.** If the agent cannot change
something at will, that thing is environment — even if it lives inside the robot.

## Personal takeaways

The evaluative-vs-instructive distinction is the whole subject in one line. A
supervised label says *"the answer was 7."* A reward says *"that was worth 0.3."*
Every algorithm in this book exists because the second sentence is so much
weaker than the first.

## Where next

The next chapter strips the problem down as far as it goes: one state, several
actions, immediate reward.
[Multi-armed bandits](/Reinforcement-Learning/chapters/multi-armed-bandits/) is
where exploration can be studied on its own, with nothing else in the way.

## References

- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., ch. 1 & 3.
