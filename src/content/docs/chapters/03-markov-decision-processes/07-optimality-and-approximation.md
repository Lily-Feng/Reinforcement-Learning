---
title: "Optimality and Approximation"
description: "Understand why optimality is a useful target and why practical agents use estimates and approximation."
sidebar:
  order: 8
---

The Bellman optimality equations specify what an ideal solution must satisfy. They do not make that solution cheap to obtain. A practical agent usually has limited information, memory, computation, and experience.

## Why exact optimality can be difficult

| Requirement for an exact model-based solution | What can make it difficult |
| --- | --- |
| Know transition and reward probabilities. | The model may be unknown and must be learned from interaction. |
| Represent values for all relevant states and actions. | Images, positions, and histories can create enormous or continuous state spaces. |
| Solve the optimality equations. | Large sums and repeated updates can exceed the available time. |
| Choose a state representation that captures the dynamics. | Important information may be hidden or omitted. |

Even a known finite model can be too large for a table of every state–action value. The existence of an optimal policy is a mathematical result; finding it with available resources is a computational task.

## Approximate values can still produce useful decisions

An agent acts by comparing action values. It may not need every estimate to be numerically exact to select the best action.

| Action in a state | True action value | Estimated action value |
| --- | --- | --- |
| Right | $3.5$ | $3.3$ |
| Down | $2.0$ | $2.2$ |

The estimates are imperfect, but choosing the larger estimate still selects Right. If errors reverse the ranking, the decision can become worse. Small errors matter more when actions have nearly equal values.

This does not mean value accuracy is irrelevant. It means we should also examine the resulting behavior and achieved return, rather than judging a policy only by numerical prediction error.

## Where approximations enter

- **Sample outcomes:** estimate expectations from experienced transitions instead of summing over a known model.
- **Stop iterative computation early:** use a chosen tolerance when exact convergence is unnecessary or too costly.
- **Share information across states:** use features or a function approximator, such as a neural network, instead of a separate table entry for every state.
- **Concentrate experience:** improve decisions in states the agent encounters, while retaining enough exploration to discover useful alternatives.

A table can represent every value in a small finite MDP exactly, yet its learned entries may still be inaccurate because of limited data. Function approximation adds a different limitation: the chosen representation may be unable to express the true value function exactly.

## Approximation does not fix a poorly specified task

An agent can optimize its model perfectly and still behave poorly if the reward encodes the wrong goal. Likewise, treating observations as a Markov state can fail when they omit information that changes future outcomes.

For a recommendation system, accurate click predictions do not by themselves establish good long-term recommendations. The reward, state, and evaluation must reflect the intended objective.

## What to check in a practical solution

1. Does the reward reflect the task we want to solve?
2. Does the state include information needed for good decisions?
3. Are value estimates useful for choosing actions?
4. Does the resulting policy earn good return across repeated evaluations?
5. Is the method affordable in data, memory, and computation?

The answers determine whether the approximation is useful for the problem. Approximate methods do not automatically inherit the guarantees of exact tabular algorithms; their assumptions need to be checked separately.

## How the rest of the book builds on this chapter

[Dynamic programming](/Reinforcement-Learning/chapters/04-dynamic-programming/) starts with a known model. [Monte Carlo methods](/Reinforcement-Learning/chapters/05-monte-carlo-methods/) estimate values from complete episodes. [Temporal-difference learning](/Reinforcement-Learning/chapters/06-temporal-difference-learning/) updates estimates from experience before an episode ends. [Function approximation](/Reinforcement-Learning/chapters/09-function-approximation/) addresses value representations that must generalize across states.

Across these approaches, the basic objects stay the same: **the MDP defines the rules, the policy chooses actions, and value measures expected return**.

## Reference

Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., chapter 3, section 3.7. The explanations and small worked examples here are written for these notes.

---

[Previous: 3.6 Optimal Policies and Optimal Value Functions](/Reinforcement-Learning/chapters/03-markov-decision-processes/06-optimal-policies-and-value-functions/) · [Chapter overview](/Reinforcement-Learning/chapters/03-markov-decision-processes/)
