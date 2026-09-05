---
title: Getting started
description: "How this book is organised and how to read it."
---

## How to read this

Chapters are ordered by difficulty, not by history. Each one assumes the
chapters above it and nothing below it, so reading top to bottom always works.

| Chapter | What it adds |
| --- | --- |
| Foundations | The three historical threads that became modern RL. |
| Multi-armed bandits | Exploration vs. exploitation, with **no state**. |
| Markov decision processes | State, transitions, value functions, Bellman equations. |
| Dynamic programming | Exact solutions when the model is **known**. |
| Monte Carlo methods | First model-free learning, from complete episodes. |
| Temporal-difference learning | Bootstrapping: TD(0), SARSA, Q-learning. |
| n-step and eligibility traces | The axis connecting Monte Carlo and TD. |
| Planning and learning | Learned models and simulated experience. |
| Function approximation | Dropping the lookup table. |
| Deep Q-networks | Neural value approximation at scale. |
| Policy gradient methods | Optimising the policy directly. |
| Advanced policy optimization | Actor-critic, trust regions, PPO. |

## What a topic page looks like

Every topic page has six required sections — learning objective, intuition,
algorithm and equations, an interactive demo, what to observe, and personal
takeaways. Worked examples, common mistakes, experiments, and references appear
when they earn their place.

## The demos

Each demo runs as its own full-screen page and is also embedded in the topic
page it belongs to. They are deterministic: every demo takes a **seed**, and the
same seed always reproduces the same run. If something surprising happens, note
the seed and it can be replayed exactly.

The simulation code is separated from the drawing code, so the same core that
animates a demo also runs headlessly for the comparison charts.

## Running the site locally

```sh
npm install
npm run dev      # local preview
npm run check    # TypeScript and component diagnostics
npm run build    # production build, including internal-link validation
```

Node is pinned in `.nvmrc`. Pushing to `main` builds and deploys to GitHub Pages
automatically.
