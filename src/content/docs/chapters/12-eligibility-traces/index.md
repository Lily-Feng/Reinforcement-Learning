---
title: Eligibility Traces
description: "Understand eligibility traces through contrasts: forward and backward views, TD(λ), Sarsa(λ), Dutch traces, and computational trade-offs."
sidebar:
  order: 1
---

## Learning objective: connect credit across time

**Eligibility traces are a fading trail of breadcrumbs: the current TD error follows that trail to adjust predictions associated with previously visited states or features.**

This chapter uses **contrastive understanding**: compare perspectives first, then algorithms, memory rules, and implementation costs, before exploring possible extensions to large models. Every equation answers the same question: **Having discovered a prediction error now, which earlier predictions should change, and by how much?**

By the end, you should be able to compute a TD($\lambda$) update by hand, explain accumulating, replacing, and Dutch traces, and identify when true online methods matter.

[Download the main Jupyter notebook: complete notes and runnable experiments](/Reinforcement-Learning/notebooks/ch12-eligibility-traces.ipynb). Only the Python 3 standard library is required; run all cells in order.

Prerequisites: [TD learning](/Reinforcement-Learning/chapters/06-temporal-difference-learning/), [n-step returns](/Reinforcement-Learning/chapters/07-n-step-and-eligibility-traces/), [function approximation](/Reinforcement-Learning/chapters/09-function-approximation/), and [semi-gradient Sarsa](/Reinforcement-Learning/chapters/10-on-policy-control-approximation/). This chapter builds directly on Chapter 7's multi-step targets and Chapter 10's control problems.

| Module | Core contrast | Guiding question |
| --- | --- | --- |
| 1: Perspectives | Forward vs. backward | Wait for the future, or remember the past? |
| 2: Algorithms | TD(0), MC, n-step, TD($\lambda$), Sarsa($\lambda$) | Which targets do we use, and which predictions change? |
| 3: Traces | Accumulating, replacing, Dutch | How should memory respond to repeated visits? |
| 4: Implementation | Full targets, truncation, sparse tables, dense networks | How do we trade off delay, computation, and memory? |
| 5: Extensions | Full-parameter vs. local or approximate tracking | How much memory can a very large model afford? |

## Module 1: forward targets and backward breadcrumbs

### Forward view vs. backward view

Consider the trajectory $A \to B \to C \to \text{terminal}$, with reward $1$ only on the final transition.

| View | Perspective | Learning mechanism | Timing |
| --- | --- | --- | --- |
| Forward | Stand at $A$ and look at subsequent rewards | Construct a multi-step target for $A$, then update $A$ | A full return waits for termination; n-step targets wait n steps; truncated targets can be computed earlier |
| Backward | Stand at the current transition and look into the past | Retain traces of past features and update them using the current TD error | Update every step without storing the complete trajectory |

The breadcrumbs compress history into a vector. They do not preserve the complete sequence of visits; after repeated visits to the same state, the original trajectory generally cannot be reconstructed from the trace.

Writing state values as $V(S_t)$, the one-step TD error is:

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t).
$$

**The signal sent backward is $\delta_t$, which includes both reward and prediction changes.** A bootstrapped estimate can produce an error even when the immediate reward is zero. A negative error lowers earlier predictions. The breadcrumb metaphor therefore describes correcting predictions along a trail, with either positive or negative adjustments.

### The backward update at a glance

For tabular accumulating traces, initialize $e_{-1}(s)=0$ at the start of each episode:

$$
e_t(s)=\gamma\lambda e_{t-1}(s)+\mathbb{1}\{S_t=s\},
\qquad
V_{t+1}(s)=V_t(s)+\alpha\delta_t e_t(s).
$$

First decay the old traces, then increase the current state's eligibility, and finally distribute the error across states with nonzero traces. Define the terminal state's value as $0$ and clear traces before the next episode.

- $\alpha$: how large a correction to make.
- $\gamma$: how the task discounts future rewards.
- $\lambda$: how multi-step targets are mixed; in the backward implementation, it also controls the decay of past eligibility.
- $\gamma\lambda$: the fraction of a visit's trace retained after each step.

When $0<\gamma\lambda<1$, a single visit contributes $(\gamma\lambda)^k$ after $k$ steps. $\lambda$ changes the learning mechanism; $\gamma$ participates in defining the task's return.

## Module 2: connecting traces to earlier algorithms

### TD(0) vs. Monte Carlo vs. TD($\lambda$)

| Method | Target or mechanism | When does it learn? | Main trade-off |
| --- | --- | --- | --- |
| TD(0) | One reward plus the next state's estimated value | Every step | Substantial bootstrapping; targets often have lower variance but depend on current estimates |
| Episodic MC | The complete return through termination | After the episode | No bootstrapping; returns may have higher variance, and the usual form requires termination |
| TD($\lambda$) | Propagate one-step TD errors through traces | Every step | Trade off different multi-step targets; tune $\lambda$ together with $\alpha$ |

Tabular TD(0) updates only the current state. With shared parameters, one update may also change predictions at other states. The analogy of “thanking only the last waiter” describes the location of a tabular update.

$\lambda=0$ gives TD(0). At $\lambda=1$, the **complete forward target** equals the MC return in a finite episode. Conventional online TD(1), however, changes estimates during the episode, so its parameter updates need not match MC updates performed at episode end. A continuing task without termination cannot wait for a complete episodic MC return; TD(1) traces can still decay when $\gamma<1$.

### Fixed n-step targets vs. the $\lambda$-return

To make the weighting clear, first hold the bootstrap estimates fixed at $\bar V$. For $t+n<T$:

$$
G_{t:t+n}=\sum_{k=1}^{n}\gamma^{k-1}R_{t+k}
+\gamma^n\bar V(S_{t+n}).
$$

There is no bootstrapping beyond termination. Let $G_t$ denote the complete return. The finite-episode $\lambda$-return is:

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_{t:t+n}
+\lambda^{T-t-1}G_t.
$$

Earlier n-step targets receive geometric weights. **All remaining weight goes to the terminal return**, making the weights sum to $1$. With only one step remaining, the target is simply $G_t$, including at the $\lambda=0$ boundary.

| Fixed n-step | $\lambda$-return |
| --- | --- |
| Select one target horizon per update | Mix targets from multiple horizons per update |
| n=3 still uses all three observed rewards | Weight return targets of different lengths |
| Delay is tied to n and is easy to control | Full forward computation waits for termination; traces enable step-by-step learning |

A fixed n is a deliberate choice about bias, variance, and delay. Mixing horizons with $\lambda$ does not guarantee better performance.

**Worked example:** For three rewards $(0,0,1)$, with $\gamma=1$, $\bar V=0$, and $\lambda=0.8$, the targets from $A$ are $(0,0,1)$ and their weights are $(0.2,0.16,0.64)$. Thus $G_A^\lambda=0.64$.

### Sarsa vs. Sarsa($\lambda$)

Use state-action features $\mathbf{x}(S_t,A_t)$ to apply the same credit-assignment idea to control:

$$
\delta_t=R_{t+1}+\gamma\hat q(S_{t+1},A_{t+1},\mathbf w_t)
-\hat q(S_t,A_t,\mathbf w_t),
$$

$$
\mathbf e_t=\gamma\lambda\mathbf e_{t-1}
+\nabla_{\mathbf w}\hat q(S_t,A_t,\mathbf w_t),
\qquad
\mathbf w_{t+1}=\mathbf w_t+\alpha\delta_t\mathbf e_t.
$$

For a terminal transition, the next action value is $0$. Otherwise, select $A_{t+1}$ from the current behavior policy and use it to compute the error. This is **on-policy Sarsa**.

One-step Sarsa updates along the current state-action gradient. Sarsa($\lambda$) also updates along traces left by past gradients. Along a tabular trajectory without repeated visits, more recent actions generally receive larger adjustments from the same error. Repeated visits, shared features, and negative errors complicate the claim that “later actions receive more reward.” Faster propagation does not guarantee faster convergence to an optimal policy under every parameter setting.

## Module 3: three ways to remember repeated visits

### Accumulating vs. replacing

| Trace | Currently visited component | Intuition | Scope |
| --- | --- | --- | --- |
| Accumulating | Add 1 after decay | Repeated visits accumulate eligibility | Standard TD($\lambda$); with linear features, add the feature vector |
| Replacing | Set to 1 after decay | A visit refreshes that state's eligibility | Tabular state traces here; active binary features can be replaced componentwise |
| Dutch | Adjust using the step size and overlap between old traces and current features | Maintain a backward implementation of the online forward target | Requires a matching weight correction |

Accumulating traces do not always grow without bound. If a state is visited at every step and $\rho=\gamma\lambda<1$, then $e_t=1+\rho+\cdots+\rho^t\leq1/(1-\rho)$. Unbounded growth can occur when, for example, $\rho=1$ and visits continue indefinitely. Large traces combined with an excessive step size can cause instability.

Replacing a state's trace leaves other states' history intact. Replacing traces are a distinct mathematical rule with their own analysis. For action values, replacing traces also require deciding how to handle other actions at the same state; the state-trace description above does not settle that choice.

### Conventional vs. true online TD($\lambda$)

For linear prediction $v_t=\mathbf w_t^\top\mathbf x_t$, conventional accumulating traces use $\mathbf e_t=\gamma\lambda\mathbf e_{t-1}+\mathbf x_t$. With frozen weights, forward and backward views can produce exactly matching total updates. During online learning, changing estimates break this simple correspondence, and the discrepancy need not be small.

True online methods match an **online forward view that allows predictions to change**. For linear function approximation, including the tabular special case, the correspondence holds at every step. This result has a specific scope: it does not establish unconditional equivalence for arbitrary neural networks or forward targets. [Original paper: True Online TD(λ)](https://proceedings.mlr.press/v32/seijen14.html)

The Dutch trace is:

$$
\mathbf e_t=\gamma\lambda\mathbf e_{t-1}
+\left(1-\alpha\gamma\lambda\mathbf e_{t-1}^{\top}\mathbf x_t\right)\mathbf x_t.
$$

Its matching weight update is:

$$
\mathbf w_{t+1}=\mathbf w_t
+\alpha(\delta_t+v_t-v_{\mathrm{old}})\mathbf e_t
-\alpha(v_t-v_{\mathrm{old}})\mathbf x_t.
$$

Initialize $\mathbf e=0$ and $v_{\mathrm{old}}=0$ each episode. At each step, compute $v_t$ and $v_{t+1}$ using the weights before the update. After updating, set $v_{\mathrm{old}}\leftarrow v_{t+1}$. At the next step, the saved prediction and the new prediction can differ; the correction accounts for that difference.

**Both pieces are essential: the Dutch trace and the prediction correction.** Like conventional dense TD($\lambda$), this takes $O(d)$ work per step, with additional constant-factor overhead. See the authors' [True Online Temporal-Difference Learning](https://arxiv.org/abs/1512.04087) for further derivations and experiments.

## Hands-on experiments: turn breadcrumbs into numbers

These cells also appear in the downloadable notebook. Run them in order. All examples use the standard library so that the update logic stays visible.

### Experiment 1: which states change when a reward arrives?

```python
def td_lambda(states, rewards, lam, alpha=0.1, gamma=1.0,
              replacing=False):
    # states contains nonterminal states; the final transition terminates.
    assert len(states) == len(rewards) and states
    values = {s: 0.0 for s in states}
    traces = dict.fromkeys(values, 0.0)
    history = []
    for t, (state, reward) in enumerate(zip(states, rewards)):
        next_v = values[states[t + 1]] if t + 1 < len(states) else 0.0
        delta = reward + gamma * next_v - values[state]
        for s in traces:
            traces[s] *= gamma * lam
        traces[state] = 1.0 if replacing else traces[state] + 1.0
        for s in values:
            values[s] += alpha * delta * traces[s]
        history.append((delta, traces.copy(), values.copy()))
    return values, history

for lam in (0.0, 0.8, 1.0):
    values, _ = td_lambda(['A', 'B', 'C'], [0, 0, 1], lam)
    print(lam, {s: round(v, 4) for s, v in values.items()})

values, history = td_lambda(['A', 'B', 'C'], [0, 0, 1], 0.8)
assert all(abs(values[s] - v) < 1e-12
           for s, v in zip(['A', 'B', 'C'], [0.064, 0.08, 0.1]))
```

The expected final values $(V(A),V(B),V(C))$ are $(0,0,0.1)$, $(0.064,0.08,0.1)$, and $(0.1,0.1,0.1)$, respectively. With zero initialization, the first two errors are zero and the last error is $1$. For $\lambda=0.8$, the traces at that point are $(0.64,0.8,1)$.

**What to observe:** Set $\gamma=0.9$ to reduce the earlier states' adjustments further. Then change the rewards to $(0,0,-1)$ and watch the same mechanism lower their values.

### Experiment 2: accumulate or refresh on a repeated visit?

```python
for replacing in (False, True):
    values, history = td_lambda(['A', 'B', 'A'], [0, 0, 1], 0.8,
                                replacing=replacing)
    final_trace = history[-1][1]
    print('replacing' if replacing else 'accumulating', final_trace, values)
    expected = 1.0 if replacing else 1.64
    assert abs(final_trace['A'] - expected) < 1e-12
```

The accumulating trace for $A$ is $1+0.8^2=1.64$; the replacing trace is $1$. The trace for $B$ is $0.8$ under both rules. **Replacing the current state's trace preserves other states' traces.**

### Experiment 3: sum the same target forward and backward

Freeze all bootstrap estimates and compare total updates over one episode. This experiment uses a different update schedule from online learning with changing weights.

```python
states, rewards = ['A', 'B', 'A'], [0.2, -0.1, 1.0]
vbar = {'A': 0.3, 'B': -0.2}
gamma, lam = 0.9, 0.8
T = len(states)

def n_return(t, n):
    end = min(t + n, T)
    g = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
    if end < T:
        g += gamma ** (end - t) * vbar[states[end]]
    return g

forward = dict.fromkeys(vbar, 0.0)
backward = dict.fromkeys(vbar, 0.0)
e = dict.fromkeys(vbar, 0.0)
for t, s in enumerate(states):
    horizon = T - t
    target = sum((1 - lam) * lam ** (n - 1) * n_return(t, n)
                 for n in range(1, horizon))
    target += lam ** (horizon - 1) * n_return(t, horizon)
    forward[s] += target - vbar[s]
    next_v = vbar[states[t + 1]] if t + 1 < T else 0.0
    delta = rewards[t] + gamma * next_v - vbar[s]
    for key in e:
        e[key] *= gamma * lam
    e[s] += 1.0
    for key in e:
        backward[key] += delta * e[key]

print('forward:', forward, 'backward:', backward)
assert all(abs(forward[s] - backward[s]) < 1e-12 for s in vbar)
```

The accumulated increments have not yet been multiplied by $\alpha$. Freezing the estimates lets the future TD errors telescope. If one side updates its estimates immediately, the assertion no longer has the same guarantee.

### Experiment 4: a minimal linear true online implementation

```python
def true_online_td(features, rewards, lam, alpha=0.1, gamma=1.0):
    # Fixed feature vectors; no terminal feature is supplied.
    assert features and len(features) == len(rewards)
    d = len(features[0])
    w, e, v_old = [0.0] * d, [0.0] * d, 0.0
    dot = lambda a, b: sum(ai * bi for ai, bi in zip(a, b))
    for t, x in enumerate(features):
        x_next = features[t + 1] if t + 1 < len(features) else [0.0] * d
        v, v_next = dot(w, x), dot(w, x_next)
        delta = rewards[t] + gamma * v_next - v
        overlap = dot(e, x)  # Must use the old trace.
        e = [gamma * lam * ei + (1 - alpha * gamma * lam * overlap) * xi
             for ei, xi in zip(e, x)]
        correction = v - v_old
        w = [wi + alpha * (delta + correction) * ei - alpha * correction * xi
             for wi, ei, xi in zip(w, e, x)]
        v_old = v_next  # Prediction made BEFORE the weight update.
    return w

one_hot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
w = true_online_td(one_hot, [0, 0, 1], 0.8)
assert all(abs(a - b) < 1e-12 for a, b in zip(w, [0.064, 0.08, 0.1]))

# lambda=0 must reduce to ordinary online TD(0), even with revisits.
revisits = [[1, 0], [0, 1], [1, 0]]
w0 = true_online_td(revisits, [0.2, -0.1, 1.0], 0.0)
v0, _ = td_lambda(['A', 'B', 'A'], [0.2, -0.1, 1.0], 0.0)
assert all(abs(a - b) < 1e-12 for a, b in zip(w0, [v0['A'], v0['B']]))
print('True online weights:', w, 'TD(0) boundary:', w0)
```

These checks cover a simple trajectory and the $\lambda=0$ boundary; **they do not prove the full online forward-view equivalence**. Try $\lambda=0.8$ with `revisits` and compare against conventional accumulating traces. Nonzero intermediate rewards and repeated features make the online correction more visible.

## Module 4: target fidelity and implementation resources

### Complete vs. truncated $\lambda$-returns

The complete forward target depends on the remainder of the episode, so it waits for termination. Having the complete target does not guarantee the best empirical performance. Truncate the forward horizon at $h>t$, before termination:

$$
G_{t:h}^\lambda=(1-\lambda)\sum_{n=1}^{h-t-1}\lambda^{n-1}G_{t:t+n}
+\lambda^{h-t-1}G_{t:h}.
$$

The last target bootstraps at $S_h$ and absorbs all remaining weight. If termination occurs first, use the actual terminal return. Choosing $h=t+n$ gives a delay of at most n steps. This is a **truncated forward method**.

Distinguish three operations: truncating the return horizon, dropping small trace components, and truncating backpropagation through time (TBPTT). All can save resources, but they modify different objects.

### Tables vs. linear features vs. neural networks

| Implementation | Work per step | Memory and practical trade-offs |
| --- | --- | --- |
| Dense table | A naive implementation visits every stored trace and value: $O(\lvert S\rvert)$ | Wasteful when there are many states |
| Active-trace dictionary | Maintain K components with significant remaining eligibility: roughly $O(K)$ | Dropping components below a threshold introduces approximation; K need not stay small |
| d-dimensional linear approximation | Dense trace and weight updates take $O(d)$ | One extra d-dimensional trace; sparse current features do not ensure sparse historical traces |
| Neural network with P parameters | Compute gradients, maintain parameter traces, and update weights | Roughly P extra trace values; GPUs offer parallelism, but bandwidth, backpropagation, and sequential dependencies still cost time |

When $\gamma\lambda<1$ and revisits are limited, a threshold can help control the active set. When $\gamma\lambda=1$, old traces may not decay. There is no unconditional guarantee that traces cost only a few times as much as TD(0).

A neural network can use the semi-gradient extension $\mathbf e_t=\gamma\lambda\mathbf e_{t-1}+\nabla_{\mathbf w}\hat v(S_t,\mathbf w_t)$. Its gradient directions change with the parameters, however, so the linear true online theorem above does not transfer directly.

One extra trace vector does not imply that **total training memory** exactly doubles. Optimizer states, gradients, activations, mixed-precision copies, and distributed communication also consume resources. Eligibility traces can accumulate gradients online across time, but do not automatically recover the gradient through an entire differentiable trajectory.

## Module 5: eligibility traces meet 100-billion-parameter models

This module presents **research and engineering possibilities** motivated by classical algorithms. Chapter 12 does not establish them as large-model training recipes, and they are not direct replacements for existing language-model training methods.

Start with the arithmetic: for $10^{11}$ parameters, a complete trace vector with 4 bytes per entry needs about **400 GB in decimal units**. Even 2-byte entries require about 200 GB, before model weights, optimizer states, and activations.

| Direction | What is stored? | Benefit and limitation |
| --- | --- | --- |
| LoRA-style local tracking | Freeze the backbone and track only trainable adapter parameters | Trace memory scales with adapter size; credit can only be assigned through the trainable subspace |
| Sparse or Top-k tracking | Retain selected parameter or trace components and their indices | Less storage; a parameter with a small current gradient may still have important historical eligibility, and selection has a cost |
| Low-precision tracking | Approximate traces with 8-bit or 4-bit values plus scales and other metadata | Less storage; decay underflow, rounding, and accumulated errors may change learning behavior |

“Top 1% of neurons” and “Top 1% of parameter gradients” refer to different selections; define the tracked object first. All three approaches can change the original update and cannot automatically claim the linear true online equivalence.

**Before evaluating a design, ask:** Which value prediction is being tracked? What are the rewards and time steps? Which parameters are frozen? Does each parallel trajectory need its own trace? Compare using the same data, parameter budget, and actual runtime to determine whether the savings are worthwhile.

## Review checks and personal takeaway

Cover the answers and explain each contrast in one sentence.

| Question | Check your answer |
| --- | --- |
| Can earlier states change without an immediate reward? | Yes. Bootstrapped predictions can produce a nonzero TD error. |
| Why do traces decay by $\gamma\lambda$? | Discounting and horizon-mixture weights both appear when multi-step returns are expanded. |
| Does $\lambda=1$ make conventional online TD(1) identical to episode-end MC? | Matching target endpoints does not imply matching online weight sequences. |
| Why is a Dutch trace alone insufficient? | The weight correction must also account for changing online predictions. |
| Does a replacing trace erase the entire history? | No. Here it refreshes only the currently visited component. |
| Why can a neural network's extra trace be expensive? | Parameter count, bandwidth, backpropagation, and the number of parallel trajectories affect the actual cost. |

**My one-sentence takeaway:** The forward view defines the target to learn, the backward view distributes current errors across history, $\lambda$ connects time scales, trace rules determine how repeated visits are remembered, and true online methods make the online target and implementation agree step by step under explicit conditions.

For further experiments, hold $\gamma$ fixed and vary both $\lambda$ and $\alpha$. Record learning speed, final error, and runtime. A single trajectory is insufficient to rank methods reliably.

## References and reading path

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 12: read the λ-return, TD(λ), truncated and online forward algorithms, true online TD, Sarsa(λ), and implementation issues. Later sections also cover variable λ and off-policy traces; these main notes focus on the five modules above. [Authors' textbook page](http://incompleteideas.net/book/the-book-2nd.html)
- van Seijen & Sutton (2014), [True Online TD(λ)](https://proceedings.mlr.press/v32/seijen14.html): the original derivation of the online forward view and Dutch traces.
- van Seijen et al. (2016), [True Online Temporal-Difference Learning](https://arxiv.org/abs/1512.04087): expanded theory, experiments with linear representations, and discussion of nonlinear approximation.
