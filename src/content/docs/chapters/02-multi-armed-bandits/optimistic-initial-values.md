---
title: Optimistic initial values
description: "Start every estimate absurdly high and a purely greedy agent explores on its own — systematically, without randomness, and only once."
sidebar:
  order: 5
---

## Learning objective

Be able to explain how an initial value can substitute for an exploration rule,
predict how much exploration a given $Q_1$ and $\alpha$ buy, and say precisely
why the technique stops working on a nonstationary problem.

## Intuition

Every method so far has depended on where the estimates start, and that
dependence has been a nuisance — a bias to be minimised or removed. Optimistic
initial values turn it into the exploration mechanism itself.

Set every estimate higher than any reward the problem can plausibly deliver. On
the standard 10-armed testbed, where $q_*(a) \sim \mathcal{N}(0,1)$, take:

$$
Q_1(a) = +5 \quad \text{for every } a
$$

Now run a **purely greedy** agent — $\varepsilon = 0$, no randomness anywhere. It
explores anyway.

## The algorithm

It is [the simple bandit
algorithm](/Reinforcement-Learning/chapters/02-multi-armed-bandits/) with two
lines changed:

<div class="pseudocode">

**Optimistic greedy**

Initialise, for $a = 1$ to $k$: &nbsp; $Q(a) \leftarrow +5$ &nbsp; *(was 0)*

**Loop forever:**

$\quad A \leftarrow \arg\max_a Q(a)$ &nbsp; (breaking ties randomly) &nbsp; *(no $\varepsilon$ branch)*

$\quad R \leftarrow \mathrm{bandit}(A)$

$\quad Q(A) \leftarrow Q(A) + \alpha\left[R - Q(A)\right]$

</div>

No exploration parameter appears in it. The exploration is a consequence of the
initialisation.

## Exploration by disappointment

The agent is greedy, so it pulls whatever looks best. Every arm looks equally,
absurdly good, so it picks one arbitrarily — and the reward comes back around 0,
nowhere near the $+5$ it expected. With $\alpha = 0.1$:

$$
Q(A) \leftarrow 5 + 0.1\,(0 - 5) = 4.5
$$

That arm now ranks *below* the nine untried arms still sitting at 5. So the
greedy choice is a different arm. Same disappointment, same demotion. The agent
works through all ten arms in the first ten steps, then starts a second pass over
arms now at 4.5, and a third, and so on.

The whole time it believes it is exploiting. It never chooses to explore; it is
simply always disappointed by whatever it just tried, and the arm it has *not*
tried recently always looks better. Reality underperforming expectations is what
drives the search, and the search stops when the expectations become accurate.

Note which arms sink slowest: an arm returning higher rewards loses less on each
update, so the genuinely good arms stay near the top of the ranking while the bad
ones fall away. The sweeping is not blind — it is a sweep that gradually sorts.

## The step size decides how much exploration you get

This is the part that is easy to miss, and it is why the technique is usually
paired with a constant $\alpha$ rather than the sample average.

**With sample averages ($1/n$):** the first update is

$$
Q_2 = Q_1 + \tfrac{1}{1}\left(R_1 - Q_1\right) = R_1
$$

The initial value is *erased outright* by a single pull. You get exactly one
forced sweep through the $k$ arms and then the optimism is gone completely.

**With constant $\alpha$:** the estimate descends gradually — $5 \to 4.5 \to
4.05 \to 3.65$ — so an arm must disappoint repeatedly before it drops below its
rivals. The agent cycles through all the arms many times over. This is where the
real exploration comes from.

The same fact read from
[the nonstationary page](/Reinforcement-Learning/chapters/02-multi-armed-bandits/nonstationary-problems/):
plain constant $\alpha$ retains a $(1-\alpha)^n Q_1$ term forever. Ordinarily
that lingering bias is the flaw. Here it is the entire feature. The unbiased step
size from exercise 2.7 has $\beta_1 = 1$ by construction and wipes $Q_1$ on the
first pull — it is engineered to destroy exactly what this method runs on. The
two techniques are incompatible, and that is not a coincidence.

## How long the optimism lasts

Since the excess decays as $Q_1(1-\alpha)^m$ after $m$ pulls of an arm, the
exploration lasts until that excess falls to roughly the spread $\sigma$ of the
true values:

$$
m^* \approx \frac{\ln(Q_1/\sigma)}{-\ln(1-\alpha)}
\qquad\text{pulls per arm, so about}\qquad
k \cdot m^* \ \text{steps in total}
$$

For $Q_1 = 5$, $\sigma = 1$, $\alpha = 0.1$, $k = 10$: $m^* \approx 15$ pulls per
arm, so roughly **150 steps** of near-uniform sweeping before the method starts
behaving like an exploiter. That is the exploration budget, and it is fixed
before the run begins by two numbers chosen in advance.

## On the 10-armed testbed

Optimistic greedy ($Q_1 = 5$, $\varepsilon = 0$) against realistic
$\varepsilon$-greedy ($Q_1 = 0$, $\varepsilon = 0.1$), both with $\alpha = 0.1$,
averaged over 3000 runs:

| Step | Optimistic, % optimal | $\varepsilon$-greedy, % optimal |
| --- | --- | --- |
| 1 | 9.8 | 10.3 |
| 10 | 9.4 | 28.5 |
| **11** | **42.9** | 29.0 |
| 12 | 23.9 | 29.7 |
| 50 | 19.3 | 39.7 |
| 100 | 29.2 | 45.0 |
| 200 | 59.5 | 52.8 |
| 500 | 80.6 | 65.7 |
| 1000 | 84.6 | 75.8 |

Optimistic greedy is **worse for the first 160 steps or so** — it is busy being
disappointed by all ten arms, roughly the $k \cdot m^* \approx 150$ predicted
above — and takes a durable lead from about step 163 onward, finishing 9 points
higher. The cost is paid entirely up front; the benefit is permanent, because
after the sweep it wastes nothing on random actions the way a fixed $\varepsilon$
does forever.

## The spike at step 11

The early part of that curve has a feature worth explaining, because it looks
like a plotting error and is not (this is exercise 2.6). Averaging over thousands
of runs removes **noise**, not **structure** — and optimistic greedy's
exploration is deterministic, so it does the same thing at the same step in every
single run. Randomly-timed $\varepsilon$-greedy exploration averages smooth;
time-locked sweeping does not.

**Steps 1–10 sit at 10%.** The forced sweep visits each arm once in an arbitrary
order, so the optimal arm turns up at any given step with probability $1/k$.

**Step 11 spikes to 43%.** After one sweep every arm holds
$Q = 4.5 + 0.1 R_i$, so the ranking of estimates is exactly the ranking of the
*first rewards observed*. At step 11 the agent picks the winner of its first
complete comparison, and the optimal arm wins that comparison far more often than
chance. Simulating the underlying quantity directly:

$$
\Pr\left(\arg\max_i R_i^{(1)} = \arg\max_i q_*(i)\right) = 43.0\%
$$

against 42.9% observed at step 11. The spike height *is* that probability.

**Step 12 collapses to 24%.** Having pulled its favourite a second time, the
agent knocks it down to ~4.05, below the nine arms still at ~4.5 — so it is
*forbidden* from repeating and must move on. The optimal arm can only be picked
at step 12 if it was not picked at step 11. Steps 13–17 sag to around 10%,
because the best arm has usually already been consumed near the top of the
ranking.

**The ripples have period $k$.** Each completed sweep produces another
re-ranking at steps 21, 31, and so on, better informed but blunter each time.
They damp out as the sweeps desynchronise — arms stop being pulled equally often,
and the time-locking that let the spike survive averaging breaks down.

## What is wrong with it

**The exploration is transient.** The drive to explore is spent once and does not
return. Any method whose exploration is front-loaded is, as Sutton and Barto put
it, focused on the beginning of time — and the beginning of time occurs only
once.

**So it is the wrong tool for nonstationary problems.** If $q_*(a)$ drifts at
step 5000, the agent has no reason to go looking. The need for exploration
recurs; the optimism does not. This is the sharpest limitation, and it rules the
method out for most problems that matter.

**You must know the reward scale.** $+5$ is only optimistic because we know
$q_* \sim \mathcal{N}(0,1)$. Without a bound on plausible rewards you cannot set
$Q_1$ — too low buys no exploration, too high wastes thousands of pulls sweeping.

**It is not tunable.** $\varepsilon$ is a dial for how much exploration you are
buying and when. Here the amount is an emergent consequence of $Q_1$ and $\alpha$
together, spent at the start, on a schedule you do not control.

## Common mistakes

**Pairing it with sample averages and expecting sustained exploration.** $1/n$
gives $\beta_1 = 1$ in effect: one sweep, then nothing. Use a constant $\alpha$.

**Reading the early curve as a bug.** The spikes are the algorithm, not the
plot. More runs will not smooth them.

**Treating it as a general-purpose exploration method.** It is a good trick on
stationary problems with a known reward scale. Both conditions fail routinely.

**Forgetting that ties still need random breaking.** With every $Q(a)$ equal at
$+5$, an `argmax` that returns the lowest index makes the first sweep run in
index order — harmless here, but the same bug is fatal elsewhere.

## Personal takeaways

The appeal of this method is that it gets exploration for free, without an
exploration parameter, from a quantity you had to choose anyway. The catch is
that "free" means "unbudgeted": you do not decide how much you are spending or
when to stop, and there is no second helping.

The durable idea is bigger than the trick. *Optimism in the face of
uncertainty* — prefer what you have not yet ruled out — is the principle behind
UCB, R-max, and count-based exploration bonuses in deep RL. Optimistic
initialisation is its crudest possible implementation: one optimistic prior,
decaying on a fixed schedule, with no notion of *which* arms it is actually
uncertain about. It treats an arm pulled 200 times and an arm pulled twice
identically once their estimates match. Spending the exploration budget where the
uncertainty really is, and continuing to spend it as uncertainty returns, is what
every better method does.

## Where next

Optimism spent on a fixed schedule is the crude version of the idea. The next
page makes the optimism proportional to the uncertainty that actually remains,
arm by arm:
[upper-confidence-bound action
selection](/Reinforcement-Learning/chapters/02-multi-armed-bandits/upper-confidence-bound/).

## References

- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., §2.6 and exercise 2.6.
