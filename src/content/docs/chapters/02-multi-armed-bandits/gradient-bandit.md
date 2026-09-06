---
title: Gradient bandit algorithms
description: "Learn a preference for each arm instead of an estimate of its value — softmax, a fully derived stochastic gradient ascent, and what the baseline is actually for."
sidebar:
  order: 7
---

## Learning objective

Be able to state the gradient bandit update from memory, derive it line by line
as stochastic gradient ascent on expected reward — including why a baseline may
be subtracted for free and why $\pi_t(A_t)$ cancels — and predict when the
baseline changes the outcome and when it does not.

## Intuition

Every method so far estimates $q_*(a)$ and then argues about how to act on the
estimates. $\varepsilon$-greedy, optimistic starts and
[UCB](/Reinforcement-Learning/chapters/02-multi-armed-bandits/upper-confidence-bound/)
differ only in the argument; all three keep a number per arm that is trying to
be a value.

Gradient bandits skip the value. They keep a **preference** $H_t(a)$ — a number
with no units and no meaning on its own, whose only job is to be larger for arms
worth taking. Preferences are turned into a probability distribution and actions
are drawn from it:

$$
\pi_t(a) \;=\; \Pr\{A_t = a\} \;=\; \frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}}
$$

Two consequences fall out of that definition immediately, and both matter later:

- **Only differences of preferences matter.** Add a constant $c$ to every
  $H_t(a)$ and the $e^{c}$ factors cancel top and bottom, leaving $\pi_t$
  unchanged. There is no absolute scale to learn, which is why $H$ never has to
  agree with $q_*$ about anything.
- **Exploration is graded and automatic.** An arm whose preference is only
  slightly lower keeps a substantial probability; one far behind is nearly never
  taken. Nothing needs an $\varepsilon$ or a bonus term — the policy is already
  stochastic and sharpens itself as the preferences spread out.

## The algorithm

On each step, take $A_t \sim \pi_t$, observe $R_t$, and update **every** arm:

$$
H_{t+1}(a) \;=\; H_t(a) \;+\; \alpha \left(R_t - \bar{R}_t\right)\left(\mathbf{1}_{a = A_t} - \pi_t(a)\right)
$$

where $\mathbf{1}_{a = A_t}$ is 1 for the arm actually taken and 0 otherwise, and
$\bar{R}_t$ is the average of the rewards received before step $t$ — the
**baseline**. Written out as the two cases:

$$
\begin{aligned}
H_{t+1}(A_t) &= H_t(A_t) + \alpha\left(R_t - \bar{R}_t\right)\left(1 - \pi_t(A_t)\right) && \text{the arm taken} \\[4pt]
H_{t+1}(a) &= H_t(a) - \alpha\left(R_t - \bar{R}_t\right)\pi_t(a) && \text{for all } a \neq A_t
\end{aligned}
$$

Read it as a comparison against the average rather than a measurement. If the
reward beat the baseline, the arm taken is made more likely and every other arm
less likely; if it fell short, the arm taken is made less likely and everything
else picks up the slack. The size of each rival's demotion is proportional to how
likely it already was, so probability mass is taken from where it currently sits.

<div class="pseudocode">

**Gradient bandit algorithm**

Initialise, for $a = 1$ to $k$: &nbsp; $H(a) \leftarrow 0$ &nbsp; *(so $\pi$ starts uniform)*, &nbsp; $\bar{R} \leftarrow 0$, &nbsp; $n \leftarrow 0$

**Loop forever:**

$\quad \pi(a) \leftarrow \dfrac{e^{H(a)}}{\sum_b e^{H(b)}}$ &nbsp; for all $a$

$\quad A \sim \pi$ &nbsp; *(sample, do not maximise)*

$\quad R \leftarrow \mathrm{bandit}(A)$

$\quad H(a) \leftarrow H(a) + \alpha\,(R - \bar{R})\,(\mathbf{1}_{a=A} - \pi(a))$ &nbsp; for **all** $a$

$\quad n \leftarrow n + 1$; &nbsp; $\bar{R} \leftarrow \bar{R} + \dfrac{1}{n}\left(R - \bar{R}\right)$ &nbsp; *(baseline updated after the preferences)*

</div>

The order of the last two lines is deliberate: the baseline used at step $t$ must
not depend on which action was taken at step $t$. That is the condition under
which the derivation below goes through, and it is easy to break by folding
$R_t$ into $\bar{R}_t$ first.

## A zero-sum update

Sum the update over all arms:

$$
\sum_{a} \left(\mathbf{1}_{a = A_t} - \pi_t(a)\right) \;=\; \underbrace{1}_{\text{one indicator fires}} \;-\; \underbrace{\sum_a \pi_t(a)}_{=\,1} \;=\; 0
$$

So $\sum_a H_{t+1}(a) = \sum_a H_t(a)$ exactly, on every step, for any reward.
Starting from all zeros, the preferences always sum to zero: the algorithm never
moves their mean, only their spread. This is the same fact as "only differences
matter" seen from the update side, and it is the cheapest possible unit test —
if the sum of your preferences drifts, the update is wrong.

## Worked example

Three arms, $\alpha = 0.1$, all preferences starting at 0 so $\pi$ is uniform.

| Step | $\bar{R}$ | $A$ | $R$ | $R - \bar{R}$ | $H$ after | $\pi$ after |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00 | 2 | 1.0 | $+1.0$ | $(-0.033,\; 0.067,\; -0.033)$ | $(0.322,\; 0.356,\; 0.322)$ |
| 2 | 1.00 | 2 | −0.4 | $-1.4$ | $(0.012,\; -0.024,\; 0.012)$ | $(0.337,\; 0.326,\; 0.337)$ |
| 3 | 0.30 | 3 | 0.6 | $+0.3$ | $(0.002,\; -0.033,\; 0.032)$ | $(0.334,\; 0.322,\; 0.344)$ |

Step 1 rewards arm 2 for beating a baseline of zero. Step 2 punishes it: the
reward was −0.4 against a baseline of 1.0, so the *sign flips* and arm 2 drops
below the two arms that were not even tried. Step 3 promotes arm 3 for a reward
of 0.6 that only just beat the baseline of 0.3, so the move is small. In every
row the three preferences sum to zero.

Notice what the second row means. Arm 2 returned a reward of −0.4, and nothing in
the algorithm knows whether that is good — it is judged entirely against what the
run has been paying so far. That is the whole design.

## Questions that make the update click

<details>
<summary><strong>1. What is a preference, really?</strong></summary>

$H_t(a)$ is a score used to rank actions, not an estimate of the reward from
action $a$. Its absolute value has no meaning. What matters is the gap between
two preferences, because softmax turns that gap into an odds ratio:

$$
\frac{\pi_t(a)}{\pi_t(b)} = e^{H_t(a)-H_t(b)}
$$

If $H_t(a)-H_t(b)=0$, the two actions are equally likely. If the gap is 1,
action $a$ is $e\approx2.72$ times as likely as action $b$. This is why adding
the same constant to every preference changes nothing.

</details>

<details>
<summary><strong>2. If larger preference means a better action, why not choose the largest one?</strong></summary>

Because choosing the largest preference would remove exploration. Gradient
bandits learn a **stochastic policy**, so the softmax probabilities are the
policy, not just an intermediate ranking. Sampling from them lets less-preferred
actions continue to provide evidence. As the evidence accumulates, preference
gaps usually grow and the policy becomes more decisive on its own.

</details>

<details>
<summary><strong>3. What does the reward-minus-baseline term tell us?</strong></summary>

It answers: **was this reward better or worse than what I usually receive?**
This difference is often called an *advantage*:

- $R_t-\bar{R}_t>0$: the chosen action did better than usual, so make it more
  likely.
- $R_t-\bar{R}_t<0$: it did worse than usual, so make it less likely.
- $R_t-\bar{R}_t=0$: this observation gives no reason to change the policy.

A reward of 2 is therefore not inherently good. It is good when the baseline is
1 and bad when the baseline is 3.

</details>

<details>
<summary><strong>4. Why are actions that were not selected updated too?</strong></summary>

Softmax couples all the actions: increasing one action's probability must reduce
the probability available to the others. The factor
$\mathbf{1}_{a=A_t}-\pi_t(a)$ implements that transfer. For a better-than-usual
reward, the selected action moves up while every unselected action moves down;
for a worse-than-usual reward, all the signs reverse.

Updating only the selected action would ignore this competition and would no
longer be the gradient of the softmax policy.

</details>

<details>
<summary><strong>5. Why does the selected action use the one-minus-probability factor?</strong></summary>

The factor measures how much room its probability has to move. A rarely selected
action has $\pi_t(A_t)\approx0$, so a surprising result produces a large update.
An action already chosen with probability near 1 has
$1-\pi_t(A_t)\approx0$, so one more expected success changes little.

For an unselected action, the factor is $-\pi_t(a)$. Likely competitors absorb
more of the change than actions that already have almost no probability. These
factors also make all preference changes sum to zero.

</details>

<details>
<summary><strong>6. Does the baseline change which policy the algorithm is trying to learn?</strong></summary>

No. Subtracting the same action-independent number from every possible reward
leaves the expected gradient unchanged. It changes the variability of the
one-sample update, not its average direction. A useful baseline removes the
common reward level so that the update focuses on differences between actions.

The baseline must not use the current action or reward. That is why the
preferences are updated with the old $\bar{R}_t$ before $R_t$ is folded into the
running average.

</details>

<details>
<summary><strong>7. How can every preference change if only one reward was observed?</strong></summary>

The reward tells us directly about only the selected action, but it tells us how
to redistribute a probability budget shared by all actions. The algorithm is
not pretending that it observed the rewards of the other actions. It is applying
the gradient of one log-probability, and that gradient has one component for
every preference.

This is also why the update is noisy but valid: a single step is only a sample of
the gradient; averaged over many sampled actions and rewards, it points in the
true gradient direction.

</details>

<details>
<summary><strong>8. What is the shortest way to remember the formula?</strong></summary>

Remember three pieces:

$$
\text{preference change}
= \underbrace{\alpha}_{\text{how fast}}
  \underbrace{(R_t-\bar{R}_t)}_{\text{better or worse than usual}}
  \underbrace{(\mathbf{1}_{a=A_t}-\pi_t(a))}_{\text{redistribute probability}}
$$

In words: **move the policy toward a sampled action when it beats the baseline,
and away from it when it falls short**. The derivation below explains why this
simple rule is an unbiased stochastic gradient rather than just a plausible
heuristic.

</details>

## Deriving the update

The claim is that this is not a heuristic: it is stochastic gradient ascent on
expected reward, which is why it inherits the convergence guarantees of gradient
ascent. Here is the derivation with every step justified, because each line uses
a trick worth keeping.

**The objective.** With $q_*$ fixed and the policy determined by the
preferences, the expected reward on step $t$ is

$$
\mathbb{E}[R_t] \;=\; \sum_{x} \pi_t(x)\, q_*(x)
$$

The sum runs over all arms $x$; $\pi_t(x)$ depends on **every** $H_t(a)$ through
the softmax, which is why one arm's preference affects the expected reward of the
whole policy. Exact gradient ascent would be

$$
H_{t+1}(a) \;=\; H_t(a) \;+\; \alpha \frac{\partial\, \mathbb{E}[R_t]}{\partial H_t(a)}
$$

but $\partial \mathbb{E}[R_t] / \partial H_t(a)$ contains $q_*$, which we do not
know. The derivation's job is to rewrite that derivative as the expectation of
something we can compute from one sampled action and its reward.

**Step 1 — differentiate the objective.**

$$
\frac{\partial\, \mathbb{E}[R_t]}{\partial H_t(a)}
\;=\; \frac{\partial}{\partial H_t(a)} \sum_{x} \pi_t(x)\, q_*(x)
\;=\; \sum_{x} q_*(x)\, \frac{\partial \pi_t(x)}{\partial H_t(a)}
$$

*Why:* $q_*(x)$ is a property of the environment, not of our preferences, so it
is a constant with respect to $H_t(a)$ and comes straight out of the derivative.
Only $\pi_t$ is differentiated.

**Step 2 — subtract a baseline, for free.**

$$
\frac{\partial\, \mathbb{E}[R_t]}{\partial H_t(a)}
\;=\; \sum_{x} \left(q_*(x) - B_t\right) \frac{\partial \pi_t(x)}{\partial H_t(a)}
$$

*Why:* the term we just added is $-B_t \sum_x \partial \pi_t(x) / \partial H_t(a)$,
and that sum is zero:

$$
\sum_{x} \frac{\partial \pi_t(x)}{\partial H_t(a)}
\;=\; \frac{\partial}{\partial H_t(a)} \underbrace{\sum_{x} \pi_t(x)}_{=\,1}
\;=\; \frac{\partial\, 1}{\partial H_t(a)} \;=\; 0
$$

The probabilities always sum to one no matter what the preferences are, so the
rates at which they change must cancel. The only requirement on $B_t$ is that it
does **not** depend on $x$ — it may depend on $t$ and on everything that happened
before step $t$, which is what makes the running average $\bar{R}_t$ a legal
choice. Nothing has been approximated here: the gradient is *identical* with any
such baseline. What the baseline changes is the variance of the sample we are
about to take, and that is the entire subject of the next section.

**Step 3 — turn the sum into an expectation.** Multiply and divide each term by
$\pi_t(x)$:

$$
\frac{\partial\, \mathbb{E}[R_t]}{\partial H_t(a)}
\;=\; \sum_{x} \pi_t(x) \left(q_*(x) - B_t\right) \frac{\partial \pi_t(x) / \partial H_t(a)}{\pi_t(x)}
\;=\; \mathbb{E}\!\left[\left(q_*(A_t) - B_t\right) \frac{\partial \pi_t(A_t) / \partial H_t(a)}{\pi_t(A_t)}\right]
$$

*Why:* a sum of the form $\sum_x \pi_t(x) f(x)$ *is* the expectation of $f(A_t)$
when $A_t$ is drawn from $\pi_t$ — which is exactly how the algorithm chooses its
action. This is the pivotal step of the whole derivation: it converts a quantity
that needs all $k$ arms into one that can be estimated from the single arm we
actually pull. (Dividing by $\pi_t(x)$ is safe because the softmax never assigns
an arm probability zero.)

**Step 4 — replace $q_*(A_t)$ with the observed reward.**

$$
\frac{\partial\, \mathbb{E}[R_t]}{\partial H_t(a)}
\;=\; \mathbb{E}\!\left[\left(R_t - \bar{R}_t\right) \frac{\partial \pi_t(A_t) / \partial H_t(a)}{\pi_t(A_t)}\right]
$$

*Why:* by definition $\mathbb{E}[R_t \mid A_t] = q_*(A_t)$, so $R_t$ is an
unbiased sample of the unknown $q_*(A_t)$ and may be substituted inside an
expectation without changing it. This is the step that removes the last piece of
unknown information from the formula, and it is also where the noise enters:
$R_t$ is right on average and wrong on any particular step.

**Step 5 — the softmax derivative.** Write $S = \sum_b e^{H_t(b)}$, so
$\pi_t(x) = e^{H_t(x)} / S$. By the quotient rule:

$$
\begin{aligned}
\frac{\partial \pi_t(x)}{\partial H_t(a)}
&= \frac{\dfrac{\partial e^{H_t(x)}}{\partial H_t(a)} \cdot S \;-\; e^{H_t(x)} \cdot \dfrac{\partial S}{\partial H_t(a)}}{S^2}
&& \text{quotient rule} \\[8pt]
&= \frac{\mathbf{1}_{a=x}\, e^{H_t(x)} S \;-\; e^{H_t(x)} e^{H_t(a)}}{S^2}
&& \begin{array}{l}\text{the numerator moves only if } a = x;\\ \text{in } S \text{ exactly one term contains } H_t(a)\end{array} \\[8pt]
&= \mathbf{1}_{a=x} \frac{e^{H_t(x)}}{S} \;-\; \frac{e^{H_t(x)}}{S}\cdot\frac{e^{H_t(a)}}{S}
&& \text{split the fraction} \\[8pt]
&= \pi_t(x)\left(\mathbf{1}_{a=x} - \pi_t(a)\right)
&& \text{by the definition of } \pi_t
\end{aligned}
$$

*Why it looks the way it does:* raising $H_t(a)$ raises $\pi_t(a)$ and lowers
every other probability, because they must still sum to one. The
$-\pi_t(x)\pi_t(a)$ term is that competition, and it is proportional to how much
probability each arm currently holds.

**Step 6 — substitute and cancel.** Putting step 5 into step 4 with $x = A_t$:

$$
\frac{\partial\, \mathbb{E}[R_t]}{\partial H_t(a)}
\;=\; \mathbb{E}\!\left[\left(R_t - \bar{R}_t\right) \frac{\pi_t(A_t)\left(\mathbf{1}_{a = A_t} - \pi_t(a)\right)}{\pi_t(A_t)}\right]
\;=\; \mathbb{E}\!\left[\left(R_t - \bar{R}_t\right)\left(\mathbf{1}_{a = A_t} - \pi_t(a)\right)\right]
$$

*Why this matters:* the $1/\pi_t(A_t)$ that step 3 introduced is exactly
cancelled by the $\pi_t(x)$ that the softmax derivative produces. That is a
property of the softmax, not a coincidence of algebra, and it is why the final
algorithm contains no division and does not explode when a rarely-taken arm
happens to be sampled.

**Step 7 — drop the expectation.** Gradient ascent with the expectation replaced
by the single sample just drawn is

$$
H_{t+1}(a) \;=\; H_t(a) \;+\; \alpha \left(R_t - \bar{R}_t\right)\left(\mathbf{1}_{a = A_t} - \pi_t(a)\right)
$$

which is the algorithm. Because the sampled quantity has the true gradient as its
expectation, this is stochastic gradient ascent, and its convergence properties
follow — with a decreasing step size it converges to a local optimum, which for
the bandit problem is the global one.

## What the baseline is for

The derivation says the baseline does not change the gradient *at all*. So why
does it change the results so much?

Because the update is a single noisy sample of that gradient, and the baseline
scales the noise. The multiplier on every arm's update is $R_t - B_t$. Choose
$B_t$ badly and that multiplier carries a large constant offset that says nothing
about which arm is good:

- **On a testbed with $q_*(a)$ around $+4$ and no baseline**, $R_t - 0 \approx
  +4$ on every step, whatever was pulled. Every update therefore *promotes the
  arm just taken*, good or bad. The useful signal — the roughly $\pm 1$ variation
  between arms — is a fifth of the size of the meaningless common offset, so the
  preference gaps grow mostly by whichever arm was sampled most, a rich-get-richer
  effect that entrenches early accidents.
- **With the running average as the baseline**, $R_t - \bar{R}_t$ is centred on
  zero. Better than usual promotes, worse than usual demotes, and the common
  offset is gone.

Both versions are climbing the same gradient in expectation. One of them is
reading it through four times as much noise.

## Average performance on the 10-armed testbed

Two testbeds, identical except for where the arms are centred: $q_*(a) \sim
\mathcal{N}(+4, 1)$ — the shifted version from Sutton and Barto's figure 2.5 —
and the standard $\mathcal{N}(0, 1)$. Rewards $R_t \sim \mathcal{N}(q_*(A_t), 1)$,
$H_1(a) = 0$, 2000 runs of 1000 steps.

Every figure here comes from `experiments/gradient-bandit.mjs` in this
repository; it is seeded and dependency-free, so `node
experiments/gradient-bandit.mjs` reprints the tables exactly.

<figure class="chart">
<svg viewBox="0 0 760 400" role="img" aria-label="Gradient bandit on the shifted 10-armed testbed, q* around +4, with and without a baseline" style="width:100%;height:auto;font:13px system-ui,sans-serif">
<title>Gradient bandit on the shifted 10-armed testbed, q* around +4, with and without a baseline</title>
<line x1="52" y1="12" x2="78" y2="12" stroke="#6366f1" stroke-width="2.5"/>
<text x="86" y="16" fill="currentColor" fill-opacity="0.85">α = 0.1, with baseline</text>
<line x1="382" y1="12" x2="408" y2="12" stroke="#10b981" stroke-width="2.5"/>
<text x="416" y="16" fill="currentColor" fill-opacity="0.85">α = 0.4, with baseline</text>
<line x1="52" y1="34" x2="78" y2="34" stroke="#f59e0b" stroke-width="2.5" stroke-dasharray="5 4"/>
<text x="86" y="38" fill="currentColor" fill-opacity="0.85">α = 0.1, no baseline</text>
<line x1="382" y1="34" x2="408" y2="34" stroke="#ef4444" stroke-width="2.5" stroke-dasharray="5 4"/>
<text x="416" y="38" fill="currentColor" fill-opacity="0.85">α = 0.4, no baseline</text>
<line x1="52" y1="360.0" x2="744" y2="360.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="364.0" text-anchor="end" fill="currentColor" fill-opacity="0.7">0%</text>
<line x1="52" y1="285.5" x2="744" y2="285.5" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="289.5" text-anchor="end" fill="currentColor" fill-opacity="0.7">25%</text>
<line x1="52" y1="211.0" x2="744" y2="211.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="215.0" text-anchor="end" fill="currentColor" fill-opacity="0.7">50%</text>
<line x1="52" y1="136.5" x2="744" y2="136.5" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="140.5" text-anchor="end" fill="currentColor" fill-opacity="0.7">75%</text>
<line x1="52" y1="62.0" x2="744" y2="62.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="66.0" text-anchor="end" fill="currentColor" fill-opacity="0.7">100%</text>
<line x1="52.0" y1="360" x2="52.0" y2="365.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="52.0" y="380.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">1</text>
<line x1="224.5" y1="360" x2="224.5" y2="365.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="224.5" y="380.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">250</text>
<line x1="397.7" y1="360" x2="397.7" y2="365.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="397.7" y="380.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">500</text>
<line x1="570.8" y1="360" x2="570.8" y2="365.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="570.8" y="380.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">750</text>
<line x1="744.0" y1="360" x2="744.0" y2="365.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="744.0" y="380.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">1000</text>
<line x1="52" y1="360" x2="744" y2="360" stroke="currentColor" stroke-opacity="0.45"/>
<line x1="52" y1="62" x2="52" y2="360" stroke="currentColor" stroke-opacity="0.45"/>
<polyline points="55.1,326.9 62.0,320.3 69.0,312.0 75.9,301.0 82.8,289.2 89.8,272.8 96.7,259.7 103.6,243.9 110.5,233.0 117.5,221.5 124.4,211.7 131.3,201.6 138.2,194.3 145.2,185.8 152.1,180.6 159.0,175.2 165.9,170.3 172.9,165.9 179.8,161.5 186.7,157.8 193.7,154.3 200.6,150.7 207.5,147.8 214.4,146.6 221.4,144.4 228.3,142.8 235.2,141.4 242.1,139.5 249.1,137.9 256.0,135.5 262.9,135.5 269.9,134.4 276.8,132.9 283.7,131.1 290.6,132.3 297.6,129.4 304.5,129.3 311.4,129.5 318.3,127.3 325.3,126.6 332.2,125.6 339.1,125.3 346.0,125.9 353.0,124.8 359.9,124.0 366.8,123.6 373.8,123.4 380.7,123.3 387.6,122.4 394.5,121.9 401.5,121.1 408.4,120.5 415.3,121.0 422.2,119.8 429.2,120.6 436.1,119.6 443.0,120.1 450.0,118.4 456.9,119.0 463.8,118.5 470.7,119.2 477.7,117.9 484.6,118.2 491.5,117.8 498.4,117.5 505.4,117.6 512.3,117.3 519.2,117.4 526.1,116.5 533.1,116.8 540.0,116.7 546.9,116.1 553.9,116.0 560.8,115.9 567.7,116.0 574.6,115.3 581.6,115.9 588.5,114.7 595.4,115.3 602.3,114.6 609.3,114.7 616.2,114.7 623.1,114.2 630.1,114.1 637.0,113.8 643.9,113.6 650.8,114.4 657.8,113.7 664.7,113.4 671.6,112.7 678.5,113.0 685.5,112.7 692.4,113.3 699.3,113.0 706.2,112.7 713.2,112.4 720.1,112.7 727.0,111.9 734.0,111.7 740.9,112.0" fill="none" stroke="#6366f1" stroke-width="2" stroke-linejoin="round"/>
<polyline points="55.1,315.2 62.0,284.9 69.0,258.8 75.9,239.6 82.8,226.4 89.8,215.6 96.7,209.3 103.6,203.3 110.5,199.4 117.5,196.5 124.4,193.3 131.3,191.5 138.2,189.8 145.2,187.8 152.1,187.2 159.0,186.1 165.9,184.6 172.9,183.2 179.8,182.2 186.7,181.8 193.7,181.1 200.6,180.5 207.5,179.5 214.4,179.2 221.4,178.8 228.3,177.7 235.2,177.0 242.1,176.4 249.1,176.0 256.0,176.0 262.9,174.9 269.9,174.1 276.8,174.1 283.7,173.4 290.6,173.0 297.6,172.8 304.5,172.7 311.4,171.7 318.3,171.6 325.3,171.2 332.2,171.3 339.1,170.8 346.0,170.9 353.0,170.6 359.9,170.5 366.8,170.1 373.8,170.4 380.7,170.2 387.6,169.7 394.5,169.8 401.5,169.2 408.4,169.2 415.3,169.0 422.2,169.0 429.2,168.3 436.1,168.4 443.0,168.3 450.0,167.6 456.9,167.6 463.8,167.7 470.7,167.3 477.7,166.8 484.6,167.1 491.5,166.8 498.4,166.7 505.4,166.7 512.3,166.3 519.2,166.3 526.1,165.9 533.1,165.7 540.0,166.0 546.9,165.6 553.9,165.6 560.8,165.2 567.7,165.3 574.6,164.9 581.6,164.6 588.5,164.5 595.4,164.6 602.3,164.5 609.3,164.4 616.2,164.1 623.1,164.3 630.1,164.5 637.0,164.3 643.9,164.3 650.8,164.3 657.8,164.1 664.7,164.3 671.6,164.1 678.5,163.8 685.5,164.0 692.4,163.7 699.3,163.4 706.2,163.5 713.2,163.3 720.1,163.6 727.0,163.3 734.0,163.3 740.9,163.5" fill="none" stroke="#10b981" stroke-width="2" stroke-linejoin="round"/>
<polyline points="55.1,325.9 62.0,318.2 69.0,310.6 75.9,300.7 82.8,291.8 89.8,282.4 96.7,276.7 103.6,271.7 110.5,266.3 117.5,261.2 124.4,257.3 131.3,254.2 138.2,250.0 145.2,248.0 152.1,246.3 159.0,243.2 165.9,242.4 172.9,240.5 179.8,239.3 186.7,237.8 193.7,236.2 200.6,235.0 207.5,232.9 214.4,231.9 221.4,231.1 228.3,230.0 235.2,228.6 242.1,228.0 249.1,228.0 256.0,226.7 262.9,225.8 269.9,224.8 276.8,223.7 283.7,223.8 290.6,222.5 297.6,222.1 304.5,221.6 311.4,221.6 318.3,220.9 325.3,220.4 332.2,220.3 339.1,219.6 346.0,219.5 353.0,219.1 359.9,218.3 366.8,218.3 373.8,218.3 380.7,217.8 387.6,216.9 394.5,216.8 401.5,216.1 408.4,215.5 415.3,215.4 422.2,215.2 429.2,215.0 436.1,215.1 443.0,214.9 450.0,214.6 456.9,214.5 463.8,214.1 470.7,214.2 477.7,214.3 484.6,213.3 491.5,213.7 498.4,213.5 505.4,213.6 512.3,213.0 519.2,213.1 526.1,212.4 533.1,212.9 540.0,212.9 546.9,212.6 553.9,212.4 560.8,212.2 567.7,212.2 574.6,211.7 581.6,211.4 588.5,211.2 595.4,211.7 602.3,211.3 609.3,211.0 616.2,211.0 623.1,210.8 630.1,210.7 637.0,211.1 643.9,210.5 650.8,210.6 657.8,210.4 664.7,210.0 671.6,210.1 678.5,209.7 685.5,209.8 692.4,209.7 699.3,209.7 706.2,209.1 713.2,209.0 720.1,208.9 727.0,208.5 734.0,208.7 740.9,208.8" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linejoin="round" stroke-dasharray="5 4"/>
<polyline points="55.1,317.1 62.0,302.7 69.0,296.2 75.9,293.5 82.8,291.8 89.8,290.8 96.7,290.2 103.6,289.4 110.5,289.3 117.5,288.7 124.4,287.7 131.3,287.2 138.2,286.6 145.2,286.1 152.1,285.7 159.0,285.4 165.9,284.8 172.9,284.2 179.8,284.2 186.7,283.4 193.7,283.6 200.6,283.3 207.5,283.2 214.4,282.6 221.4,282.1 228.3,281.8 235.2,281.7 242.1,281.8 249.1,281.4 256.0,281.1 262.9,281.0 269.9,280.6 276.8,280.5 283.7,280.2 290.6,280.1 297.6,279.3 304.5,279.2 311.4,278.8 318.3,278.6 325.3,278.4 332.2,278.8 339.1,278.9 346.0,278.7 353.0,278.7 359.9,278.5 366.8,278.4 373.8,278.3 380.7,278.2 387.6,278.0 394.5,277.7 401.5,277.6 408.4,277.5 415.3,277.3 422.2,277.4 429.2,277.6 436.1,277.3 443.0,277.2 450.0,276.9 456.9,277.1 463.8,277.3 470.7,277.3 477.7,277.3 484.6,277.5 491.5,277.5 498.4,277.1 505.4,277.0 512.3,276.9 519.2,276.9 526.1,276.9 533.1,277.1 540.0,277.0 546.9,277.1 553.9,276.8 560.8,276.6 567.7,276.6 574.6,276.3 581.6,276.3 588.5,276.2 595.4,276.4 602.3,276.7 609.3,276.5 616.2,276.5 623.1,276.7 630.1,276.9 637.0,277.2 643.9,277.1 650.8,277.1 657.8,276.9 664.7,276.9 671.6,276.7 678.5,276.9 685.5,277.1 692.4,277.0 699.3,277.0 706.2,276.8 713.2,276.7 720.1,276.7 727.0,276.5 734.0,276.4 740.9,276.7" fill="none" stroke="#ef4444" stroke-width="2" stroke-linejoin="round" stroke-dasharray="5 4"/>
<text x="398" y="396" text-anchor="middle" fill="currentColor" fill-opacity="0.7">Steps</text>
<text transform="translate(14 211) rotate(-90)" text-anchor="middle" fill="currentColor" fill-opacity="0.7">% optimal action</text>
</svg>
<figcaption>Percent optimal action on the shifted testbed (q* around +4), 2000 runs. Without a baseline every reward is positive, so every update promotes whatever was just tried.</figcaption>
</figure>

| Step | $\alpha=0.1$, baseline | $\alpha=0.4$, baseline | $\alpha=0.1$, none | $\alpha=0.4$, none |
| --- | --- | --- | --- | --- |
| 10 | 11.3 | 19.9 | 12.5 | 17.4 |
| 50 | 26.6 | 46.5 | 23.6 | 22.7 |
| 100 | 47.3 | 54.9 | 34.3 | 24.1 |
| 200 | 68.3 | 59.9 | 41.8 | 25.8 |
| 500 | 80.2 | 63.9 | 48.0 | 27.7 |
| 1000 | **83.4** | 66.0 | 50.8 | **27.9** |

Removing the baseline costs 33 percentage points at $\alpha = 0.1$ and 38 at
$\alpha = 0.4$, and the larger step size makes things worse rather than better
without one — it entrenches the early accidents faster. The average preference
gap between the optimal arm and the mean tells the same story from inside the
algorithm: 5.58 with the baseline against 3.32 without, at $\alpha = 0.1$.

Now the same four settings on the standard testbed, where the rewards are already
centred near zero:

<figure class="chart">
<svg viewBox="0 0 760 400" role="img" aria-label="The same four settings on the standard testbed, q* around 0, where the baseline barely matters" style="width:100%;height:auto;font:13px system-ui,sans-serif">
<title>The same four settings on the standard testbed, q* around 0, where the baseline barely matters</title>
<line x1="52" y1="12" x2="78" y2="12" stroke="#6366f1" stroke-width="2.5"/>
<text x="86" y="16" fill="currentColor" fill-opacity="0.85">α = 0.1, with baseline</text>
<line x1="382" y1="12" x2="408" y2="12" stroke="#10b981" stroke-width="2.5"/>
<text x="416" y="16" fill="currentColor" fill-opacity="0.85">α = 0.4, with baseline</text>
<line x1="52" y1="34" x2="78" y2="34" stroke="#f59e0b" stroke-width="2.5" stroke-dasharray="5 4"/>
<text x="86" y="38" fill="currentColor" fill-opacity="0.85">α = 0.1, no baseline</text>
<line x1="382" y1="34" x2="408" y2="34" stroke="#ef4444" stroke-width="2.5" stroke-dasharray="5 4"/>
<text x="416" y="38" fill="currentColor" fill-opacity="0.85">α = 0.4, no baseline</text>
<line x1="52" y1="360.0" x2="744" y2="360.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="364.0" text-anchor="end" fill="currentColor" fill-opacity="0.7">0%</text>
<line x1="52" y1="285.5" x2="744" y2="285.5" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="289.5" text-anchor="end" fill="currentColor" fill-opacity="0.7">25%</text>
<line x1="52" y1="211.0" x2="744" y2="211.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="215.0" text-anchor="end" fill="currentColor" fill-opacity="0.7">50%</text>
<line x1="52" y1="136.5" x2="744" y2="136.5" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="140.5" text-anchor="end" fill="currentColor" fill-opacity="0.7">75%</text>
<line x1="52" y1="62.0" x2="744" y2="62.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="66.0" text-anchor="end" fill="currentColor" fill-opacity="0.7">100%</text>
<line x1="52.0" y1="360" x2="52.0" y2="365.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="52.0" y="380.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">1</text>
<line x1="224.5" y1="360" x2="224.5" y2="365.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="224.5" y="380.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">250</text>
<line x1="397.7" y1="360" x2="397.7" y2="365.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="397.7" y="380.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">500</text>
<line x1="570.8" y1="360" x2="570.8" y2="365.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="570.8" y="380.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">750</text>
<line x1="744.0" y1="360" x2="744.0" y2="365.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="744.0" y="380.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">1000</text>
<line x1="52" y1="360" x2="744" y2="360" stroke="currentColor" stroke-opacity="0.45"/>
<line x1="52" y1="62" x2="52" y2="360" stroke="currentColor" stroke-opacity="0.45"/>
<polyline points="55.1,327.8 62.0,320.6 69.0,313.0 75.9,301.9 82.8,289.2 89.8,272.1 96.7,259.4 103.6,243.7 110.5,232.0 117.5,219.2 124.4,209.6 131.3,200.1 138.2,190.6 145.2,182.8 152.1,178.2 159.0,172.5 165.9,167.8 172.9,164.0 179.8,158.4 186.7,154.9 193.7,152.4 200.6,148.0 207.5,146.0 214.4,144.1 221.4,141.3 228.3,140.0 235.2,137.7 242.1,136.1 249.1,134.4 256.0,132.7 262.9,132.0 269.9,130.9 276.8,130.0 283.7,127.8 290.6,128.6 297.6,126.5 304.5,125.6 311.4,126.1 318.3,124.1 325.3,123.1 332.2,122.9 339.1,122.1 346.0,123.0 353.0,121.6 359.9,120.8 366.8,121.1 373.8,120.9 380.7,120.1 387.6,119.6 394.5,119.4 401.5,118.1 408.4,117.8 415.3,118.1 422.2,116.9 429.2,117.3 436.1,116.9 443.0,116.6 450.0,115.6 456.9,116.2 463.8,115.3 470.7,115.9 477.7,114.7 484.6,115.0 491.5,114.8 498.4,114.6 505.4,114.5 512.3,114.0 519.2,114.6 526.1,113.1 533.1,113.3 540.0,113.1 546.9,113.2 553.9,112.8 560.8,112.6 567.7,112.5 574.6,112.0 581.6,112.5 588.5,111.8 595.4,112.0 602.3,111.6 609.3,110.6 616.2,110.9 623.1,110.7 630.1,110.8 637.0,110.5 643.9,110.7 650.8,111.0 657.8,110.4 664.7,110.2 671.6,109.9 678.5,110.0 685.5,109.9 692.4,110.4 699.3,109.7 706.2,109.8 713.2,109.9 720.1,109.6 727.0,109.5 734.0,109.4 740.9,109.2" fill="none" stroke="#6366f1" stroke-width="2" stroke-linejoin="round"/>
<polyline points="55.1,315.7 62.0,276.1 69.0,240.5 75.9,220.2 82.8,205.3 89.8,195.8 96.7,190.1 103.6,185.4 110.5,182.0 117.5,178.5 124.4,175.9 131.3,174.4 138.2,172.1 145.2,170.4 152.1,169.3 159.0,168.3 165.9,166.7 172.9,166.3 179.8,165.3 186.7,164.6 193.7,164.5 200.6,163.6 207.5,162.7 214.4,161.9 221.4,161.5 228.3,160.2 235.2,159.5 242.1,158.9 249.1,158.7 256.0,158.3 262.9,157.7 269.9,157.2 276.8,157.1 283.7,156.3 290.6,156.3 297.6,155.7 304.5,155.6 311.4,155.0 318.3,155.0 325.3,154.7 332.2,154.7 339.1,154.0 346.0,154.4 353.0,153.5 359.9,153.8 366.8,153.7 373.8,153.7 380.7,153.1 387.6,152.9 394.5,152.4 401.5,152.2 408.4,151.8 415.3,151.8 422.2,151.5 429.2,151.5 436.1,151.2 443.0,151.3 450.0,150.9 456.9,150.7 463.8,150.3 470.7,150.1 477.7,150.2 484.6,150.0 491.5,149.4 498.4,149.3 505.4,149.3 512.3,149.1 519.2,149.0 526.1,148.4 533.1,148.4 540.0,148.5 546.9,148.3 553.9,148.4 560.8,148.1 567.7,148.0 574.6,148.0 581.6,147.7 588.5,147.6 595.4,147.4 602.3,147.3 609.3,147.4 616.2,147.2 623.1,147.1 630.1,147.5 637.0,147.2 643.9,147.2 650.8,147.4 657.8,147.1 664.7,147.4 671.6,146.9 678.5,147.0 685.5,147.0 692.4,147.0 699.3,146.8 706.2,146.6 713.2,146.5 720.1,146.6 727.0,146.5 734.0,146.5 740.9,146.6" fill="none" stroke="#10b981" stroke-width="2" stroke-linejoin="round"/>
<polyline points="55.1,327.7 62.0,320.2 69.0,313.2 75.9,301.2 82.8,288.1 89.8,271.7 96.7,258.0 103.6,243.1 110.5,231.3 117.5,219.5 124.4,208.0 131.3,199.7 138.2,191.5 145.2,183.5 152.1,178.4 159.0,173.9 165.9,167.9 172.9,164.3 179.8,159.9 186.7,155.8 193.7,152.8 200.6,149.2 207.5,146.2 214.4,145.1 221.4,142.6 228.3,140.3 235.2,138.9 242.1,136.9 249.1,136.6 256.0,134.0 262.9,132.8 269.9,132.8 276.8,131.2 283.7,129.7 290.6,129.6 297.6,128.3 304.5,127.5 311.4,127.7 318.3,127.0 325.3,125.7 332.2,124.4 339.1,124.1 346.0,124.5 353.0,123.7 359.9,123.5 366.8,123.0 373.8,122.5 380.7,122.4 387.6,121.5 394.5,121.3 401.5,120.3 408.4,120.2 415.3,119.9 422.2,119.2 429.2,119.9 436.1,119.0 443.0,119.1 450.0,116.9 456.9,117.9 463.8,117.3 470.7,117.3 477.7,116.7 484.6,116.8 491.5,116.8 498.4,116.4 505.4,116.8 512.3,116.2 519.2,115.8 526.1,114.9 533.1,115.1 540.0,115.4 546.9,114.9 553.9,114.5 560.8,114.1 567.7,114.5 574.6,114.3 581.6,114.8 588.5,113.9 595.4,114.0 602.3,113.4 609.3,113.2 616.2,113.5 623.1,113.2 630.1,112.7 637.0,112.5 643.9,112.6 650.8,113.2 657.8,113.0 664.7,112.5 671.6,112.1 678.5,111.8 685.5,111.5 692.4,112.1 699.3,111.4 706.2,111.5 713.2,111.1 720.1,111.3 727.0,111.1 734.0,111.0 740.9,110.9" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linejoin="round" stroke-dasharray="5 4"/>
<polyline points="55.1,315.6 62.0,274.7 69.0,239.1 75.9,216.7 82.8,201.9 89.8,191.9 96.7,186.5 103.6,182.1 110.5,178.7 117.5,175.6 124.4,174.2 131.3,172.6 138.2,171.9 145.2,170.4 152.1,169.3 159.0,167.7 165.9,166.2 172.9,165.5 179.8,164.5 186.7,164.2 193.7,163.6 200.6,163.5 207.5,163.0 214.4,162.8 221.4,162.6 228.3,161.4 235.2,161.1 242.1,160.1 249.1,159.8 256.0,159.4 262.9,159.1 269.9,158.7 276.8,158.3 283.7,157.5 290.6,157.5 297.6,157.1 304.5,157.3 311.4,156.8 318.3,157.1 325.3,156.4 332.2,156.4 339.1,155.9 346.0,156.1 353.0,155.9 359.9,155.7 366.8,155.7 373.8,156.0 380.7,155.5 387.6,155.8 394.5,155.5 401.5,155.3 408.4,154.9 415.3,154.5 422.2,154.6 429.2,154.3 436.1,154.4 443.0,154.3 450.0,153.7 456.9,153.6 463.8,153.9 470.7,153.6 477.7,153.5 484.6,153.3 491.5,153.1 498.4,152.9 505.4,153.2 512.3,153.1 519.2,153.0 526.1,152.8 533.1,152.8 540.0,152.8 546.9,152.5 553.9,152.4 560.8,152.4 567.7,152.4 574.6,152.3 581.6,152.2 588.5,152.2 595.4,152.2 602.3,151.6 609.3,151.7 616.2,151.5 623.1,151.6 630.1,151.7 637.0,151.5 643.9,151.6 650.8,151.6 657.8,151.5 664.7,151.8 671.6,151.6 678.5,151.5 685.5,151.7 692.4,151.5 699.3,151.2 706.2,151.1 713.2,151.2 720.1,151.2 727.0,151.1 734.0,151.0 740.9,151.1" fill="none" stroke="#ef4444" stroke-width="2" stroke-linejoin="round" stroke-dasharray="5 4"/>
<text x="398" y="396" text-anchor="middle" fill="currentColor" fill-opacity="0.7">Steps</text>
<text transform="translate(14 211) rotate(-90)" text-anchor="middle" fill="currentColor" fill-opacity="0.7">% optimal action</text>
</svg>
<figcaption>The same four settings with the arms centred at zero. The dashed no-baseline curves lie on top of the solid ones: the baseline was only ever removing the offset.</figcaption>
</figure>

| Step | $\alpha=0.1$, baseline | $\alpha=0.4$, baseline | $\alpha=0.1$, none | $\alpha=0.4$, none |
| --- | --- | --- | --- | --- |
| 100 | 48.0 | 61.0 | 48.4 | 61.7 |
| 500 | 80.5 | 69.8 | 80.5 | 68.8 |
| 1000 | 84.5 | 71.7 | 84.0 | 70.0 |

The curves collapse into two pairs — half a point apart at $\alpha = 0.1$, under
two at $\alpha = 0.4$, against gaps of 33 and 38 on the shifted testbed.
That is the cleanest statement of what it does: it is not a general accelerator,
it is the term that makes the method **indifferent to a constant added to every
reward**. Without it, the algorithm's behaviour depends on where zero happens to
sit — which is an arbitrary fact about how the problem was written down, and
should not affect anything.

The step size shows the familiar horizon trade-off, unchanged by any of this:
$\alpha = 0.4$ leads for the first 133 steps and then plateaus lower, because
large steps sharpen the policy before the evidence justifies it and the resulting
near-deterministic $\pi$ stops collecting evidence about the arms it has written
off.

## What is wrong with it

**It learns no values.** Preferences are only comparable within one problem, and
there is nothing to inspect afterwards: the algorithm cannot tell you what an arm
is worth, only that it prefers it. Every value-based method leaves behind an
estimate that is useful on its own.

**The policy can sharpen prematurely.** Nothing forces continued exploration. If
early noise pushes one preference far ahead, $\pi$ becomes nearly deterministic
and the updates to the neglected arms shrink to almost nothing — visible above as
the $\alpha = 0.4$ plateau.

**It is sensitive to reward scale, not just offset.** The baseline removes a
constant, but multiplying every reward by 100 multiplies the effective step size
by 100. $\alpha$ must be tuned for the scale of $R_t - \bar{R}_t$.

**The baseline is a running average of everything.** On a nonstationary problem
the average over the whole history is the wrong reference point, and a constant
step-size average of recent rewards is a better one — the same argument as
[tracking a nonstationary
problem](/Reinforcement-Learning/chapters/02-multi-armed-bandits/nonstationary-problems/).

## Common mistakes

**Updating only the arm that was taken.** The other $k-1$ preferences move on
every step. Skipping them breaks the zero-sum property, and the policy drifts
toward whatever is sampled most.

**Letting the baseline see the current action.** Folding $R_t$ into $\bar{R}_t$
*before* the preference update makes $B_t$ a function of $A_t$, which is exactly
the assumption step 2 needs. The bias is small but the derivation no longer
applies, and it is free to avoid.

**Taking $\arg\max_a \pi_t(a)$ instead of sampling.** The method is a stochastic
policy; the exploration *is* the sampling. Acting greedily on $\pi$ turns it into
a worse value method with no exploration at all.

**Exponentiating raw preferences.** $e^{H}$ overflows once preferences reach a
few hundred. Subtract $\max_b H_t(b)$ from every preference before
exponentiating — it cancels exactly, by the same argument that says only
differences matter.

**Reading $H$ as a value.** $H_t(a) = 3$ says nothing about reward. Only
$H_t(a) - H_t(b)$ has meaning, and only through the softmax.

## Personal takeaways

The gradient bandit is the first method here that does not pretend to be
estimating anything. It optimises the policy directly, and the derivation is the
reason to care: seven lines take "maximise expected reward" to an update rule
that uses one sampled action, with no unknown quantities left in it, and every
step is a legitimate equality rather than an approximation. The one place a
sample replaces an expectation is step 7, and that is what makes it *stochastic*
gradient ascent.

The baseline result is the part I expect to keep re-using. It is a term that
provably does not change what is being optimised and dramatically changes how
fast you get there, purely by removing a common offset from a noisy estimate.
Subtracting a baseline is the standard variance-reduction move in policy gradient
methods for exactly this reason, and this chapter is where it is small enough to
verify by hand.

The connection to chapter 11 is direct. Replace "arm" with "state–action pair"
and $\bar{R}_t$ with a learned value function, and this update is REINFORCE with
a baseline. The softmax over preferences becomes a softmax policy over actions,
and step 3 — turning a sum over all actions into an expectation over the action
taken — becomes the policy gradient theorem.

## Where next

That closes the bandit chapter: four exploration strategies, all of them
operating on a problem with no state, no transitions and no delay. The next
chapter puts all three back:
[Markov decision processes](/Reinforcement-Learning/chapters/03-markov-decision-processes/).

## References

- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., §2.8 and figure 2.5.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning* 8, 229–256.
