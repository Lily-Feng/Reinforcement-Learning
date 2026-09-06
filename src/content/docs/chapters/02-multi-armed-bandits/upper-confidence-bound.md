---
title: Upper-confidence-bound action selection
description: "Explore the arm you are least sure about, not one drawn at random — and the mysterious spike at step 11 that falls out of doing so."
sidebar:
  order: 6
---

## Learning objective

Be able to write the UCB rule from memory, say what each term in the bonus is
doing and why the bonus has that shape, predict how $c$ changes behaviour, and
explain the step-11 spike on the 10-armed testbed in both directions — why the
reward jumps and why it falls again.

## Intuition

Every method so far explores badly in the same way.
[$\varepsilon$-greedy](/Reinforcement-Learning/chapters/02-multi-armed-bandits/epsilon-greedy/)
explores *uniformly*: an arm pulled 500 times with a terrible payout is exactly
as likely to be sampled as a promising arm pulled twice. It knows which arms it
is unsure about and throws that information away at the moment it acts on it.
[Optimistic initial
values](/Reinforcement-Learning/chapters/02-multi-armed-bandits/optimistic-initial-values/)
do better — they at least sweep the arms in a useful order — but the optimism
decays on a fixed schedule regardless of what has actually been learned, and once
spent it never comes back.

UCB fixes both faults with one idea. Do not choose *whether* to explore; make
uncertainty part of what "best" means. An arm is worth taking if it either looks
good or is poorly understood, and the rule adds those two things together.

## The rule

$$
A_t \;=\; \arg\max_a \left[\, Q_t(a) \;+\; c \sqrt{\frac{\ln t}{N_t(a)}} \,\right]
$$

$Q_t(a)$ is the usual estimate, $N_t(a)$ the number of times $a$ has been taken
before step $t$, and $c > 0$ controls how much the uncertainty is worth. If
$N_t(a) = 0$, arm $a$ is treated as maximising — it is taken before any arm with
a finite value.

The whole rule is one quantity: an **upper confidence bound** on $q_*(a)$. The
square-root term is a plausible amount by which the true value might exceed the
current estimate, so the bracket is roughly the best case still consistent with
the data. UCB then acts greedily with respect to that best case. This is
*optimism in the face of uncertainty*, made specific: not "assume everything is
great", but "assume each arm is as good as its own evidence still permits".

Read the bonus one symbol at a time:

- **$N_t(a)$ in the denominator.** Each pull of $a$ shrinks $a$'s own bonus. Take
  an arm and you reduce your reason to take it again — self-correcting, and
  targeted at the arm actually sampled rather than spread over all of them.
- **$\ln t$ in the numerator.** Every arm's bonus creeps up whenever *any* arm is
  pulled. A neglected arm slowly becomes attractive again purely because time has
  passed, so no arm is abandoned forever.
- **The logarithm.** $\ln t$ grows without bound, so exploration never fully
  stops, but it grows slower than any power of $t$, so the fraction of pulls
  spent exploring goes to zero. That is exactly the balance you want: enough
  exploration to be sure, little enough to be cheap.
- **The square root.** The standard error of a mean of $n$ samples shrinks like
  $1/\sqrt{n}$. The bonus is a confidence-interval half-width, not an arbitrary
  penalty.
- **$c$.** The confidence level, in units of that half-width. Larger $c$ means
  wider intervals, more optimism, more exploration.

Nothing here is random. Given the rewards, the sequence of actions is
determined — a fact that looks like a footnote and turns out to explain the
whole shape of the learning curve below.

## Where the $\sqrt{\ln t / N}$ comes from

Hoeffding's inequality says that for a mean $Q_t(a)$ of $N_t(a)$ bounded samples,

$$
\Pr\left(q_*(a) > Q_t(a) + u\right) \;\le\; \exp\left(-2 N_t(a) u^2\right)
$$

Ask for that failure probability to be $t^{-4}$ — a tolerance that tightens as
the run gets longer, so the total chance of ever being wrong stays finite — and
solve for $u$:

$$
\exp\left(-2 N_t(a) u^2\right) = t^{-4}
\quad\Longrightarrow\quad
u = \sqrt{\frac{2 \ln t}{N_t(a)}}
$$

which is the bonus, with $c$ absorbing the constant and the reward scale. So $c$
is not a free-floating knob: it is a claim about how wide the confidence interval
on an arm's value really is, and the right value depends on the noise in the
rewards. With $c$ too small the bound is not an upper bound at all and UCB can
settle early on a wrong arm; with $c$ too large it keeps re-checking arms it has
already ruled out.

## The algorithm

<div class="pseudocode">

**UCB, sample-average estimates**

Initialise, for $a = 1$ to $k$: &nbsp; $Q(a) \leftarrow 0$, &nbsp; $N(a) \leftarrow 0$

**For** $t = 1, 2, 3, \dots$:

$\quad A \leftarrow \arg\max_a \left[ Q(a) + c\sqrt{\dfrac{\ln t}{N(a)}} \right]$ &nbsp; *(any $a$ with $N(a) = 0$ counts as maximal; ties broken randomly)*

$\quad R \leftarrow \mathrm{bandit}(A)$

$\quad N(A) \leftarrow N(A) + 1$

$\quad Q(A) \leftarrow Q(A) + \dfrac{1}{N(A)}\left[R - Q(A)\right]$

</div>

There is no $\varepsilon$ and no coin flip. The only new parameter is $c$, and
$t$ is the global step count, not the per-arm count — the two appear in the same
expression and confusing them breaks the method.

## How big is the bonus, really

With $c = 2$, on a testbed where the rewards themselves are around $\pm 2$:

| $t$ | $N_t(a)$ | bonus $2\sqrt{\ln t / N_t(a)}$ |
| --- | --- | --- |
| 11 | 1 | 3.10 |
| 100 | 1 | 4.29 |
| 100 | 10 | 1.36 |
| 1000 | 10 | 1.66 |
| 1000 | 100 | 0.53 |
| 1000 | 900 | 0.18 |

Early on the bonus dwarfs every difference in $Q$, which is why the first
several steps are pure exploration. By step 1000 an arm that has taken 900 of the
pulls carries a bonus of 0.18 while an arm sampled only 10 times carries 1.66 —
so the neglected arm gets pulled again unless it is believed to be about 1.5
worse. That gap is the exploration, and it is priced per arm.

## Average performance on the 10-armed testbed

Standard testbed: $k = 10$, $q_*(a) \sim \mathcal{N}(0,1)$ drawn fresh per run,
$R_t \sim \mathcal{N}(q_*(A_t), 1)$, sample-average estimates, $Q_1(a) = 0$,
2000 independent runs of 1000 steps. UCB at $c = 2$ and $c = 1$ against
$\varepsilon$-greedy at $\varepsilon = 0.1$.

Every figure below comes from `experiments/ucb-testbed.mjs` in this repository.
It is seeded and dependency-free, so `node experiments/ucb-testbed.mjs` reprints
the table and the spike diagnostics exactly.

<figure class="chart">
<svg viewBox="0 0 760 380" role="img" aria-label="Average reward per step on the 10-armed testbed: UCB c=2, UCB c=1, and epsilon-greedy" style="width:100%;height:auto;font:13px system-ui,sans-serif">
<title>Average reward per step on the 10-armed testbed: UCB c=2, UCB c=1, and epsilon-greedy</title>
<line x1="52" y1="303.8" x2="744" y2="303.8" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="307.8" text-anchor="end" fill="currentColor" fill-opacity="0.7">0</text>
<line x1="52" y1="213.2" x2="744" y2="213.2" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="217.2" text-anchor="end" fill="currentColor" fill-opacity="0.7">0.5</text>
<line x1="52" y1="122.7" x2="744" y2="122.7" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="126.7" text-anchor="end" fill="currentColor" fill-opacity="0.7">1</text>
<line x1="52" y1="32.1" x2="744" y2="32.1" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="36.1" text-anchor="end" fill="currentColor" fill-opacity="0.7">1.5</text>
<line x1="52.0" y1="340" x2="52.0" y2="345.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="52.0" y="360.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">1</text>
<line x1="224.5" y1="340" x2="224.5" y2="345.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="224.5" y="360.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">250</text>
<line x1="397.7" y1="340" x2="397.7" y2="345.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="397.7" y="360.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">500</text>
<line x1="570.8" y1="340" x2="570.8" y2="345.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="570.8" y="360.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">750</text>
<line x1="744.0" y1="340" x2="744.0" y2="345.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="744.0" y="360.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">1000</text>
<line x1="52" y1="340" x2="744" y2="340" stroke="currentColor" stroke-opacity="0.45"/>
<line x1="52" y1="14" x2="52" y2="340" stroke="currentColor" stroke-opacity="0.45"/>
<polyline points="52.0,293.7 52.7,309.1 53.4,304.6 54.1,297.6 54.8,307.8 55.5,317.2 56.2,308.9 56.8,304.9 57.5,304.2 58.2,308.5 58.9,102.7 59.6,117.6 60.3,115.5 61.0,118.7 61.7,109.3 62.4,116.4 63.1,107.7 63.8,103.2 64.5,92.2 65.2,95.9 65.9,102.9 66.5,88.6 67.2,86.8 67.9,90.1 68.6,83.8 69.3,85.4 70.0,78.4 70.7,81.1 71.4,73.9 72.1,77.8 72.8,85.6 73.5,81.4 74.2,72.2 74.9,69.3 75.6,73.6 76.2,76.0 76.9,74.2 77.6,64.9 78.3,66.6 79.0,70.1 82.8,61.0 89.8,54.8 96.7,52.9 103.6,50.4 110.5,45.4 117.5,45.9 124.4,43.0 131.3,42.9 138.2,39.7 145.2,38.0 152.1,38.4 159.0,37.6 165.9,36.8 172.9,37.2 179.8,36.7 186.7,34.6 193.7,37.8 200.6,37.3 207.5,33.8 214.4,35.0 221.4,34.4 228.3,33.4 235.2,36.3 242.1,34.2 249.1,34.0 256.0,32.4 262.9,35.1 269.9,33.6 276.8,32.2 283.7,31.6 290.6,34.4 297.6,33.4 304.5,33.7 311.4,32.1 318.3,33.8 325.3,30.4 332.2,30.1 339.1,31.3 346.0,34.6 353.0,31.9 359.9,33.2 366.8,32.0 373.8,30.8 380.7,30.4 387.6,30.5 394.5,32.4 401.5,30.6 408.4,31.4 415.3,33.4 422.2,32.8 429.2,32.5 436.1,29.7 443.0,31.0 450.0,29.7 456.9,30.8 463.8,31.9 470.7,29.4 477.7,29.3 484.6,29.4 491.5,30.5 498.4,28.2 505.4,29.7 512.3,28.5 519.2,29.3 526.1,29.6 533.1,29.8 540.0,29.3 546.9,30.3 553.9,29.3 560.8,29.8 567.7,30.6 574.6,29.3 581.6,31.2 588.5,28.9 595.4,28.4 602.3,29.5 609.3,29.9 616.2,28.8 623.1,29.0 630.1,30.2 637.0,27.4 643.9,30.4 650.8,29.2 657.8,29.9 664.7,28.3 671.6,27.4 678.5,28.2 685.5,30.8 692.4,29.2 699.3,32.5 706.2,29.6 713.2,28.0 720.1,29.4 727.0,30.4 734.0,27.6 740.9,29.8" fill="none" stroke="#10b981" stroke-width="2" stroke-linejoin="round"/>
<polyline points="52.0,299.3 52.7,258.2 53.4,238.9 54.1,206.0 54.8,203.4 55.5,191.7 56.2,175.0 56.8,168.7 57.5,156.7 58.2,160.7 58.9,151.7 59.6,150.7 60.3,153.9 61.0,142.0 61.7,142.3 62.4,149.5 63.1,134.8 63.8,129.8 64.5,130.3 65.2,135.5 65.9,131.3 66.5,119.6 67.2,123.9 67.9,126.3 68.6,123.5 69.3,119.8 70.0,130.0 70.7,123.8 71.4,118.2 72.1,127.0 72.8,120.9 73.5,116.4 74.2,117.2 74.9,114.5 75.6,107.5 76.2,115.7 76.9,117.2 77.6,108.6 78.3,122.1 79.0,112.4 82.8,110.4 89.8,102.7 96.7,98.7 103.6,94.1 110.5,93.6 117.5,90.7 124.4,88.4 131.3,85.2 138.2,83.1 145.2,80.9 152.1,81.7 159.0,77.3 165.9,77.4 172.9,77.2 179.8,74.8 186.7,73.7 193.7,71.0 200.6,74.0 207.5,72.7 214.4,72.2 221.4,70.8 228.3,70.2 235.2,69.3 242.1,67.6 249.1,67.4 256.0,66.9 262.9,67.7 269.9,66.6 276.8,64.5 283.7,62.9 290.6,65.0 297.6,63.6 304.5,62.6 311.4,63.4 318.3,61.7 325.3,60.8 332.2,61.8 339.1,61.4 346.0,62.9 353.0,61.6 359.9,62.5 366.8,60.1 373.8,61.3 380.7,61.8 387.6,64.4 394.5,59.7 401.5,60.5 408.4,61.2 415.3,59.5 422.2,58.0 429.2,59.0 436.1,59.9 443.0,58.8 450.0,57.8 456.9,57.6 463.8,60.0 470.7,60.1 477.7,60.7 484.6,60.7 491.5,57.4 498.4,60.4 505.4,58.7 512.3,62.2 519.2,61.2 526.1,59.1 533.1,59.0 540.0,58.4 546.9,60.0 553.9,60.0 560.8,59.7 567.7,58.3 574.6,55.6 581.6,59.9 588.5,58.6 595.4,58.6 602.3,57.9 609.3,58.9 616.2,59.9 623.1,60.0 630.1,58.4 637.0,58.6 643.9,57.9 650.8,60.0 657.8,55.6 664.7,58.9 671.6,58.0 678.5,56.0 685.5,57.5 692.4,57.2 699.3,57.6 706.2,58.9 713.2,54.1 720.1,58.0 727.0,58.3 734.0,58.1 740.9,58.3" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linejoin="round"/>
<polyline points="52.0,293.7 52.7,309.1 53.4,304.6 54.1,297.6 54.8,307.8 55.5,317.2 56.2,308.9 56.8,304.9 57.5,304.2 58.2,308.5 58.9,102.7 59.6,140.0 60.3,144.5 61.0,160.9 61.7,151.7 62.4,165.8 63.1,163.2 63.8,160.6 64.5,156.8 65.2,158.8 65.9,155.3 66.5,151.9 67.2,148.1 67.9,156.8 68.6,146.5 69.3,149.0 70.0,139.4 70.7,142.1 71.4,133.7 72.1,138.8 72.8,144.6 73.5,141.2 74.2,133.5 74.9,129.5 75.6,130.6 76.2,131.1 76.9,132.5 77.6,125.5 78.3,128.2 79.0,122.0 82.8,114.3 89.8,103.8 96.7,98.6 103.6,90.5 110.5,86.1 117.5,84.4 124.4,78.0 131.3,75.9 138.2,71.3 145.2,67.5 152.1,66.3 159.0,65.3 165.9,61.6 172.9,62.5 179.8,60.6 186.7,58.5 193.7,60.6 200.6,59.2 207.5,54.1 214.4,55.1 221.4,55.0 228.3,51.5 235.2,56.0 242.1,53.3 249.1,52.5 256.0,49.4 262.9,51.3 269.9,50.2 276.8,48.8 283.7,47.0 290.6,48.8 297.6,48.4 304.5,47.7 311.4,47.3 318.3,46.4 325.3,44.8 332.2,41.9 339.1,43.7 346.0,46.6 353.0,44.1 359.9,45.0 366.8,43.8 373.8,42.2 380.7,42.9 387.6,42.5 394.5,44.1 401.5,41.5 408.4,42.8 415.3,43.6 422.2,43.0 429.2,42.4 436.1,39.9 443.0,40.3 450.0,39.2 456.9,38.9 463.8,41.7 470.7,38.9 477.7,38.2 484.6,38.0 491.5,39.5 498.4,36.9 505.4,38.9 512.3,36.5 519.2,37.4 526.1,37.5 533.1,38.7 540.0,37.2 546.9,38.1 553.9,37.7 560.8,37.5 567.7,38.3 574.6,37.8 581.6,39.9 588.5,37.3 595.4,35.6 602.3,37.6 609.3,37.4 616.2,36.5 623.1,37.1 630.1,37.5 637.0,34.1 643.9,37.4 650.8,36.2 657.8,37.0 664.7,35.1 671.6,32.7 678.5,33.9 685.5,36.7 692.4,36.5 699.3,39.7 706.2,36.0 713.2,35.2 720.1,35.6 727.0,37.2 734.0,34.0 740.9,36.2" fill="none" stroke="#6366f1" stroke-width="2" stroke-linejoin="round"/>
<text x="398" y="376" text-anchor="middle" fill="currentColor" fill-opacity="0.7">Steps</text>
<text transform="translate(14 177) rotate(-90)" text-anchor="middle" fill="currentColor" fill-opacity="0.7">Average reward</text>
<text x="466.9" y="191.5" text-anchor="start" fill="#6366f1" font-weight="600">UCB, c = 2</text>
<text x="466.9" y="220.5" text-anchor="start" fill="#10b981" font-weight="600">UCB, c = 1</text>
<text x="466.9" y="249.4" text-anchor="start" fill="#f59e0b" font-weight="600">ε-greedy, ε = 0.1</text>
<line x1="58.9" y1="126.3" x2="58.9" y2="238.6" stroke="currentColor" stroke-opacity="0.45"/>
<text x="67.9" y="249.4" text-anchor="start" fill="currentColor" font-weight="600">step 11</text>
</svg>
<figcaption>Average reward per step on the 10-armed testbed, 2000 runs. UCB pays for a ten-step sweep, spikes at step 11, and settles above ε-greedy.</figcaption>
</figure>

Beyond step 40 the curves are averaged over blocks of ten steps, or the spike
would be lost in per-step noise. The numbers:

| Step | UCB $c=2$ | UCB $c=1$ | $\varepsilon$-greedy | UCB $c=2$ % optimal | $\varepsilon$-greedy % optimal |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.06 | 0.06 | 0.02 | 11.1 | 10.7 |
| 5 | −0.02 | −0.02 | 0.55 | 10.1 | 21.8 |
| 10 | −0.03 | −0.03 | 0.79 | 10.2 | 28.6 |
| **11** | **1.11** | **1.11** | 0.84 | **42.1** | 28.9 |
| 12 | 0.90 | 1.03 | 0.85 | 32.3 | 30.4 |
| 20 | 0.80 | 1.15 | 0.93 | 34.4 | 35.1 |
| 50 | 1.14 | 1.42 | 1.10 | 50.4 | 46.1 |
| 100 | 1.23 | 1.43 | 1.21 | 60.9 | 55.4 |
| 500 | 1.44 | 1.51 | 1.36 | 78.9 | 75.5 |
| 1000 | 1.48 | 1.51 | 1.34 | 86.8 | 80.8 |

UCB pays for its first ten steps: the forced sweep earns an average reward of
−0.01 while $\varepsilon$-greedy is already exploiting. It draws level at around
step 60 on average reward (step 30 if you score it on percent optimal action) and
stays ahead from there, finishing at 1.48 against 1.34 and playing the optimal
arm 87% of the time against 81%. The
advantage is structural: $\varepsilon$-greedy keeps spending 10% of every step on
a uniformly random arm forever, while UCB's exploration is concentrated on arms
that are still genuinely in doubt and thins out as the doubt does.

Note also that $c = 1$ beats $c = 2$ over this horizon (mean reward 1.47 against
1.38 across the 1000 steps), and that the gap has nearly closed by step 1000 as
the $c = 2$ curve is still climbing. This is the same horizon argument as
$\varepsilon$: more exploration costs more now and is worth more later, and the
best setting is a function of how long you get to play.

## The spike at step 11

The early part of the curve has a feature that looks like a plotting artefact and
is not — this is exercise 2.8. Here are the first 40 steps unsmoothed:

<figure class="chart">
<svg viewBox="0 0 760 340" role="img" aria-label="The first 40 steps: the UCB spike at step 11, unsmoothed" style="width:100%;height:auto;font:13px system-ui,sans-serif">
<title>The first 40 steps: the UCB spike at step 11, unsmoothed</title>
<line x1="52" y1="261.9" x2="744" y2="261.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="265.9" text-anchor="end" fill="currentColor" fill-opacity="0.7">0</text>
<line x1="52" y1="166.5" x2="744" y2="166.5" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="170.5" text-anchor="end" fill="currentColor" fill-opacity="0.7">0.5</text>
<line x1="52" y1="71.2" x2="744" y2="71.2" stroke="currentColor" stroke-opacity="0.15"/>
<text x="44" y="75.2" text-anchor="end" fill="currentColor" fill-opacity="0.7">1</text>
<line x1="52.0" y1="300" x2="52.0" y2="305.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="52.0" y="320.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">1</text>
<line x1="229.4" y1="300" x2="229.4" y2="305.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="229.4" y="320.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">11</text>
<line x1="389.1" y1="300" x2="389.1" y2="305.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="389.1" y="320.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">20</text>
<line x1="566.6" y1="300" x2="566.6" y2="305.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="566.6" y="320.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">30</text>
<line x1="744.0" y1="300" x2="744.0" y2="305.0" stroke="currentColor" stroke-opacity="0.4"/>
<text x="744.0" y="320.0" text-anchor="middle" fill="currentColor" fill-opacity="0.7">40</text>
<line x1="52" y1="300" x2="744" y2="300" stroke="currentColor" stroke-opacity="0.45"/>
<line x1="52" y1="14" x2="52" y2="300" stroke="currentColor" stroke-opacity="0.45"/>
<polyline points="52.0,251.3 69.7,267.4 87.5,262.7 105.2,255.3 123.0,266.2 140.7,276.0 158.5,267.2 176.2,263.1 193.9,262.4 211.7,266.8 229.4,50.2 247.2,65.9 264.9,63.6 282.7,67.0 300.4,57.1 318.2,64.6 335.9,55.4 353.6,50.7 371.4,39.1 389.1,43.0 406.9,50.4 424.6,35.3 442.4,33.4 460.1,36.9 477.8,30.3 495.6,32.0 513.3,24.6 531.1,27.4 548.8,19.9 566.6,23.9 584.3,32.2 602.1,27.7 619.8,18.1 637.5,15.0 655.3,19.6 673.0,22.1 690.8,20.2 708.5,10.4 726.3,12.2 744.0,15.9" fill="none" stroke="#10b981" stroke-width="2" stroke-linejoin="round"/>
<polyline points="52.0,257.2 69.7,213.9 87.5,193.6 105.2,158.9 123.0,156.2 140.7,143.9 158.5,126.3 176.2,119.7 193.9,107.0 211.7,111.3 229.4,101.7 247.2,100.7 264.9,104.1 282.7,91.6 300.4,91.8 318.2,99.4 335.9,83.9 353.6,78.7 371.4,79.2 389.1,84.8 406.9,80.3 424.6,68.0 442.4,72.5 460.1,75.1 477.8,72.1 495.6,68.2 513.3,78.9 531.1,72.4 548.8,66.5 566.6,75.8 584.3,69.4 602.1,64.6 619.8,65.4 637.5,62.6 655.3,55.2 673.0,63.8 690.8,65.4 708.5,56.4 726.3,70.6 744.0,60.4" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linejoin="round"/>
<polyline points="52.0,251.3 69.7,267.4 87.5,262.7 105.2,255.3 123.0,266.2 140.7,276.0 158.5,267.2 176.2,263.1 193.9,262.4 211.7,266.8 229.4,50.2 247.2,89.4 264.9,94.1 282.7,111.5 300.4,101.8 318.2,116.6 335.9,113.9 353.6,111.1 371.4,107.2 389.1,109.3 406.9,105.6 424.6,102.0 442.4,98.0 460.1,107.1 477.8,96.3 495.6,99.0 513.3,88.8 531.1,91.6 548.8,82.8 566.6,88.1 584.3,94.3 602.1,90.7 619.8,82.7 637.5,78.4 655.3,79.6 673.0,80.1 690.8,81.5 708.5,74.2 726.3,77.0 744.0,70.5" fill="none" stroke="#6366f1" stroke-width="2" stroke-linejoin="round"/>
<text x="398" y="336" text-anchor="middle" fill="currentColor" fill-opacity="0.7">Steps</text>
<text transform="translate(14 157) rotate(-90)" text-anchor="middle" fill="currentColor" fill-opacity="0.7">Average reward</text>
<text x="477.8" y="193.2" text-anchor="start" fill="#6366f1" font-weight="600">UCB, c = 2</text>
<text x="477.8" y="218.0" text-anchor="start" fill="#10b981" font-weight="600">UCB, c = 1</text>
<text x="477.8" y="242.8" text-anchor="start" fill="#f59e0b" font-weight="600">ε-greedy, ε = 0.1</text>
<circle cx="229.4" cy="50.2" r="4" fill="none" stroke="currentColor" stroke-opacity="0.6"/>
</svg>
<figcaption>The first 40 steps, unsmoothed. The step-11 peak is identical for both values of <em>c</em>; only the trough after it differs.</figcaption>
</figure>

Average reward sits near zero for ten steps, jumps to 1.11 at step 11, drops back
to 0.90 at step 12, and sags further before recovering. Averaging over 2000 runs
removes *noise*, not *structure*: UCB's action sequence is a deterministic
function of the rewards, and its exploration is locked to the step number rather
than scattered in time by a coin flip, so every run does the same thing at the
same step and the feature survives averaging. Two separate mechanisms produce
the rise and the fall.

### Why the reward rises at step 11

**Steps 1–10 are a forced sweep.** Every arm starts with $N_t(a) = 0$ and counts
as maximal, so the first ten steps try all ten arms in an arbitrary order. Each
step therefore lands on a uniformly random arm, and the average reward is the
average of $q_*$ over all arms: zero, measured at −0.01.

**At step 11 the bonus cancels.** Every arm now has $N_{11}(a) = 1$ and $\ln t$
is shared, so the bonus is the *same constant* $c\sqrt{\ln 11}$ for all ten arms.
It shifts every entry in the bracket equally and drops out of the $\arg\max$
entirely. Step 11 is a purely greedy choice, and with one sample per arm
$Q_{11}(a)$ is just the single reward that arm returned:

$$
A_{11} = \arg\max_a \left[ Q_{11}(a) + c\sqrt{\tfrac{\ln 11}{1}} \right] = \arg\max_a R^{(1)}(a)
$$

In the simulation this identity holds in 100% of runs, as it must.

**That choice is much better than chance.** The arm with the largest of the ten
first rewards is the optimal arm 42.1% of the time rather than 10%, and its true
value averages $\mathbb{E}[q_*(A_{11})] = 1.08$. The reward collected at step 11
is a fresh draw from that arm, so the average lands at 1.11. The spike is the
payoff from the first complete comparison of all $k$ arms — the sweep bought ten
samples, and step 11 is the first step allowed to use them.

### Why the reward falls at step 12 and after

**The winner's bonus collapses.** The arm taken at step 11 now has $N = 2$ while
the other nine still have $N = 1$, and $\ln t$ has gone up slightly for everyone.
The gap it must overcome to be repeated is

$$
c\sqrt{\ln 12} - c\sqrt{\tfrac{\ln 12}{2}} = c\sqrt{\ln 12}\left(1 - \tfrac{1}{\sqrt 2}\right) \approx 0.46\,c
$$

which is 0.92 at $c = 2$. Its $Q$ must lead the runner-up by more than that or
UCB is obliged to go elsewhere. It only does so in 18.4% of runs.

**And its estimate regresses.** The arm was selected precisely for having the
largest of ten noisy samples, so its $Q$ is biased *above* its true value. The
second pull averages in an unbiased draw and pulls the estimate back down. The
step-11 winner loses on both terms at once, while every rival gains slightly on
the bonus.

**So step 12 is nearly forced onto a worse arm** — one that just lost the
step-11 comparison — and the average drops to 0.90. Steps 12 to 20 continue down
the ranking, working through the arms UCB already believes are inferior, and
average 0.82: below the spike, and for a while below $\varepsilon$-greedy, which
is free to keep exploiting its favourite. Faint ripples with period near $k$
follow, each new sweep producing another re-ranking, damping out as the pull
counts desynchronise and the time-locking that let the structure survive
averaging breaks down.

The spike is therefore a *contrast* effect: step 11 is the one step where UCB is
allowed to be purely greedy, and step 12 is a step where it is nearly forbidden
from repeating itself.

### What $c = 1$ shows

Setting $c = 1$ makes the spike less prominent, and how it does so pins the
explanation down:

| | $c = 2$ | $c = 1$ |
| --- | --- | --- |
| Average reward, step 11 | 1.110 | 1.110 |
| Average reward, step 12 | 0.904 | 1.028 |
| Step 12 repeats the step-11 arm | 18.4% | 37.4% |
| Ratio $R_{11} / R_{12}$ | 1.23 | 1.08 |

**The peak does not move at all.** It cannot: the bonus cancels at step 11 for
*any* $c$, so step 11 is the same greedy choice in both runs and collects the
same 1.11. Whatever makes the spike less prominent is not acting on the peak.

**The trough is what changes.** Halving $c$ halves the switching threshold from
0.92 to 0.46, so the step-11 winner is retained twice as often and step 12 falls
only to 1.03. With the floor raised and the peak fixed, the spike flattens into
the curve. Push $c$ toward zero and the sweep is followed by ordinary greedy
behaviour with no discontinuity at all.

That is the cleanest confirmation available that the fall is caused by the bonus
term forcing a switch, and not by anything about the peak itself.

## What is wrong with it

**Nonstationary problems.** $\ln t$ grows over the whole history, so an arm
pulled heavily long ago is slow to be re-examined even if the world has since
changed. UCB's confidence intervals assume the thing being estimated holds still.

**Large state spaces.** UCB needs a visit count $N_t(a)$ per action. In a bandit
that is $k$ counters; with states and function approximation there is no
equivalent, and the count-based bonus has to be replaced by something learned —
pseudo-counts, ensemble disagreement, random network distillation.

**$c$ has to match the reward scale.** The bonus is in reward units. Rescale the
rewards by 100 and the same $c$ explores essentially not at all.

**The first $k$ steps are spent regardless.** On a problem with thousands of
arms, the mandatory sweep alone may exhaust the budget.

## Common mistakes

**Using the per-arm count where $\ln t$ belongs.** $t$ is the total step count.
With $N_t(a)$ in both places the bonus becomes $c$ and never decays relative to
anything, and the method stops working.

**Forgetting the $N_t(a) = 0$ case.** In code, $\ln t / 0$ is $\infty$ or a
divide-by-zero depending on the language. Handle untried arms explicitly, and
break the resulting ten-way tie *randomly* — an index-ordered `argmax` makes the
opening sweep run in arm order, which is harmless here and a real bug elsewhere.

**Reading the spike as noise.** More runs make it sharper, not smoother. It is
the algorithm.

**Expecting the ranking at step 11 to mean much.** It is a comparison of ten
single samples. It is far better than chance and still wrong 58% of the time.

## Personal takeaways

What makes UCB feel like a real advance rather than another heuristic is that the
exploration is *derived* rather than chosen. $\varepsilon$ is a number you pick;
$c\sqrt{\ln t / N_t(a)}$ is what a confidence interval already implies. The
method does not have an exploration policy bolted onto a greedy policy — it is
greedy, with respect to a quantity that happens to reward ignorance.

The step-11 exercise is worth the time it takes, because the useful habit it
teaches is not about bandits. A feature that survives averaging over thousands of
runs is telling you something about the mechanism, and the way to find out what
is to ask which step of the algorithm is time-locked. Here the answer was the one
step where the exploration term cancels — and the $c = 1$ check separates the
cause from the coincidence, because it moves the trough while leaving the peak
untouched.

## Where next

That is as far as the bandit problem goes here. The next chapter puts back
everything this one deleted — state, transitions, and consequences that arrive
later than the action that caused them:
[Markov decision processes](/Reinforcement-Learning/chapters/03-markov-decision-processes/).

## References

- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., §2.7, figure 2.4, and exercise 2.8.
- Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning* 47, 235–256.
