# Implementation brief: Chapter 10 interactive lab (GPU admission control)

You are working in the `Reinforcement-Learning` repo, an Astro 5 + Starlight book of RL learning notes published to GitHub Pages. Read `README.md` first: it defines the demo architecture, the base-path rule, the page template and the testing style. Follow those conventions; this brief only adds what is specific to Chapter 10.

**Goal.** Build **one single-page application** containing 18 interactive modules that teach Sutton & Barto Chapter 10, *On-policy Control with Approximation*, through one running example: **admission control for a GPU inference cluster whose requests have a continuous job length**. Embed the app in the chapter page, section by section, and also make it available full screen.

**Audience.** A graduate student reading the chapter. Each module makes one idea *visible* and carries a short "what to observe" telling the reader what to try and what they should see.

---

## 0. Ground rules

- **README conventions are mandatory:** the app is a standalone Astro page under `src/pages/demos/`, logic lives in `src/demos/`, simulation cores are pure (no DOM) and take a seeded RNG from `src/demos/shared/rng.ts`, every demo link goes through `src/lib/url.ts` / `DemoFrame.astro`, no CDNs, **no new runtime dependencies** (no UI framework, no router library, no chart library). Charts are canvas, built on `src/demos/shared/chart.ts`.
- TypeScript must be **erasable syntax only** (no `enum`, no parameter properties, no namespaces): tests import `.ts` directly with Node 24 type stripping, as `tests/baird.test.mjs` does.
- **Do not fake results.** Every number shown is computed by the simulator or the reference solvers. If an expected phenomenon (e.g. policy chattering) does not appear, search parameters honestly; if it still doesn't appear, show what does happen and say so.
- **Label references honestly.** The continuous environment has no closed-form optimum. References come from a bucketed model (section 4) or from Monte Carlo with confidence intervals, and the UI must say which, e.g. "r\* ≈ 2.41 (exact on 3-bucket model)".
- **Copyright:** do not copy text or figures from the textbook. Paraphrase, cite section and exercise numbers, draw your own figures.
- All text is in English.
- Finish with `npm run check`, `npm run build` (runs the link validator) and `node --test tests/` all passing.

---

## 1. Fix the chapter slug first

`astro.config.mjs` lists chapter 10 as `['Deep Q-networks', '10-deep-q-networks']`, but the directory is `src/content/docs/chapters/10-On-policy-Control-Approximation/`, so the chapter is missing from the sidebar.

1. `git mv` the directory to `10-on-policy-control-approximation`.
2. Change the array entry to `['On-policy control with approximation', '10-on-policy-control-approximation']`.
3. Update the chapter-10 row of the README table.
4. `npm run build` and confirm the chapter appears in the sidebar.

The demo route slug (unprefixed, per README) is **`on-policy-control-approximation`**.

---

## 2. The running example

A cluster of `K` GPUs serves inference requests from four tiers. Every request also has a **continuous job length** `ℓ` (expected runtime in steps). Each step, the request at the head of the queue is **accepted** (takes one free GPU and pays a flat per-request tier price) or **rejected** (pays 0). The queue never empties.

Because the price is flat but occupancy grows with `ℓ`, a request's **revenue density** `pay_c / ℓ` is what matters: a long, cheap job ties up a GPU for little money. The optimal decision is therefore a boundary over (free GPUs, tier, job length), which a lookup table cannot represent without discretizing `ℓ`. This is why the chapter needs function approximation.

### Parameters (defaults in bold)

| Symbol | Meaning | Default | Range |
|---|---|---|---|
| `K` | GPUs | **10** | 4–16 (reference solvers); up to 32 in simulation only |
| `pay` | price per request: Free, Basic, Pro, Enterprise | **[1, 2, 4, 8]** | fixed |
| `base` | base tier mix | **[0.25, 0.25, 0.25, 0.25]** | editable |
| `ℓ` | job length: LogNormal with median `m`, log-std `σ`, clipped to `[2, 120]` | **m = 16, σ = 0.6** | m 4–40, σ 0–1.2 |
| `κ` | energy cost per busy GPU per step | **0** | 0–0.5 |
| `P` | steps per day (15 minutes each) | **96** | fixed |
| `A` | diurnal amplitude of the tier mix | **0** | 0–1 |
| `H` | shift length for the episodic variant | **48** | 24–96 |

**σ = 0 recovers Example 10.2 exactly:** every job has `ℓ = 1/0.06`, so every busy GPU frees with probability 0.06 per step. Use `m = 1/0.06`, `σ = 0` for the book preset.

Diurnal tier mix: `d(φ) = −cos(2πφ)` (−1 at midnight, +1 at noon), `dir = [−1, −0.5, +0.5, +1]`, `w_c(φ) = base_c · (1 + A · dir_c · d(φ))`, normalized.

### Simulator state and one step

True state: `K` slots, each free or holding a job with completion probability `q_i = 1/ℓ_i`; the head request `(c, ℓ)`; the clock `τ ∈ {0..P−1}`, phase `φ = τ/P`.

One step, in this exact order (identical in the simulator and the bucketed model):

1. If `a = accept` and a slot is free: put the job in a free slot with `q = 1/ℓ`, `r = pay[c]`. Otherwise `r = 0` (accepting with no free GPU is coerced to reject).
2. `r ← r − κ · (busy slots)`.
3. Each busy slot completes independently with its own `q_i`. A job admitted this step may complete this step.
4. `τ ← (τ + 1) mod P`.
5. Draw the next head request: `c ~ w(φ(τ))`, `ℓ ~ LogNormal(ln m, σ)` clipped, independently of `c`.

**Continuing task** by default. **Episodic variant ("one shift"),** used by M03–M07: starts with all GPUs free at `τ = 0`, ends after `H` steps, undiscounted. Time remaining is part of the state.

### What the agent observes

Agents never see the per-slot list. They see the observation

`o = (g, c, ℓ_head, ρ, φ)`

where `g` is free GPUs and `ρ = Σ_busy q_i` is the expected number of completions next step (a continuous summary of how soon capacity returns). The observation is a lossy summary of the true state. Point this out in M02 and M13: it is exactly the situation of Section 10.4, where feature vectors may not fully distinguish states.

### Policies

- **Threshold policies** `θ = (θ_0..θ_3, Λ_0..Λ_3)`: accept tier `c` iff `g ≥ θ_c` and `ℓ ≤ Λ_c` (`θ_c ∈ {1..K+1}`, `K+1` = never; `Λ_c ∈ (0, ∞]`). Used as baselines and as the candidate family in M08, M14 and M16.
- **Baselines:** accept-all, Enterprise-only, a hand-tuned threshold, and the bucketed optimum (section 4).
- **Learned:** greedy or ε-greedy with respect to `q̂(o, a)`. Ties are broken uniformly at random with the seeded RNG; this matters for M05.

---

## 3. App architecture: one SPA, embedded section by section

### Route and code layout

```text
src/pages/demos/on-policy-control-approximation/
  lab.astro                 # the SPA shell: layout, design tokens, nav; mounts the app
src/demos/gpu-cluster/
  app/
    main.ts                 # boot: read URL, create store, start router
    router.ts               # tiny hash router: #/, #/m13, #/section/<id>
    store.ts                # shared state + subscribe; config, cached references
    registry.ts             # module metadata (id, section, title, loader)
    shell.ts                # nav, section tabs, settings drawer, embed mode
  core/
    env.ts                  # continuous simulator; pure
    buckets.ts              # job-length buckets and their completion probabilities
    model.ts                # bucketed exact model: state indexing, transitions
    exact.ts                # policy evaluation, μ_π, r(π), v^γ, differential values, RVI, finite-horizon DP
    montecarlo.ts           # long-run and rollout estimates with confidence intervals
    toy-chains.ts           # small Markov reward processes for M09, M10, M12
    tiles.ts                # tile coder (port of tiles3: IHT, tiles, tileswrap)
    features.ts             # tabular (σ = 0 only) and tile-coded features of (o, a)
    agents.ts               # episodic n-step Sarsa; differential n-step Sarsa; discounted Sarsa
    policies.ts             # threshold, accept-all, greedy/ε-greedy wrappers
    evaluate.ts             # map a learned greedy policy onto the bucketed model and score it exactly
    stats.ts
  modules/
    m01-playground.ts … m18-concept-check.ts   # each exports mount(el, ctx): () => void
  worker.ts                 # one shared module Worker for heavy jobs
src/demos/shared/
  heatmap.ts                # NEW: canvas grid/heatmap with legend and hover readout
  bars.ts                   # NEW: canvas bar chart / histogram
```

### SPA requirements

- **Routing:** hash-based (`#/`, `#/m13`, `#/section/average-reward`), so the app works under the GitHub Pages base path and inside an iframe. Back/forward and deep links work. `#/` is a home grid of all 18 modules grouped by section.
- **Embed mode via query string:** `?embed=1` hides the global nav and shows only a compact tab bar; `?section=<id>` limits the tab bar to that section's modules; `?module=<id>` sets the starting module.
- **Code splitting:** each module is loaded with dynamic `import()` from `registry.ts`, so opening one section does not download all 18 modules.
- **Mount contract:** `mount(el, ctx)` returns an unmount function that **must** stop animation loops, remove listeners and cancel worker jobs. Navigating between modules must not leak loops (verify by switching modules repeatedly with the profiler open).
- **Shared state (`store.ts`):** one cluster configuration (K, m, σ, κ, A, base mix, seed) editable from a settings drawer available in every module; modules subscribe and restart when it changes. Cache reference solutions (RVI, μ_π, DP) keyed by a hash of the config, so moving between modules does not recompute them. A trained agent from M13 can be reused by M11, M15 and M17 ("use the agent from the lab" button).
- **Persistence:** keep the config and current module in `localStorage`, wrapped in try/catch; the app must work with storage blocked.
- **Design tokens:** reuse the token set from `src/pages/demos/function-approximation/baird.astro` (light/dark variables, `[data-theme]` overrides, card/control/button styles) in the shell's global style. Do not modify existing demos.
- **Performance:** run interactive training on the main thread in time-sliced chunks (≤ 8 ms per animation frame) with Play / Pause / Step / ×10 / Fast-forward. Sweeps, RVI on larger models, finite-horizon DP and Monte Carlo references run in `worker.ts`, with a progress bar and cancellation. Downsample chart series. Respect `prefers-reduced-motion` (no auto-play). Layout stacks to one column under 760 px.
- **Auto-height when embedded:** the app posts `{ type: 'gpu-lab-height', height }` to `window.parent` whenever its content height changes (ResizeObserver).

### Embedding in the chapter

- Extend `DemoFrame.astro`, **backwards compatibly**: optional `query` prop (appended to the URL built by `demoUrl`) and optional `autoHeight` prop. When `autoHeight` is set, a small inline script listens for `gpu-lab-height` messages, checks `event.source === iframe.contentWindow`, and sets the iframe height (with min and max bounds). Existing embeds must render exactly as before.
- Convert the chapter's `index.md` to `index.mdx` (keep its frontmatter and content). After each section's summary, embed the app for that section:

  ```mdx
  <DemoFrame chapter="on-policy-control-approximation" demo="lab"
    query="embed=1&section=average-reward" autoHeight
    title="Average-reward modules" source="src/demos/gpu-cluster/" />
  ```

  Section ids: `problem` (M01–M02, placed after the Overview), `episodic` (M03–M05), `n-step` (M06–M07), `average-reward` (M08–M13), `discounting` (M14–M17), `review` (M18, placed after Key takeaways).
- Add a short "Interactive lab" paragraph near the top of the page with a full-screen link (through `DemoFrame`/`url.ts`, never a hand-written path).
- Add one card for the lab to `SimulationGallery.astro`, with its own thumbnail branch in the existing SVG style.

---

## 4. Shared algorithms

### Job-length buckets and the bucketed model (`buckets.ts`, `model.ts`)

The continuous environment is the one agents learn in. For exact references, use a **bucketed model**:

- Split the clipped LogNormal into `B` equal-probability buckets (default `B = 3`: short, medium, long; `B = 1` when σ = 0). Bucket `b` gets completion probability `q_b = E[1/ℓ | ℓ ∈ bucket b]`, computed by numerical integration, and a representative length `1/q_b`.
- Exact state: busy counts per bucket `(n_1..n_B)` with `Σ n_b ≤ K`, head tier `c`, head bucket `b`. Count: `C(K+B, B) · 4 · B` (3,432 at K = 10, B = 3; 44 when σ = 0, matching Example 10.2).
- Transitions: after admission, each bucket loses `Binomial(n_b, q_b)` jobs independently; next head tier from `w(φ)`, next head bucket uniform.
- The simulator has `mode: 'continuous' | 'bucketed'`. In bucketed mode each sampled `ℓ` is snapped to its bucket's `q_b`, so simulation and model agree exactly (tested).
- **Observation on bucketed states:** `g = K − Σ n_b`, `ℓ_head = 1/q_b`, `ρ = Σ n_b q_b`. This lets `evaluate.ts` turn any learned greedy policy into a policy on the bucketed model and **score it exactly**; modules report both the exact bucketed score and the simulated continuous average reward.
- Reference solvers run on the bucketed model with the clock omitted, so they require `A = 0`. With `A > 0`, references come from `montecarlo.ts` (long runs, 95% CIs), and the UI says so.

### Tile coder (`tiles.ts`, `features.ts`)

Port Sutton's `tiles3` (index hash table, `tiles`, `tileswrap`) to TypeScript. Default feature groups, each with integer inputs `[c, a]`:

| Group | Inputs (each scaled to `[0, 4)`) | Tilings |
|---|---|---|
| G1 | `g/K`, `z = normalized log ℓ_head` | 8 |
| G2 | `g/K`, `ρ / ρ_max` | 4 |
| G3 (only when A > 0) | `g/K`, `φ` **with wrap** | 4 |

IHT size 8192. Step sizes are always shown to the reader **as α × number of active features**, as the book does with "α × number of tilings". Tabular one-hot features over `(g, c)` are available only when σ = 0.

### Agents (`agents.ts`)

- **Episodic n-step semi-gradient Sarsa** (10.1–10.2): ring buffers indexed mod `n+1`, following the chapter's n-step pseudocode. With `n ≥ H` it must equal the Monte Carlo target.
- **Differential n-step semi-gradient Sarsa** (10.3, 10.5) with an `R̄` rule option:
  - `'td'` (default): `R̄ ← R̄ + βδ`
  - `'reward'`: `R̄ ← R̄ + β(R − R̄)`
  - `'sample-average'`
  - `'unbiased'` (Exercise 10.9): `ō ← ō + β(1 − ō)`, `ō₀ = 0`, step size `β/ō`, applied to δ
- **Discounted continuing semi-gradient Sarsa** with γ, for M15.
- Every update returns a trace record (features, q̂ before, target, δ, Δw per active feature, R̄ before/after), as `baird/core.ts` does, so steppers display numbers without recomputing them.
- Initial value `q₀`: every weight set to `q₀ / activeFeatures`.

### Reference solvers (`exact.ts`, `montecarlo.ts`)

- **Policy evaluation on the bucketed model:** `μ_π` by power iteration with Cesàro averaging; `r(π) = Σ μ_π(s) E[r | s, π]`; discounted `v^γ_π` (including γ = 0); differential values `h_π` normalized so `Σ μ_π h_π = 0` (the normalization produced by formula 10.13).
- **Relative value iteration** for `r*`, the optimal policy and its bias, with the aperiodicity transform `p̃ = (1−η)I + ηp`, `r̃ = ηr`, `η = 0.9` (same optimal policy and bias; `r* = g̃/η`). Iterate to a span tolerance of 1e-10.
- **Finite-horizon DP** over the shift (bucketed model plus time-to-go) for the episodic variant: the optimal shift policy and `q_π` for a fixed policy. Run it in the worker.
- **Monte Carlo** (`montecarlo.ts`): long-run average reward of a policy in the continuous simulator with batch-means 95% CIs; rollout estimates of `q_π(s, a)` for M06.

### Toy chains (`toy-chains.ts`)

Alternating A ↔ B (+1 leaving A; Exercise 10.6); three-state ring with +1 on arrival at A (Exercises 10.7, 10.8); two-state chain with P(A→B) = 0.3, P(B→A) = 0.6 and expected rewards 1 and 5; lazy ring (stay with probability ε); reducible pair of rings with different reward rates.

---

## 5. The 18 modules

Defaults are those of section 2 unless stated. Each module shows its own short "what to observe" text inside the app.

### Section `problem`

**M01 · Playground: you are the policy**
- Visual: K GPU tiles (busy ones colored by tier, with a ring showing expected time left `1/q_i`); the incoming request card with tier, price, job length `ℓ` and revenue density `pay/ℓ`; a clock dial when A > 0; running average reward.
- Controls: Accept / Reject buttons (keys A and R); speed; Autopilot (accept-all, Enterprise-only, threshold sliders, bucketed optimum); K, m, σ, κ, A; seed.
- Scoreboard: your average reward vs. accept-all vs. the reference `r*`.
- Observe: accepting everything loses; the expensive mistake is accepting a long, cheap job when the cluster is nearly full; short cheap jobs are often fine.

**M02 · How the agent sees the cluster (tile coding)**
- Visual: a 2-D grid for a chosen feature group: free GPUs × log job length (G1), free GPUs × ρ (G2), or free GPUs × time of day (G3, wrap on/off). Clicking a point highlights its active tile in each tiling. A heatmap of `q̂(·, c, accept)` for the selected tier.
- "Teach this point": one update toward a chosen target at the clicked point, then watch the heatmap change in a neighborhood.
- A small panel shows two different true states (different per-slot job lists) that produce the same observation: aliasing.
- Controls: tilings, tiles per dimension, wrap, α × active features.
- Observe: updates generalize to neighbors; more tilings give finer resolution at the same generalization width; with wrap, 23:45 and 00:00 share tiles; observations do not uniquely identify states.

### Section `episodic` (10.1)

**M03 · One semi-gradient Sarsa update, fully exposed**
- The transition `(S, A, R, S′, A′)` in the one-shift task; active features per group; `q̂(S, A)` before; target `R + q̂(S′, A′)`; δ; Δw per active weight; `q̂(S, A)` after.
- Toggle "Recompute target after update": the target moved too, because it depends on w. That dependence is what *semi*-gradient ignores.
- Controls: Step, Back (history of records), α, ε.

**M04 · Training over shifts**
- Learning curves (revenue per shift, mean over runs) for three α values; the learned greedy policy per tier as a map over free GPUs × job length at a chosen time in the shift (slider), next to the optimal shift policy from finite-horizon DP (shown as bucket bands).
- Observe: larger α learns faster but noisier; late in the shift both the learned and optimal policies accept almost everything, because occupancy no longer costs anything.

**M05 · Optimism explores even with ε = 0**
- Visitation heatmap (free GPUs × job length), learning curve, fraction of Reject decisions.
- Controls: `q₀` from −100 to 600 with a marker at the DP upper bound on remaining shift revenue; ε including 0.
- Observe: pessimistic `q₀` with ε = 0 locks into accept-all because Reject is never tried; optimistic `q₀` makes every tried action disappoint, so the agent works through the alternatives (the analogue of Figure 10.1's "Step 428" panel).

### Section `n-step` (10.2)

**M06 · Building G_{t:t+n}**
- A trajectory timeline from one state-action pair; slider n highlights the summed rewards and the bootstrap term with the numeric target. Below, histograms of the target for n ∈ {1, 2, 4, 8, 16, ∞} over many sampled trajectories under a fixed policy and a fixed, deliberately imperfect `q̂`, with the Monte Carlo estimate of `q_π(s, a)` (with CI) as a vertical band.
- Observe: small n is tight but biased by `q̂`; large n is centered but wide.

**M07 · Which n learns fastest?**
- Worker sweep, the analogue of Figure 10.4: mean revenue over the first 50 shifts (error bars over runs) vs. α × active features for n ∈ {1, 2, 4, 8, 16}; plus best-α learning curves for n = 1 vs. n = 8 (Figure 10.3 analogue).
- Observe: intermediate n wins; larger n wants smaller α; spread grows with n (Exercise 10.3).

### Section `average-reward` (10.3, 10.5)

**M08 · Where does the cluster spend its time?**
- Exact `μ_π` on the bucketed model shown as stacked bars over free GPUs (stack = busy jobs by length bucket), with the histogram from a continuous simulation converging toward it; a table decomposing `r(π) = Σ μ_π(s) E[r | s]`.
- Controls: threshold-policy sliders; start state (all free / all busy with long jobs); "Apply one step to μ_π" (unchanged: equation 10.8).
- Observe: different starts reach the same distribution; changing the policy changes `μ_π`, and that is how a policy changes long-run reward.

**M09 · When does the limit exist? (ergodicity)**
- For a chosen chain: `Pr{S_t = s}` over time (exact for toys; estimated from 2,000 parallel simulator replicas for the cluster) and the running time-average reward. Badges: irreducible, aperiodic, limiting distribution exists, time-average exists.
- Presets: alternating (Exercise 10.6), ring, lazy ring (slider), reducible pair (start selector), GPU cluster with A = 0.8.
- Observe: periodic chains never settle but their time average converges (ergodicity is sufficient, not necessary); reducible chains depend on the start; any laziness restores convergence.

**M10 · Defining values when the sum oscillates**
- Alternating chain and ring: oscillating partial sums of `R_{t+1} − r(π)`; `f_s(γ) = Σ γ^t (E[R_{t+1} | S_0 = s] − r(π))` as γ → 1 approaching the dashed limits (1/4, −1/4; −1/3, 0, 1/3); `v^γ(s)` and `r(π)/(1−γ)` both diverging while their difference converges. "Show algebra" toggle with the closed form for the chosen chain (geometric sums, common denominator, L'Hôpital).
- Then the cluster: exact differential values from the bucketed model, read as "how much better than average it is to be here", e.g. having k free GPUs when a long Free-tier request arrives.

**M11 · Values are defined up to a constant**
- Exact `h_π` (lines per tier and head bucket vs. free GPUs); a slider adding c to every value while a table of sampled transitions shows TD errors (10.11) and Bellman residuals unchanged. Then learned values (from the M13 agent, mapped onto bucketed observations) vs. exact values as a scatter, with the fitted offset reported.

**M12 · How R̄ should be updated**
- Ring panel (Exercise 10.8): freeze `R̄` at the true 1/3 with exact values; the `R − R̄` errors cycle −1/3, −1/3, +2/3 while δ is exactly 0. Unfreeze and compare `R̄` trajectories.
- Cluster panel (A = 0.8, a fixed policy): `R̄` traces under the four rules vs. the Monte Carlo `r(π)` band; initial `R̄` slider (0–20); β slider; "traffic shift" button doubling the Enterprise share mid-run.
- Observe: δ strips out the predictable part of reward fluctuation; small constant β stays biased by its initial value for a long time; the sample average lags after the shift; the unbiased trick fixes the start and still tracks.

**M13 · The main lab: differential semi-gradient Sarsa**
- Rolling average reward with the reference `r*`; `R̄` trace; the exact bucketed score of the current greedy policy at checkpoints; the learned decision boundary per tier as a map over free GPUs × job length (ρ fixed at its typical value for each g, with a slider to change it), next to the bucketed optimum shown as bands; % agreement on bucketed states weighted by `μ` of the optimal policy; the differential value of the best action per tier vs. free GPUs (Figure 10.5 analogue).
- Controls: features (tabular when σ = 0; tile groups and their settings), α, β, ε, n (Section 10.5), R̄ rule, K, m, σ, κ, A; Play; Fast-forward to N steps.
- Presets: **"Book Example 10.2"** (σ = 0, tabular, α = β = 0.01, ε = 0.1, 2,000,000 steps); **"Continuous jobs"** (defaults, tile coding); **"Diurnal"** (A = 0.8, G3 on, Monte Carlo reference).
- Observe: the book preset reproduces the threshold shape of Figure 10.5; with continuous job lengths the boundary over `ℓ` shifts with congestion, which tile coding captures by generalizing across nearby lengths; rarely visited states keep values near their initial value.

### Section `discounting` (10.4)

**M14 · γ cannot change the ranking**
- About 20 threshold policies on the bucketed model; two sorted bar charts, by `r(π)` and by `J_γ(π) = Σ μ_π(s) v^γ_π(s)`; a scatter of `J_γ(π)(1−γ)` vs. `r(π)` on the diagonal. Contrast panel: ranking by `v^γ_π(s₀)` from one fixed start state, with any pair whose order swaps as γ moves highlighted (search automatically; if none exists, say so).
- Control: γ from 0 to 0.999, including exactly 0.
- Observe: averaged over the on-policy distribution the ranking never changes, even at γ = 0 where `J = r(π)`; from a single state it can.

**M15 · γ as an algorithm parameter**
- Worker: train discounted Sarsa with γ ∈ {0, 0.5, 0.9, 0.99} and differential Sarsa on identical seeds in the continuous simulator; score every learned greedy policy exactly on the bucketed model; bar chart against `r*`; the five decision-boundary maps.
- Observe: γ = 0 learns immediate reward only and accepts everything; larger γ helps but must be tuned; differential Sarsa needs no γ. The discounted algorithms do not optimize `J`.

**M16 · The derivation, line by line, with numbers**
- Each line of `J(π) = Σμv^γ = r(π) + γ Σ_{s′} v^γ(s′) Σ_s μ(s) Σ_a π(a|s) p(s′|s,a) = r(π) + γJ(π) = r(π)/(1−γ)` rendered with KaTeX, each with its numeric value for the current policy and γ. The invariance step is highlighted; a toggle replaces `μ_π` with a uniform distribution and shows which line breaks.

**M17 · No policy improvement theorem: chattering**
- A raster of the greedy policy over time (rows = a fixed sample of bucketed states, columns = checkpoints, colored Accept/Reject), flips per window, and the exact bucketed `r(π)` of the greedy policy at each checkpoint.
- Controls: coarse tiles vs. default tiles vs. tabular (σ = 0), constant α, ε.
- Observe: with coarse shared features, an update aimed at one state flips decisions in others, and the greedy policy's `r(π)` can go down after an update; tabular settles. Find an honest configuration (ground rules).

### Section `review`

**M18 · Check yourself**
- True/False and numeric cards, each revealing an explanation:
  - All policies attaining the maximal `r(π)` are optimal (true; gain optimality ignores transient differences).
  - With on-policy averaging, the ordering of policies is the same with or without discounting (true, and what it does not claim).
  - γ can be 0 in the objective (true) and in the algorithm (problematic, link to M15).
  - Discounted control with approximation is hard because the policy improvement theorem is lost (true; the gap exists in every setting).
  - The agent's observation `(g, c, ℓ, ρ, φ)` is a Markov state (false; link to M02).
  - `E[R₁ | S₀ = A]`, `E[R₁ | S₀ = B]` for Exercise 10.6 (1, 0); `r(π)` of the two-state chain (7/3); `v(A), v(B)` (1/4, −1/4); ring values (−1/3, 0, 1/3).
- Numeric answers accept fractions or decimals within 1e-3.

---

## 6. Tests (must pass, `node --test tests/`)

Reference math (`tests/gpu-cluster-exact.test.mjs`):
- Alternating chain: `r = 1/2`, values `(1/4, −1/4)`. Ring: `r = 1/3`, values `(−1/3, 0, 1/3)`. Two-state chain: `μ = (2/3, 1/3)`, `r = 7/3`. Tolerance 1e-9.
- σ = 0 gives exactly 44 states and the Example 10.2 transition structure.
- `‖μ_π P_π − μ_π‖∞ < 1e-10` for several policies; `J_γ(π)(1−γ) = r(π)` within 1e-8 for γ ∈ {0, 0.5, 0.9, 0.99}.
- RVI: span Bellman residual < 1e-8; adding a constant to `h` leaves every TD error unchanged; the aperiodicity transform matches plain RVI (after rescaling) on an aperiodic instance.
- Bucket probabilities: `q_b` integrate correctly (compare with high-resolution numerical integration) and sum consistency holds.

Simulator (`tests/gpu-cluster-env.test.mjs`):
- Same seed gives an identical trajectory; accept with no free GPU is coerced to reject.
- Bucketed mode: simulated average reward of a fixed policy over 1,000,000 steps is within 2% of the exact `r(π)`.
- Continuous mode: as B grows (1, 2, 3, 4), the exact bucketed `r(π)` of a fixed threshold policy approaches the Monte Carlo estimate from the continuous simulator; B = 4 lies within 3%.
- On bucketed states, the observation satisfies `ρ = Σ n_b q_b` and `g = K − Σ n_b`.
- Empirical next-tier frequencies match `w(φ)` at several phases.

Tiles and agents (`tests/gpu-cluster-agents.test.mjs`, `tests/gpu-cluster-tiles.test.mjs`):
- Same inputs give the same indices; the number of active features equals the total tilings of enabled groups; with wrap, φ = 0.999 and φ = 0.001 share a tile.
- Episodic n-step Sarsa with `n ≥ H` equals the Monte Carlo target.
- The `td` R̄ rule with frozen exact ring values gives δ = 0 every step; the `reward` rule gives the −1/3, −1/3, +2/3 cycle.
- Book preset (σ = 0, tabular, fixed seed, 2M steps): the exact `r(π)` of the learned greedy policy is at least 97% of `r*`. If it fails, investigate the implementation before touching the threshold, and report what you found.

---

## 7. Order of work

1. Section 1 (slug fix), then build.
2. `buckets.ts`, `model.ts`, `exact.ts`, `toy-chains.ts` and their tests. Everything else is checked against these.
3. `env.ts`, `montecarlo.ts`, `evaluate.ts` and their tests.
4. `tiles.ts`, `features.ts`, `agents.ts` and their tests.
5. SPA skeleton: `lab.astro`, router, store, shell, registry with placeholder modules; `heatmap.ts`, `bars.ts`, `worker.ts`; the `DemoFrame` extension and the section embeds in `index.mdx`. Verify embed mode and auto-height in the built site before writing modules.
6. Modules in this order: M13, M01, M08, M12, M14, then the rest.
7. Gallery card; README notes on the lab's structure (short).
8. `npm run check`, `npm run build`, `node --test tests/`. Open the full-screen app and every section embed at desktop and 380 px widths, in light and dark themes; switch between modules repeatedly and confirm no animation loop or worker job survives an unmount.

When done, report: files added or changed, test results, the reference `r*` for the defaults and for the book preset, and any module where the expected phenomenon was weak or absent, with the parameters you tried.
