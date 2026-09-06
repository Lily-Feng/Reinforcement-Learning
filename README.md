# Reinforcement Learning

A chapter-based reinforcement learning book with explanations, worked simulation examples,
personal takeaways, and interactive demos. The site is built with
[Astro Starlight](https://starlight.astro.build/) and published to GitHub Pages
by GitHub Actions.

[Read the published learning notes](https://lily-feng.github.io/Reinforcement-Learning/).

## Reading order

Chapters run from easy to complex, and the directory name carries the
reading-order number: `02-multi-armed-bandits`. The number is zero-padded so
alphabetical order matches numeric order, and hyphenated because Starlight's
slugifier strips a dot (`2.multi-armed-bandits` would publish as
`/chapters/2multi-armed-bandits/`).

The number reaches the URL, so **it is permanent**. Renumbering a chapter
changes its published address and breaks every existing link to it — prefer
appending a chapter to inserting one. Sidebar labels stay unnumbered; the
`sidebar` array in `astro.config.mjs` still defines the order.

| # | Chapter | Why it sits here |
| - | ------- | ---------------- |
| 1 | `foundations` | Trial and error, optimal control, and temporal difference converge into modern RL. |
| 2 | `multi-armed-bandits` | Exploration vs exploitation with **no state**. The simplest possible RL problem. |
| 3 | `markov-decision-processes` | Introduces state, transitions, value functions, Bellman equations. |
| 4 | `dynamic-programming` | Solves MDPs when the model is **known**. Policy evaluation, policy/value iteration. |
| 5 | `monte-carlo-methods` | First model-free methods. Learns from complete episodes. |
| 6 | `temporal-difference-learning` | Bootstrapping: TD(0), SARSA, Q-learning. The core of modern RL. |
| 7 | `n-step-and-eligibility-traces` | Unifies Monte Carlo and TD along one axis. |
| 8 | `planning-and-learning` | Dyna-Q, learned models, simulated experience. |
| 9 | `function-approximation` | Drops the lookup table. Linear methods, tile coding, semi-gradient TD. |
| 10 | `deep-q-networks` | Neural function approximation, replay buffers, target networks. |
| 11 | `policy-gradient-methods` | Optimizes the policy directly. REINFORCE, baselines, actor-critic. |
| 12 | `advanced-policy-optimization` | A2C, trust regions, PPO. |

Each row is a directory, prefixed with its number: `01-foundations`,
`02-multi-armed-bandits`, and so on.

Demo routes are deliberately **not** numbered. Demos live in `src/pages/`, not
in the chapter collection, and leaving them unprefixed means a demo URL
survives any chapter renumbering.

## Repository structure

```text
Reinforcement-Learning/
├── .github/
│   └── workflows/
│       └── pages.yml
├── public/
│   └── images/
├── src/
│   ├── components/
│   │   └── DemoFrame.astro
│   ├── content/docs/
│   │   ├── index.md
│   │   ├── getting-started.md
│   │   └── chapters/
│   │       ├── 01-foundations/
│   │       │   └── index.md
│   │       ├── 02-multi-armed-bandits/
│   │       │   ├── index.md
│   │       │   ├── action-value-methods.md
│   │       │   ├── epsilon-greedy.mdx
│   │       │   ├── upper-confidence-bound.md
│   │       │   └── gradient-bandit.md
│   │       ├── 03-markov-decision-processes/
│   │       │   └── index.md
│   │       └── ...
│   ├── demos/
│   │   └── bandits/
│   │       ├── core.ts          # pure simulation, no DOM
│   │       └── render.ts        # canvas + controls
│   ├── pages/demos/
│   │   └── multi-armed-bandits/
│   │       └── epsilon-greedy.astro
│   └── styles/
│       └── custom.css
├── experiments/
│   ├── ucb-testbed.mjs       # 10-armed testbed numbers quoted in chapter 2
│   └── gradient-bandit.mjs  # baseline vs no baseline, shifted and standard
├── .nvmrc
├── astro.config.mjs
├── package.json
├── package-lock.json
├── LICENSE
└── README.md
```

## Content and demo organization

Each subject has two layers:

- A **book page** with the explanation, equations, worked examples,
  observations, experiments, and references.
- A **standalone interactive demo**, embedded in the book page and also
  reachable on its own.

For epsilon-greedy:

```text
https://lily-feng.github.io/Reinforcement-Learning/chapters/02-multi-armed-bandits/epsilon-greedy/
https://lily-feng.github.io/Reinforcement-Learning/demos/multi-armed-bandits/epsilon-greedy/
```

### Demos live in `src/pages/`, not `public/`

Files in `public/` are copied verbatim and never touched by the build, so a
demo placed there can never have its styles bundled, its scripts type-checked,
or its links made base-aware. Demos are therefore standalone Astro pages under
`src/pages/demos/`. They still render full-screen with their own layout and
styling — the isolation is preserved — but they participate in the build.

Consequences:

- No `cdn.tailwindcss.com`. That script is explicitly not for production and
  logs a console warning on every load. Styles are authored in the page or in
  `src/styles/` and bundled.
- No CDN icon font. Inline the handful of icons actually used as SVG.
- Demo JavaScript is imported from `src/demos/` and bundled, so it can be
  shared between the standalone page and any embed.

### The base path rule

The site is served from `/Reinforcement-Learning/`, and Astro does **not**
rewrite root-absolute links. A hardcoded `href="/demos/..."` resolves to
`lily-feng.github.io/demos/...` and 404s in production while looking fine in
review.

Every internal link to a demo goes through `import.meta.env.BASE_URL`, and that
logic is written once inside `DemoFrame.astro`, which renders the responsive
iframe plus the **Open demo full screen** and **View source** links. No page
constructs a demo path by hand.

### Demo architecture

Each demo is split into a pure core and a rendering layer:

- `core.ts` — the simulation. Takes explicit parameters
  (`{ arms, epsilon, steps, seed }`), returns results, touches no DOM.
- `render.ts` — canvas drawing, controls, animation.

The core takes a seeded PRNG (mulberry32 is about five lines) rather than
calling `Math.random()` directly. This is not optional polish: reproducible
runs and side-by-side comparison of several epsilon values both require running
the simulation many times with no DOM attached. Retrofitting that after several
demos have copied a DOM-coupled pattern is far more work than starting with it.

## Topic page template

Six required sections, five optional. The required set is small on purpose — a
thin chapter that ships beats a complete one that stalls.

**Required**

1. Learning objective
2. Intuition
3. Algorithm and equations
4. Interactive demo
5. What to observe
6. Personal takeaways

**Optional**

7. Worked example
8. Common mistakes
9. Experiments to try
10. Prerequisites — only when a page depends on something out of reading order
11. References

## Astro configuration

Starlight renders no LaTeX by default, so the math pipeline is configured up
front rather than discovered mid-chapter.

```js
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import starlightLinksValidator from 'starlight-links-validator';

export default defineConfig({
  site: 'https://lily-feng.github.io',
  base: '/Reinforcement-Learning',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      title: 'Reinforcement Learning',
      plugins: [starlightLinksValidator()],
      customCss: [
        './src/styles/custom.css',
        'katex/dist/katex.min.css',
      ],
      // This array is the reading order. Easy to complex, top to bottom.
      sidebar: [
        { label: 'Getting started', link: '/getting-started/' },
        { label: 'Foundations', autogenerate: { directory: 'chapters/01-foundations' } },
        { label: 'Multi-armed bandits', autogenerate: { directory: 'chapters/02-multi-armed-bandits' } },
        { label: 'Markov decision processes', autogenerate: { directory: 'chapters/03-markov-decision-processes' } },
        { label: 'Dynamic programming', autogenerate: { directory: 'chapters/04-dynamic-programming' } },
        { label: 'Monte Carlo methods', autogenerate: { directory: 'chapters/05-monte-carlo-methods' } },
        { label: 'Temporal-difference learning', autogenerate: { directory: 'chapters/06-temporal-difference-learning' } },
        { label: 'n-step and eligibility traces', autogenerate: { directory: 'chapters/07-n-step-and-eligibility-traces' } },
        { label: 'Planning and learning', autogenerate: { directory: 'chapters/08-planning-and-learning' } },
        { label: 'Function approximation', autogenerate: { directory: 'chapters/09-function-approximation' } },
        { label: 'Deep Q-networks', autogenerate: { directory: 'chapters/10-deep-q-networks' } },
        { label: 'Policy gradient methods', autogenerate: { directory: 'chapters/11-policy-gradient-methods' } },
        { label: 'Advanced policy optimization', autogenerate: { directory: 'chapters/12-advanced-policy-optimization' } },
      ],
    }),
  ],
});
```

Page order *within* a chapter is set with `sidebar.order` in each page's
frontmatter; those never appear in a URL. Chapter order is the array above, and
must be kept in step with the directory prefixes — the prefix is what readers
see, the array is what the sidebar renders.

## GitHub Pages deployment

GitHub Actions builds and deploys using the supported Pages artifact workflow:

- Validate pull requests (`astro check`, `astro build`, link validation) without
  deploying.
- Build and deploy on pushes to `main`.
- Support manual runs with `workflow_dispatch`.
- Grant only `contents: read`, `pages: write`, `id-token: write`.
- Use the `github-pages` deployment environment.
- Prevent overlapping deployments with a concurrency group.
- Publish only Astro's generated `dist/`.

Node is pinned in `.nvmrc` (24) and the workflow reads it via
`node-version-file`, so CI cannot drift across Astro-sensitive Node majors.
This is the runtime for `npm`; the Node version the *actions themselves* run on
is set by the action major, which is a separate axis.

`npm run check` runs `astro check`, which covers TypeScript and component
diagnostics but **not** internal links. Link checking comes from the
`starlight-links-validator` plugin above, so a broken cross-chapter link fails
the pull request build.

### Enabling Pages — one manual step, required once

Before the first deploy can succeed, set:

**Settings → Pages → Build and deployment → Source → GitHub Actions**

Until that is done the `build` job fails with:

```text
Get Pages site failed. Please verify that the repository has Pages enabled and
configured to build using GitHub Actions
```

This cannot be automated from inside the workflow. `configure-pages` accepts an
`enablement: true` input, but *creating* a Pages site requires repository admin
rights, and `GITHUB_TOKEN` does not have them even when granted `pages: write`.
Attempting it fails differently, which is a worse error, not a fix:

```text
Create Pages site failed. Error: Resource not accessible by integration
```

If the `deploy` job later fails on permissions rather than on the Pages site,
check **Settings → Actions → General → Workflow permissions** as well.

### Action versions

All actions are pinned to majors that run on Node 24. Node 20 is deprecated on
GitHub runners, and actions targeting it emit a warning and are force-migrated.
When bumping, check that the inputs still exist — `configure-pages` needs
`enablement`, `setup-node` needs `node-version-file` and `cache`,
`upload-pages-artifact` needs `path`, and `deploy-pages` must still expose the
`page_url` output used by the deployment environment.

References:

- [Deploy an Astro site to GitHub Pages](https://docs.astro.build/en/guides/deploy/github/)
- [Use custom workflows with GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [Configure a GitHub Pages publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

## Status

Phases 1-3 are implemented and verified. Phase 4 is the ongoing work.

### Phase 1: Foundation and deployment — done

- Astro Starlight scaffolded; Node pinned in `.nvmrc`.
- `site` / `base` configured for the project subpath.
- Math pipeline (`remark-math` + `rehype-katex` + KaTeX CSS) wired in.
- Sidebar array drives reading order; all 12 chapters navigable, each
  directory prefixed `01-` through `12-`.
- `.github/workflows/pages.yml` validates pull requests and deploys `main`.
- Verified: `npm run check` clean, `npm run build` produces 18 pages, and
  `starlight-links-validator` reports every internal link valid.

### Phase 2: First chapter and demo — done

- `foundations` and `multi-armed-bandits` written; the other ten are outlined
  stubs so the reading order is real from day one.
- The old `multi-armed bandits/epsilon-greedy.html` is gone. Its behaviour now
  lives in `src/demos/` (pure core + render layer) and
  `src/pages/demos/multi-armed-bandits/epsilon-greedy.astro`.
- Tailwind and Font Awesome CDN tags dropped; styles bundled, icons inlined.
  The built output contains zero CDN references.
- `DemoFrame.astro` is the single place a demo URL is constructed.

### Phase 3: Demo quality — done

- Responsive layout, single column below 900px.
- Labelled controls, `aria-live` stats and log, visible focus rings,
  `aria-hidden` on concealed answers, `prefers-reduced-motion` respected.
- Every run is seeded and reproducible.
- Comparison mode sweeps four epsilon values over 150 runs each, headless.
- Estimated value, reward, exploration, exploitation and regret are all
  surfaced in the UI and explained on the page.

### Phase 4: Remaining chapters — ongoing

Work down the reading-order table. For each subject:

1. Add or select the chapter directory, prefixed with its reading-order
   number (`13-…`). Append rather than insert, so existing URLs survive.
2. Add the entry to the `sidebar` array in reading-order position.
3. Copy the topic-page template; fill the six required sections.
4. Add the demo as a pure core plus a rendering layer.
5. Run `npm run check` and `npm run build`.
6. Open a pull request; merging to `main` publishes.

## Verification

The simulation core is pure, so it is checkable without a browser. Two suites
were run against this implementation:

- **Core** (16 checks) — seed reproducibility, the planted best arm being the
  true argmax, $Q \to q_*$ within 0.015 after 20k steps, bookkeeping identities,
  greedy getting stuck on a suboptimal arm in 72% of seeds, $\varepsilon = 1$
  selecting optimally 19.8% of the time on 5 arms, and the
  $1 - \varepsilon + \varepsilon/k$ plateau landing at 91.9% against a 92.0%
  ceiling.
- **Render layer** (24 checks, jsdom) — arm construction, stepping, log capping,
  reset preserving the seed and arms, new-environment changing them, slider and
  toggle wiring, auto-play state, and the comparison sweep completing.

Neither suite is committed; the plan does not call for a test harness, and
adding one is a separate decision.

## Licensing and attribution

Notation and several worked examples follow Sutton & Barto, *Reinforcement
Learning: An Introduction* (2nd ed.). The `LICENSE` file covers code and prose
separately — MIT for code, CC BY 4.0 for written content — and each chapter
cites its sources. Settling this while the book is small avoids an awkward
retrofit across twenty pages.

## First release milestone

- Starlight site shell, live on GitHub Pages, deploying automatically.
- Math rendering and internal-link validation working.
- The Multi-Armed Bandits chapter with an epsilon-greedy explanation page.
- The epsilon-greedy simulator, refactored, seeded, CDN-free, and embedded.
