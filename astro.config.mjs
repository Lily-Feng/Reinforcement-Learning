// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import starlightLinksValidator from 'starlight-links-validator';

/**
 * Chapters, easy to complex. This array IS the reading order.
 *
 * Directory names carry no numeric prefixes, so a chapter can be inserted,
 * renamed, or reordered here without changing a single published URL.
 */
const chapters = [
  ['Foundations', 'foundations'],
  ['Multi-armed bandits', 'multi-armed-bandits'],
  ['Markov decision processes', 'markov-decision-processes'],
  ['Dynamic programming', 'dynamic-programming'],
  ['Monte Carlo methods', 'monte-carlo-methods'],
  ['Temporal-difference learning', 'temporal-difference-learning'],
  ['n-step and eligibility traces', 'n-step-and-eligibility-traces'],
  ['Planning and learning', 'planning-and-learning'],
  ['Function approximation', 'function-approximation'],
  ['Deep Q-networks', 'deep-q-networks'],
  ['Policy gradient methods', 'policy-gradient-methods'],
  ['Advanced policy optimization', 'advanced-policy-optimization'],
];

export default defineConfig({
  site: 'https://lily-feng.github.io',
  base: '/Reinforcement-Learning',

  // Starlight renders no LaTeX on its own. Configured here rather than
  // discovered halfway through the first chapter with equations in it.
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },

  integrations: [
    starlight({
      title: 'Reinforcement Learning',
      description:
        'A chapter-based reinforcement learning book with worked examples and interactive demos.',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/Lily-Feng/Reinforcement-Learning',
        },
      ],
      // Fails the build on a broken internal link. `astro check` does not do
      // this — it only covers TypeScript and component diagnostics.
      plugins: [starlightLinksValidator()],
      customCss: ['./src/styles/custom.css', 'katex/dist/katex.min.css'],
      sidebar: [
        { label: 'Start here', link: '/getting-started/' },
        ...chapters.map(([label, dir]) => ({
          label,
          autogenerate: { directory: `chapters/${dir}` },
          collapsed: true,
        })),
      ],
    }),
  ],
});
