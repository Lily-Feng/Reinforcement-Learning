// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import starlightLinksValidator from 'starlight-links-validator';
import remarkMermaid from './src/plugins/remark-mermaid.mjs';

/**
 * Chapters, easy to complex. This array IS the reading order.
 *
 * Directory names carry the reading-order number, so it appears in the URL
 * (/chapters/02-multi-armed-bandits/). Sidebar labels stay unnumbered.
 *
 * The numbers are load-bearing: renumbering a chapter changes its published
 * URL and breaks any existing link to it. Prefer appending over inserting.
 */
const chapters = [
  ['Foundations', '01-foundations'],
  ['Multi-armed bandits', '02-multi-armed-bandits'],
  ['Markov decision processes', '03-markov-decision-processes'],
  ['Dynamic programming', '04-dynamic-programming'],
  ['Monte Carlo methods', '05-monte-carlo-methods'],
  ['Temporal-difference learning', '06-temporal-difference-learning'],
  ['n-step and eligibility traces', '07-n-step-and-eligibility-traces'],
  ['Planning and learning', '08-planning-and-learning'],
  ['Function approximation', '09-function-approximation'],
  ['Deep Q-networks', '10-deep-q-networks'],
  ['Policy gradient methods', '11-policy-gradient-methods'],
  ['Advanced policy optimization', '12-advanced-policy-optimization'],
];

export default defineConfig({
  site: 'https://lily-feng.github.io',
  base: '/Reinforcement-Learning',

  // Starlight renders no LaTeX on its own. Configured here rather than
  // discovered halfway through the first chapter with equations in it.
  markdown: {
    remarkPlugins: [remarkMath, remarkMermaid],
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
      components: { Head: './src/components/MermaidHead.astro' },
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
