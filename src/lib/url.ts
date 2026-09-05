/**
 * Base-aware URL construction.
 *
 * The site is served from a subpath (`/Reinforcement-Learning/`) and Astro does
 * NOT rewrite root-absolute links. A hardcoded `/demos/...` looks correct in
 * review and 404s in production. Every internal link is built here instead.
 */

const BASE = import.meta.env.BASE_URL;

export function withBase(path: string): string {
  return `${BASE.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
}

/** Canonical route for a demo, given its chapter slug and demo slug. */
export function demoUrl(chapter: string, demo: string): string {
  return withBase(`demos/${chapter}/${demo}/`);
}

/**
 * Canonical route for a chapter page.
 *
 * `chapter` includes the reading-order prefix, e.g. "02-multi-armed-bandits".
 * Demo slugs do NOT carry a prefix — see `demoUrl`.
 */
export function chapterUrl(chapter: string, page = ''): string {
  return withBase(`chapters/${chapter}/${page}`);
}

const REPO = 'https://github.com/Lily-Feng/Reinforcement-Learning';

/** Link to a source file on GitHub, for the "View source" affordance. */
export function sourceUrl(path: string): string {
  return `${REPO}/blob/main/${path.replace(/^\//, '')}`;
}
