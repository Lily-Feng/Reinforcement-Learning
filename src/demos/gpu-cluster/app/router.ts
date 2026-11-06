/**
 * A tiny hash router: #/ (home), #/m13 (a module), #/section/<id>.
 * Hash routing works under the GitHub Pages base path and inside an iframe,
 * and gives back/forward and deep links for free.
 */

import { isSection, moduleById, type SectionId } from './registry';

export type Route = { kind: 'home' } | { kind: 'module'; id: string } | { kind: 'section'; id: SectionId };

export function parseHash(hash: string): Route | null {
  const path = hash.replace(/^#\/?/, '').replace(/\/$/, '');
  if (path === '') return { kind: 'home' };
  const sec = /^section\/([a-z-]+)$/.exec(path);
  if (sec && isSection(sec[1]!)) return { kind: 'section', id: sec[1] };
  if (moduleById(path)) return { kind: 'module', id: path };
  return null;
}

export function routeHash(r: Route): string {
  return r.kind === 'home' ? '#/' : r.kind === 'module' ? `#/${r.id}` : `#/section/${r.id}`;
}

export function startRouter(onRoute: (r: Route) => void, fallback: () => Route): () => void {
  const handle = () => {
    const h = window.location.hash;
    // No hash yet: let the URL's ?section / ?module (embed mode) pick the start.
    onRoute(h === '' || h === '#' ? fallback() : parseHash(h) ?? fallback());
  };
  window.addEventListener('hashchange', handle);
  handle();
  return () => window.removeEventListener('hashchange', handle);
}

export function navigate(r: Route): void {
  const h = routeHash(r);
  if (window.location.hash === h) return;
  window.location.hash = h;
}
