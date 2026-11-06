/**
 * The lab's chrome: top bar and section navigation (full screen), a compact
 * tab bar (embedded), the shared settings drawer, and the stage where one
 * module is mounted at a time. Changing the configuration remounts the
 * current module, so every module restarts on the new cluster.
 */

import { BOOK_CONFIG, DEFAULT_CONFIG, TIER_NAMES, type ClusterConfig } from '../core/config.ts';
import { createContext } from './context';
import { SECTIONS, moduleById, sectionModules, type SectionId } from './registry';
import { navigate, startRouter, type Route } from './router';
import type { Store } from './store';
import { button, el, fmt, range } from './ui';

export interface ShellOptions {
  embed: boolean;
  section: SectionId | null;
  module: string | null;
}

export function startShell(root: HTMLElement, store: Store, opts: ShellOptions): void {
  root.classList.toggle('embed', opts.embed);
  const tabs = el('nav', { class: 'tabs', attrs: { 'aria-label': 'Modules' } });
  const stage = el('main', { class: 'stage', attrs: { tabindex: '-1' } });
  const drawer = buildDrawer(store);
  const settingsBtn = button('⚙ Cluster settings', () => toggleDrawer(), 'settings-btn');
  settingsBtn.setAttribute('aria-expanded', 'false');
  const toggleDrawer = (open = drawer.hidden) => {
    drawer.hidden = !open;
    settingsBtn.setAttribute('aria-expanded', String(open));
  };
  const summary = el('span', { class: 'config-summary' });

  if (!opts.embed) {
    const sectionNav = el('nav', { class: 'sections', attrs: { 'aria-label': 'Sections' } },
      el('a', { text: 'All modules', attrs: { href: '#/' } }),
      ...SECTIONS.map(s => el('a', { text: s.title, attrs: { href: `#/section/${s.id}`, 'data-section': s.id } })));
    root.append(el('header', { class: 'topbar' },
      el('div', { class: 'brand' }, el('p', { class: 'eyebrow', text: 'Chapter 10 · On-policy control with approximation' }), el('h1', { text: 'GPU admission control lab' })),
      sectionNav));
  }
  root.append(el('div', { class: 'toolbar' }, tabs, el('div', { class: 'toolbar-right' }, summary, settingsBtn)), drawer, stage);

  let current: { id: string; dispose: () => void; unmount: () => void } | null = null;
  let token = 0;

  const showSummary = (cfg: ClusterConfig) => {
    summary.textContent = `K=${cfg.K} · m=${fmt(cfg.median, 1)} · σ=${cfg.sigma} · κ=${cfg.kappa} · A=${cfg.amp} · seed ${cfg.seed}`;
  };
  showSummary(store.config);

  const renderTabs = (activeId: string | null, section: SectionId | null) => {
    const list = opts.embed && opts.section ? sectionModules(opts.section) : section ? sectionModules(section) : [];
    tabs.replaceChildren(...list.map(m => el('a', {
      class: m.id === activeId ? 'tab active' : 'tab',
      attrs: { href: `#/${m.id}`, ...(m.id === activeId ? { 'aria-current': 'page' } : {}) },
    }, el('span', { class: 'tab-id', text: m.id.toUpperCase() }), ` ${m.title}`)));
    root.querySelectorAll<HTMLAnchorElement>('.sections a[data-section]').forEach(a => a.classList.toggle('active', a.dataset.section === section));
  };

  const unmountCurrent = () => {
    if (!current) return;
    try { current.unmount(); } catch (e) { console.error(e); }
    current.dispose();
    current = null;
  };

  const mountModule = async (id: string) => {
    const info = moduleById(id);
    if (!info) return;
    const my = ++token;
    unmountCurrent();
    renderTabs(id, info.section);
    stage.replaceChildren(el('p', { class: 'loading', text: `Loading ${info.title}…` }));
    try {
      const mod = await info.load();
      if (my !== token) return;
      stage.replaceChildren();
      const host = el('div', { class: 'module' });
      stage.append(host);
      const { ctx, dispose } = createContext(store, { embed: opts.embed, go: mid => navigate({ kind: 'module', id: mid }), openSettings: () => toggleDrawer(true) });
      const unmount = mod.mount(host, ctx);
      current = { id, dispose, unmount };
      store.rememberModule(id);
    } catch (e) {
      if (my === token) stage.replaceChildren(el('p', { class: 'error', text: `Could not load the module: ${e instanceof Error ? e.message : String(e)}` }));
    }
  };

  const showHome = () => {
    token += 1;
    unmountCurrent();
    renderTabs(null, null);
    stage.replaceChildren(el('div', { class: 'home' },
      el('p', { class: 'lede', html: 'One running example for the whole chapter: a cluster of <strong>K GPUs</strong> serving requests from four price tiers, each with a <strong>continuous job length</strong>. Every step you accept or reject the request at the head of the queue. Each module makes one idea of Chapter 10 visible.' }),
      ...SECTIONS.map(s => el('section', { class: 'home-section' },
        el('h2', {}, s.title, el('span', { class: 'hint', text: ` ${s.book}` })),
        el('div', { class: 'home-grid' }, ...sectionModules(s.id).map(m => el('a', { class: 'home-card', attrs: { href: `#/${m.id}` } },
          el('span', { class: 'tab-id', text: m.id.toUpperCase() }), el('strong', { text: m.title }), el('span', { class: 'muted', text: m.blurb }))))))));
  };

  const onRoute = (r: Route) => {
    if (r.kind === 'home') showHome();
    else if (r.kind === 'module') void mountModule(r.id);
    else void mountModule(sectionModules(r.id)[0]!.id);
  };

  const syncDrawer = (drawer as HTMLElement & { sync: () => void }).sync;
  store.subscribe(cfg => {
    showSummary(cfg);
    syncDrawer();
    if (current) void mountModule(current.id);
  });

  const fallback = (): Route => {
    if (opts.module && moduleById(opts.module)) return { kind: 'module', id: opts.module };
    if (opts.section) return { kind: 'module', id: sectionModules(opts.section)[0]!.id };
    if (opts.embed && store.lastModule && moduleById(store.lastModule)) return { kind: 'module', id: store.lastModule };
    return { kind: 'home' };
  };
  startRouter(onRoute, fallback);

  // Auto-height for the chapter embeds.
  if (opts.embed && window.parent !== window) {
    let last = 0;
    const post = () => {
      // The content's own height, not scrollHeight: scrollHeight never drops below the iframe's height, so it could only grow.
      const height = Math.ceil(root.getBoundingClientRect().height + root.offsetTop * 2);
      if (Math.abs(height - last) < 2) return;
      last = height;
      window.parent.postMessage({ type: 'gpu-lab-height', height }, '*');
    };
    new ResizeObserver(post).observe(root);
    post();
  }
}

function buildDrawer(store: Store): HTMLElement {
  let pending: Partial<ClusterConfig> = {};
  let timer = 0;
  const apply = (patch: Partial<ClusterConfig>) => {
    pending = { ...pending, ...patch };
    window.clearTimeout(timer);
    timer = window.setTimeout(() => { const p = pending; pending = {}; store.set(p); }, 350);
  };
  const c = store.config;
  const K = range('GPUs K', { min: 4, max: 16, step: 1, value: c.K }, v => apply({ K: v }));
  const m = range('Median job length m', { min: 4, max: 40, step: 0.5, value: c.median, format: v => fmt(v, 1) }, v => apply({ median: v }));
  const sigma = range('Job-length spread σ', { min: 0, max: 1.2, step: 0.05, value: c.sigma, format: v => fmt(v, 2) }, v => apply({ sigma: v }));
  const kappa = range('Energy cost κ per busy GPU', { min: 0, max: 0.5, step: 0.01, value: c.kappa, format: v => fmt(v, 2) }, v => apply({ kappa: v }));
  const amp = range('Diurnal amplitude A', { min: 0, max: 1, step: 0.05, value: c.amp, format: v => fmt(v, 2) }, v => apply({ amp: v }));
  const B = range('Buckets B (exact model)', { min: 1, max: 4, step: 1, value: c.B }, v => apply({ B: v }));
  const H = range('Shift length H', { min: 24, max: 96, step: 4, value: c.H }, v => apply({ H: v }));
  const seed = range('Seed', { min: 0, max: 99, step: 1, value: c.seed }, v => apply({ seed: v }));
  const mixes = TIER_NAMES.map((name, i) => range(`${name} share`, { min: 0, max: 1, step: 0.05, value: c.base[i]!, format: v => fmt(v, 2) }, v => {
    const base = [...store.config.base];
    base[i] = v;
    apply({ base });
  }));
  const note = el('p', { class: 'hint' });
  const drawer = el('aside', { class: 'drawer card', attrs: { hidden: '', 'aria-label': 'Cluster settings' } },
    el('h3', { text: 'Cluster settings' }),
    el('p', { class: 'hint', text: 'Shared by every module; changing a value restarts the current module. References for K ≤ 16 come from the bucketed model; with A > 0 they come from Monte Carlo.' }),
    el('div', { class: 'drawer-grid' }, K.el, m.el, sigma.el, kappa.el, amp.el, B.el, H.el, seed.el),
    el('h3', { text: 'Base tier mix (normalized)' }),
    el('div', { class: 'drawer-grid' }, ...mixes.map(x => x.el)),
    note,
    el('div', { class: 'buttons' },
      button('Defaults', () => store.set({ ...DEFAULT_CONFIG })),
      button('Book Example 10.2 (σ = 0)', () => store.set({ ...BOOK_CONFIG }))));
  const sync = () => {
    const x = store.config;
    K.set(x.K); m.set(x.median); sigma.set(x.sigma); kappa.set(x.kappa); amp.set(x.amp); B.set(x.B); H.set(x.H); seed.set(x.seed);
    mixes.forEach((r, i) => r.set(x.base[i]!));
    note.textContent = x.sigma === 0 ? 'σ = 0: every job has the same length, so the model has one bucket (Example 10.2 when m = 1/0.06 ≈ 16.7).' : '';
  };
  sync();
  return Object.assign(drawer, { sync });
}
