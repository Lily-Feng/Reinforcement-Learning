/**
 * Boot: read the URL (embed / section / module), create the shared store and
 * start the shell and router.
 */

import { isSection } from './registry';
import { startShell } from './shell';
import { createStore } from './store';

export function boot(root: HTMLElement): void {
  const params = new URLSearchParams(window.location.search);
  const section = params.get('section');
  startShell(root, createStore(), {
    embed: params.get('embed') === '1',
    section: section && isSection(section) ? section : null,
    module: params.get('module'),
  });
}
