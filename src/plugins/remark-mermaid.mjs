/** Preserve Mermaid fences as text for the browser renderer, bypassing code highlighting. */
export default function remarkMermaid() {
  return (tree) => {
    function visit(node) {
      if (node.type === 'code' && node.lang === 'mermaid') {
        const source = node.value
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;');
        node.type = 'html';
        node.value = `<pre class="mermaid">${source}</pre>`;
      }
      node.children?.forEach(visit);
    }
    visit(tree);
  };
}
