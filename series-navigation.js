/* Preserve links to the observations moved from Part 1 to Part 2. */
(() => {
  const moved = new Set(['#coordinate-pairs', '#observations', '#observations-notes', '#quantization', '#quantization-speed']);
  function route() {
    if (moved.has(window.location.hash)) {
      window.location.replace('rope-in-practice.html' + window.location.hash);
    }
  }
  window.addEventListener('hashchange', route);
  route();
})();
