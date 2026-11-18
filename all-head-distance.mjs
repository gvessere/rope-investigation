export function profileValues(record, mode) {
  if (mode === 'mean') return record.per_key;
  if (mode === 'paired') return record.conditional_mass.map(v => v * 100);
  throw new Error('Unknown distance view');
}

export function chartSvg(data, record, mode) {
  const mean = mode === 'mean', W = 440, H = 305;
  const left = 62, right = 425, top = 16, bottom = 247;
  const x = v => left + Math.log2(v / 8) / 7 * (right - left);
  const y = v => mean
    ? bottom - (Math.log10(Math.max(v, 1e-300)) + 6) / (Math.log10(.2) + 6) * (bottom - top)
    : bottom - v / 100 * (bottom - top);
  const ticks = mean ? [1e-6, 1e-4, .01, .1] : [0, 25, 50, 75, 100];
  const labels = mean ? ['1e−6', '1e−4', '0.01', '0.1'] : ['0', '25', '50', '75', '100'];
  const points = values => values.map((v, i) => `${x(data.centers[i])},${y(v)}`).join(' ');
  const clip = `distance-clip-${record.layer}-${record.head}`;
  const values = profileValues(record, mode);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Layer ${record.layer} head ${record.head}, ${mean ? 'mean weight per key' : 'mass per interval'}">
    <defs><clipPath id="${clip}"><rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}"/></clipPath></defs>
    ${ticks.map((v, i) => `<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#d4dcd2"/><text x="${left-10}" y="${y(v)+4}" text-anchor="end">${labels[i]}</text>`).join('')}
    ${[8, 32, 128, 512].map(v => `<line x1="${x(v)}" x2="${x(v)}" y1="${top}" y2="${bottom}" stroke="#e3e6df"/><text x="${x(v)}" y="${bottom+22}" text-anchor="middle">${v}</text>`).join('')}
    <path d="M${left},${top}V${bottom}H${right}" fill="none" stroke="#586a64"/>
    <g clip-path="url(#${clip})">${mean ? `<polyline points="${points(data.reference)}" fill="none" stroke="#888" stroke-width="1.5" stroke-dasharray="6 5"/>` : ''}
    <polyline points="${points(values)}" fill="none" stroke="#176452" stroke-width="2.5"/>
    ${values.map((v, i) => `<circle cx="${x(data.centers[i])}" cy="${y(v)}" r="4" fill="#176452"><title>Distance ${data.bins[i][0]}–${data.bins[i][1]}: ${v.toPrecision(5)}${mean ? ' per key' : '% of plotted-range mass'}</title></circle>`).join('')}</g>
    <text x="${(left+right)/2}" y="${H-13}" text-anchor="middle">Look-back distance (tokens, log scale)</text>
    </svg>`;
}

export function initExplorer(root, data) {
  const layer = root.querySelector('[data-layer]');
  const head = root.querySelector('[data-head]');
  const mode = root.querySelector('[data-mode]');
  const previous = root.querySelector('[data-previous]');
  const next = root.querySelector('[data-next]');
  const panels = root.querySelector('[data-panels]');
  const status = root.querySelector('[data-status]');
  for (let i = 0; i < 28; i++) layer.add(new Option(`Layer ${i}`, String(i)));
  for (let i = 0; i < 12; i++) head.add(new Option(`Head ${i}`, String(i)));
  for (const control of root.querySelectorAll('select,button')) control.disabled = false;
  function render(updateHash = true) {
    const l = Number(layer.value), m = mode.value;
    const heads = head.value === 'all' ? Array.from({length: 12}, (_, i) => i) : [Number(head.value)];
    panels.classList.toggle('distance-single', heads.length === 1);
    panels.innerHTML = heads.map(h => {
      const record = data.heads[l * 12 + h];
      return `<figure class="distance-panel"><h3>L${l} H${h}</h3>${chartSvg(data, record, m)}<figcaption>Distances 8–1,023 hold <strong>${(record.range_mass * 100).toPrecision(3)}%</strong> of this head’s full attention.</figcaption></figure>`;
    }).join('');
    status.textContent = `Layer ${l} · ${heads.length === 1 ? 'head ' + heads[0] : 'all 12 heads'} · ${m === 'mean' ? 'mean weight per key (logarithmic y-axis); dashed gray = 1/d reference' : 'mass per interval (%); seven intervals sum to 100% of the plotted range'}. Axes stay fixed across all heads and layers.`;
    previous.disabled = l === 0; next.disabled = l === 27;
    if (updateHash) history.replaceState(null, '', `#all-heads-${m}-L${l}${head.value === 'all' ? '' : '-H'+head.value}`);
  }
  function fromHash() {
    const match = location.hash.match(/^#all-heads-(mean|paired)-L(\d+)(?:-H(\d+))?$/);
    layer.value = match && Number(match[2]) < 28 ? match[2] : '6';
    mode.value = match ? match[1] : 'mean';
    head.value = match && match[3] !== undefined && Number(match[3]) < 12 ? match[3] : 'all';
    render(false);
    if (match) root.scrollIntoView({block: 'start'});
  }
  for (const control of [layer, head, mode]) control.addEventListener('change', () => render());
  previous.addEventListener('click', () => { layer.value = String(Number(layer.value)-1); render(); });
  next.addEventListener('click', () => { layer.value = String(Number(layer.value)+1); render(); });
  window.addEventListener('hashchange', fromHash);
  fromHash();
}

if (typeof document !== 'undefined') {
  const root = document.querySelector('#all-head-distance-explorer');
  if (root) fetch(new URL('./data/decay/all-head-distance.json', import.meta.url))
    .then(response => { if (!response.ok) throw new Error('Could not load saved profiles'); return response.json(); })
    .then(data => initExplorer(root, data))
    .catch(error => { root.querySelector('[data-status]').textContent = `The explorer could not load: ${error.message}. Please refresh to try again.`; });
}
