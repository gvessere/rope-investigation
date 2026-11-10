/* All visual calculations run locally. No third-party scripts or telemetry. */
(() => {
  'use strict';
  const controls = document.getElementById('observation-controls');
  const kind = document.getElementById('plot-kind');
  const view = document.getElementById('plot-view');
  const page = document.getElementById('plot-page');
  const previous = document.getElementById('plot-previous');
  const next = document.getElementById('plot-next');
  const image = document.getElementById('observation-image');
  const status = document.getElementById('observation-status');
  const caption = document.getElementById('observation-caption');
  const download = document.getElementById('observation-pdf');
  const full = document.getElementById('observation-full');
  const zoom = document.getElementById('plot-zoom');
  const pad = n => String(n).padStart(2, '0');
  let currentLayer = 0;
  function setPages() {
    const step = kind.value === 'queries' ? 1 : 4;
    page.replaceChildren();
    for (let l = 0; l < 28; l += step) {
      const option = document.createElement('option');
      option.value = l;
      option.textContent = step === 1 ? `Layer ${l}` : `Layers ${l}–${l+3}`;
      page.append(option);
    }
    page.value = String(Math.floor(currentLayer / step) * step);
  }
  function update() {
    currentLayer = Number(page.value);
    const q = kind.value === 'queries';
    const end = currentLayer + (q ? 0 : 3);
    const file = view.value === 'xy'
      ? `${kind.value}_xy_layers_${pad(currentLayer)}_${pad(end)}.png`
      : `${kind.value}_layers_${pad(currentLayer)}_${pad(end)}_y40.png`;
    const pdf = view.value === 'xy' ? `${kind.value}_xy_dynamic.pdf` : `${kind.value}_all_layers_y40.pdf`;
    const layer = q ? `Layer ${currentLayer}` : `Layers ${currentLayer}–${end}`;
    image.src = `figures/observations/${file}`;
    image.alt = `${layer}, ${q ? 'twelve query heads' : 'two shared key heads per layer'}: ${view.value === 'xy' ? 'pre-RoPE Cartesian coordinates, full range, equal x/y scales in each panel' : 'pre-RoPE phase versus amplitude, amplitude capped at 40'}. Color indicates frequency-pair index.`;
    image.width = view.value === 'xy' ? (q ? 2040 : 1320) : (q ? 1920 : 1440);
    image.height = view.value === 'xy' ? (q ? 1560 : 1920) : (q ? 1320 : 1680);
    status.textContent = `${layer} · ${q ? '12 query heads' : '8 shared-key panels'} · page ${page.selectedIndex+1} of ${page.options.length}`;
    caption.textContent = view.value === 'xy'
      ? 'Original Cartesian plot. Each panel has its own full range, with equal x/y units and no clipping. Compare tick values across heads. Color: pair 0 (highest frequency) → pair 63 (lowest).'
      : `Original phase/amplitude plot. Phase spans 0–2π; amplitude is capped at 40, hiding ${q ? '4,582 points (0.0277%)' : '61,854 points (0.7254%)'} across the complete ${kind.value} collection. Switch to Cartesian coordinates to see the full range.`;
    download.href = `documents/${pdf}`;
    full.href = image.src;
    previous.disabled = page.selectedIndex === 0;
    next.disabled = page.selectedIndex === page.options.length - 1;
  }
  kind.addEventListener('change', () => { setPages(); update(); });
  view.addEventListener('change', update);
  page.addEventListener('change', update);
  previous.addEventListener('click', () => { page.selectedIndex--; update(); });
  next.addEventListener('click', () => { page.selectedIndex++; update(); });
  zoom.addEventListener('change', () => document.getElementById('observation-frame').classList.toggle('enlarged', zoom.value === 'large'));
  setPages(); update(); controls.hidden = false;

  // Translate a fitted score by t using phases phi'_j = phi_j - omega_j*t.
  const slider = document.getElementById('shape-shift');
  const choice = document.getElementById('shape-choice');
  const output = document.getElementById('shift-value');
  const svg = document.getElementById('shift-plot');
  const original = document.getElementById('shift-original');
  const shifted = document.getElementById('shift-moved');
  function render(data) {
    const coeff = data.examples[Number(choice.value)].coefficients;
    const t = Number(slider.value), J = data.omega.length;
    output.textContent = `${t > 0 ? '+' : ''}${t} tokens`;
    function path(shift) {
      const result = [];
      let drawing = false;
      for (let n = 0; n <= 255; n++) {
        if (n - shift < 0 || n - shift > 255) { drawing = false; continue; }
        let score = 0;
        for (let j = 0; j < J; j++) {
          const phase = data.omega[j] * (n - shift);
          score += coeff[j] * Math.cos(phase) + coeff[J+j] * Math.sin(phase);
        }
        result.push(`${drawing ? 'L' : 'M'}${(48+n*620/255).toFixed(2)},${(235-score*150).toFixed(2)}`);
        drawing = true;
      }
      return result.join(' ');
    }
    original.setAttribute('d', path(0));
    shifted.setAttribute('d', path(t));
    svg.setAttribute('aria-label', `${data.examples[Number(choice.value)].name}; the green score is translated ${t} tokens relative to the dashed original. The display uses the same fitted RoPE coefficients as the static examples.`);
  }
  fetch('figures/shift-examples.json').then(r => {
    if (!r.ok) throw new Error('Could not load coefficient data');
    return r.json();
  }).then(data => {
    data.examples.forEach((entry, i) => {
      const option = document.createElement('option'); option.value = i; option.textContent = entry.name; choice.append(option);
    });
    slider.addEventListener('input', () => render(data));
    choice.addEventListener('change', () => render(data));
    document.getElementById('shift-controls').hidden = false;
    document.getElementById('shift-unavailable').hidden = true;
    svg.removeAttribute('hidden');
    render(data);
  }).catch(() => { document.getElementById('shift-unavailable').textContent = 'The interactive example could not load. The static examples and all coefficient data remain available below.'; });
})();
