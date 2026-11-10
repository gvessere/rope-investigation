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
    image.src = `figures/observations/${file}?v=pi-guides-1`;
    image.alt = `${layer}, ${q ? 'twelve query heads' : 'two shared key heads per layer'}: ${view.value === 'xy' ? 'pre-RoPE Cartesian coordinates, full range, equal x/y scales in each panel' : 'pre-RoPE phase versus amplitude, amplitude capped at 40'}. Color indicates frequency-pair index.`;
    image.width = view.value === 'xy' ? (q ? 2040 : 1320) : (q ? 1920 : 1440);
    image.height = view.value === 'xy' ? (q ? 1560 : 1920) : (q ? 1320 : 1680);
    status.textContent = `${layer} · ${q ? '12 query heads' : '8 shared-key panels'} · page ${page.selectedIndex+1} of ${page.options.length}`;
    caption.textContent = view.value === 'xy'
      ? 'Cartesian plot with dotted gray guides at multiples of π/4 and π/3. Each panel has its own full range, with equal x/y units and no clipping. Compare tick values across heads. Color: pair 0 (highest frequency) → pair 63 (lowest).'
      : `Phase/amplitude plot with dotted gray guides at multiples of π/4 and π/3. Phase spans 0–2π; amplitude is capped at 40, hiding ${q ? '4,582 points (0.0277%)' : '61,854 points (0.7254%)'} across the complete ${kind.value} collection. Switch to Cartesian coordinates to see the full range.`;
    download.href = `documents/${pdf}?v=pi-guides-1`;
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
  const pairSvg = document.getElementById('phase-pairs');
  const play = document.getElementById('shift-play');
  const wheels = [];
  let playing = false, frameId = 0, animationStart = null, startPhase = 0, lastDraw = 0;
  function svgElement(tag, attrs = {}) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k, String(v)));
    return el;
  }
  function buildPairs(data) {
    pairSvg.replaceChildren();
    const defs = svgElement('defs'), marker = svgElement('marker', {id:'pair-arrowhead',viewBox:'0 0 10 10',refX:8,refY:5,markerWidth:4,markerHeight:4,orient:'auto-start-reverse'});
    marker.append(svgElement('path',{d:'M0 0L10 5L0 10Z',fill:'#176452'}));defs.append(marker);pairSvg.append(defs);
    data.omega.forEach((omega,j) => {
      const g=svgElement('g',{transform:`translate(${j%8*90},${Math.floor(j/8)*90})`});
      const label=svgElement('text',{x:45,y:12,'text-anchor':'middle','font-size':11});label.textContent=`Pair ${j}`;
      g.append(label,svgElement('circle',{cx:45,cy:41,r:23,fill:'none',stroke:'#ccd4c9','stroke-width':1}));
      g.append(svgElement('path',{d:'M22 41H68M45 18V64',stroke:'#e0e4db','stroke-width':.7}));
      const base=svgElement('line',{x1:45,y1:41,stroke:'#8c958d','stroke-width':1.3,'stroke-dasharray':'3 2'});
      const line=svgElement('line',{x1:45,y1:41,stroke:'#176452','stroke-width':1.8,'marker-end':'url(#pair-arrowhead)',class:'pair-rotated','data-pair':j,'data-omega':omega});
      const delta=svgElement('text',{x:45,y:76,'text-anchor':'middle','font-size':9,fill:'#586a64'});
      const frequency=svgElement('text',{x:45,y:88,'text-anchor':'middle','font-size':9.5,fill:'#586a64'});frequency.textContent=`ω=${omega.toPrecision(2)}`;
      const title=svgElement('title');g.append(base,line,delta,frequency,title);pairSvg.append(g);wheels.push({base,line,delta,title});
    });
    pairSvg.removeAttribute('hidden');
  }
  function stop(reason='button') {
    play.dataset.stopReason=reason;
    playing=false;cancelAnimationFrame(frameId);play.textContent='Play phase movement';play.dataset.playing='false';
  }
  function start(data) {
    playing=true;animationStart=null;lastDraw=0;startPhase=Math.asin(Number(slider.value)/32);
    play.textContent='Pause phase movement';play.dataset.playing='true';
    function frame(now) {
      if(!playing)return;
      if(animationStart===null)animationStart=now;
      if(now-lastDraw>=1000/30){slider.value=(32*Math.sin(startPhase+(now-animationStart)/2500)).toFixed(1);render(data);lastDraw=now;}
      frameId=requestAnimationFrame(frame);
    }
    frameId=requestAnimationFrame(frame);
  }
  function render(data) {
    const coeff = data.examples[Number(choice.value)].coefficients;
    const t = Number(slider.value), J = data.omega.length;
    output.textContent = `${t > 0 ? '+' : ''}${Number(t.toFixed(1))} tokens`;
    data.omega.forEach((omega,j)=>{
      const phi=Math.atan2(-coeff[J+j],coeff[j]), angle=phi-omega*t, w=wheels[j];
      w.base.setAttribute('x2',45+21*Math.cos(phi));w.base.setAttribute('y2',41-21*Math.sin(phi));
      w.line.setAttribute('x2',45+21*Math.cos(angle));w.line.setAttribute('y2',41-21*Math.sin(angle));
      w.line.dataset.angle=angle;w.line.dataset.base=phi;w.line.dataset.shift=t;
      w.delta.textContent=`Δφ=${(-omega*t).toPrecision(2)}`;
      w.title.textContent=`Pair ${j}: ω=${omega}, amplitude=${Math.hypot(coeff[j],coeff[J+j])}, phase change=${-omega*t} radians. Unit arrow; not a measured activation.`;
    });
    pairSvg.setAttribute('aria-label',`64 constructed pair phases at shift ${t} tokens; phase changes are minus frequency times the shared shift.`);
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
    buildPairs(data);
    slider.addEventListener('input', () => {stop('slider');render(data);});
    choice.addEventListener('change', () => {stop('shape');render(data);});
    play.addEventListener('click',()=>playing?stop():start(data));

    document.getElementById('shift-controls').hidden = false;
    document.getElementById('shift-unavailable').hidden = true;
    svg.removeAttribute('hidden');
    render(data);
  }).catch(() => { document.getElementById('shift-unavailable').textContent = 'The interactive example could not load. The static examples and all coefficient data remain available below.'; });
})();
