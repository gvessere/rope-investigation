export function profileValues(record, mode) {
  if (mode === 'mean') return record.per_key.map(v => v * 100);
  if (mode === 'paired') return record.mass.map(v => v * 100);
  throw new Error('Unknown distance view');
}

export function axisBounds(values, scale = 'auto') {
  if (scale === 'linear') return [0, 100];
  const low = Math.min(...values), high = Math.max(...values);
  if (!(low > 0) || !Number.isFinite(high)) throw new Error('Log view needs positive finite weights');
  const padding = Math.max(.12, (Math.log10(high) - Math.log10(low)) * .08);
  return [10 ** (Math.log10(low) - padding), Math.min(100, 10 ** (Math.log10(high) + padding))];
}

const tickLabel = v => v === 0 ? '0' : v >= .001 ? String(Number(v.toPrecision(3))) : v.toExponential(0);

export function chartSvg(data, record, mode, compact = false, scale = 'auto', sharedBounds = null) {
  const W = 440, H = compact ? 235 : 305;
  const left = 62, right = 425, top = 16, bottom = H - 58;
  const values = profileValues(record, mode);
  const [lo, hi] = sharedBounds || axisBounds(values, scale);
  const linear = scale === 'linear';
  const x = v => left + Math.log1p(v) / Math.log1p(data.bins.at(-1)[1]) * (right-left);
  const y = v => bottom - (linear ? (v-lo)/(hi-lo) : Math.log(v/lo)/Math.log(hi/lo)) * (bottom-top);
  let ticks = linear ? [0,25,50,75,100] : Array.from({length:Math.max(0,Math.floor(Math.log10(hi))-Math.ceil(Math.log10(lo))+1)},(_,i)=>10**(Math.ceil(Math.log10(lo))+i));
  if (ticks.length > 5) ticks = ticks.filter((_,i) => i % Math.ceil(ticks.length/5) === 0);
  if (ticks.length < 2) ticks = [lo,hi];
  const points = vs => vs.map((v,i) => v === null ? null : `${x(data.centers[i])},${y(v)}`).filter(v=>v!==null).join(' ');
  const clip = `distance-clip-${record.layer}-${record.head}`;
  const reference = (mode === 'mean' ? record.reference_per_key : record.reference_mass).map(v=>v===null?null:v*100);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Layer ${record.layer} head ${record.head}, ${mode === 'mean' ? 'mean weight per available key' : 'mass per interval'}, full history, percent">
    <defs><clipPath id="${clip}"><rect x="${left-5}" y="${top-5}" width="${right-left+10}" height="${bottom-top+10}"/></clipPath></defs>
    ${ticks.map(v=>`<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#d4dcd2"/><text x="${left-6}" y="${y(v)+4}" text-anchor="end">${tickLabel(v)}</text>`).join('')}
    ${[0,1,8,64,512,2060].map(v=>`<line x1="${x(v)}" x2="${x(v)}" y1="${top}" y2="${bottom}" stroke="#e3e6df"/><text x="${x(v)}" y="${bottom+22}" text-anchor="${v===2060?'end':'middle'}">${v}</text>`).join('')}
    <path d="M${left},${top}V${bottom}H${right}" fill="none" stroke="#586a64"/>
    <g clip-path="url(#${clip})"><polyline points="${points(reference)}" fill="none" stroke="#888" stroke-width="1.5" stroke-dasharray="6 5"/>
    <polyline points="${points(values)}" fill="none" stroke="#176452" stroke-width="2.5"/>
    ${values.map((v,i)=>`<circle cx="${x(data.centers[i])}" cy="${y(v)}" r="4" fill="${i===0?'#c26632':'#176452'}"><title>${i===0?'Self':`Distance ${data.bins[i][0]}–${data.bins[i][1]}`}: ${v.toPrecision(6)}%${mode==='mean'?' per available key':' of full attention'}</title></circle>`).join('')}</g>
    <text x="${(left+right)/2}" y="${H-13}" text-anchor="middle">Distance (tokens; log(1 + d), 0 = self)</text>
    </svg>`;
}

export function initExplorer(root, data) {
  const layer = root.querySelector('[data-layer]'), head = root.querySelector('[data-head]');
  const mode = root.querySelector('[data-mode]'), scale = root.querySelector('[data-scale]');
  const previous = root.querySelector('[data-previous]'), next = root.querySelector('[data-next]');
  const panels = root.querySelector('[data-panels]'), status = root.querySelector('[data-status]');
  for (let i=0;i<28;i++) layer.add(new Option(`Layer ${i}`,String(i)));
  for (let i=0;i<12;i++) head.add(new Option(`Head ${i}`,String(i)));
  for (const control of root.querySelectorAll('select,button')) control.disabled=false;
  function render(updateHash=true) {
    const l=Number(layer.value), m=mode.value, s=scale.value;
    const heads=head.value==='all'?Array.from({length:12},(_,i)=>i):[Number(head.value)];
    const records=heads.map(h=>data.heads[l*12+h]);
    const shared=s==='shared'?axisBounds(records.flatMap(r=>profileValues(r,m))):null;
    panels.classList.toggle('distance-single',heads.length===1);
    panels.innerHTML=records.map(record=>`<figure class="distance-panel"><h3>L${l} H${record.head}</h3>${chartSvg(data,record,m,heads.length>1,s,shared)}<figcaption><strong>100% accounted for</strong> · self ${(record.mass[0]*100).toPrecision(3)}% · ages 1–7 ${(record.mass.slice(1,8).reduce((a,b)=>a+b,0)*100).toPrecision(3)}%</figcaption></figure>`).join('');
    status.textContent=`Layer ${l} · ${heads.length===1?'head '+heads[0]:'all 12 heads'} · ${m==='mean'?'mean weight per available key (%)':'mass per interval (%)'} · ${s==='auto'?'log y-axis fits each head; axis ranges differ':s==='shared'?'shared log y-axis for the displayed heads':'shared linear y-axis, 0–100%'}. All distances included; no attention renormalization. Orange = self; dashed gray = 1/d guide anchored at 16–31.`;
    previous.disabled=l===0;next.disabled=l===27;
    if(updateHash) history.replaceState(null,'',`#all-heads-${m}-L${l}${head.value==='all'?'':'-H'+head.value}-${s}`);
  }
  function fromHash() {
    const match=location.hash.match(/^#all-heads-(mean|paired)-L(\d+)(?:-H(\d+))?(?:-(auto|shared|linear))?$/);
    layer.value=match&&Number(match[2])<28?match[2]:'6';
    mode.value=match?match[1]:'paired';
    head.value=match&&match[3]!==undefined&&Number(match[3])<12?match[3]:'all';
    scale.value=match&&match[4]?match[4]:'auto';
    render(false);if(match) root.scrollIntoView({block:'start'});
  }
  for(const control of [layer,head,mode,scale])control.addEventListener('change',()=>render());
  previous.addEventListener('click',()=>{layer.value=String(Number(layer.value)-1);render();});
  next.addEventListener('click',()=>{layer.value=String(Number(layer.value)+1);render();});
  window.addEventListener('hashchange',fromHash);fromHash();
}

if(typeof document!=='undefined') {
  const root=document.querySelector('#all-head-distance-explorer');
  if(root) fetch(new URL('./data/decay/all-head-distance.json?v=full-history-1',import.meta.url))
    .then(response=>{if(!response.ok)throw new Error('Could not load saved profiles');return response.json();})
    .then(data=>initExplorer(root,data))
    .catch(error=>{root.querySelector('[data-status]').textContent=`The explorer could not load: ${error.message}. Please refresh to try again.`;});
}
