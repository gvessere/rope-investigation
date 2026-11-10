import {readNPZ,selectRange} from './scatter-data.mjs';
const $=id=>document.getElementById(id);
const controls=$('observation-controls'), image=$('observation-image'), canvas=$('observation-canvas');
const auto=$('scale-auto'), rmin=$('scale-rmin'), rmax=$('scale-rmax'), pmin=$('scale-pmin'), pmax=$('scale-pmax'), head=$('plot-head');
const note=$('scale-status'), caption=$('observation-caption');
const cache=new Map(); let revision=0,timer;
const fractions=[0,.25,1/3,.5,2/3,.75,1,1.25,4/3,1.5,5/3,1.75,2];
const labels=['0','π/4','π/3','π/2','2π/3','3π/4','π','5π/4','4π/3','3π/2','5π/3','7π/4','2π'];
const palette=await fetch('data/observations/palette.json').then(r=>{if(!r.ok)throw Error('Palette unavailable');return r.json();});

function heads() {
  const old=head.value;head.replaceChildren(new Option('All heads','all'));
  const q=$('plot-kind').value==='queries';
  for(let h=0;h<(q?12:2);h++)head.add(new Option(q?`H${h}`:`KV${h}`,String(h)));
  head.value=[...head.options].some(o=>o.value===old)?old:'all';
}
async function layer(kind,l) {
  const key=`${kind}/${l}`;
  if(cache.has(key)){const p=cache.get(key);cache.delete(key);cache.set(key,p);return p;}
  const promise=(async()=>{
    const response=await fetch(`data/observations/${kind}_layer_${String(l).padStart(2,'0')}.npz`);
    if(!response.ok)throw Error('Saved plot data could not be loaded');
    const data=await readNPZ(await response.arrayBuffer());
    const a=data.amplitude, p=data.phase;
    if(!a||!p||a.shape.join(',')!==p.shape.join(',')||a.shape[2]!==64||a.shape[0]!== (kind==='queries'?12:2))throw Error('Unexpected plot data shape');
    return data;
  })();cache.set(key,promise);
  while(cache.size>8)cache.delete(cache.keys().next().value);
  try{return await promise;}catch(e){cache.delete(key);throw e;}
}
const compact=n=>Number(n.toPrecision(3)).toString();
function text(c,s,x,y,size=13,align='left',color='#172e2b') {c.fillStyle=color;c.font=`${size}px system-ui,sans-serif`;c.textAlign=align;c.fillText(s,x,y);}
function tick(c,value,x,y,align='center'){text(c,compact(value),x,y,12,align);}
function drawPanel(c,entry,index,cols,width,height,mode,settings) {
  const {l,h,data}=entry, n=data.amplitude.shape[1]*64;
  const radii=data.amplitude.values.subarray(h*n,(h+1)*n), phases=data.phase.values.subarray(h*n,(h+1)*n);
  const range=selectRange(radii,settings);
  const col=index%cols,row=Math.floor(index/cols), ox=col*width,oy=55+row*height;
  const left=ox+76,top=oy+80,plotW=width-111;
  const plotH=mode==='xy'?plotW:height-175;
  const xmin=mode==='xy'?-range.max:settings.pmin*Math.PI;
  const xmax=mode==='xy'?range.max:settings.pmax*Math.PI;
  const ymin=mode==='xy'?-range.max:range.min, ymax=range.max;
  const X=v=>left+(v-xmin)/(xmax-xmin)*plotW,Y=v=>top+(ymax-v)/(ymax-ymin)*plotH;
  text(c,`L${l} ${$('plot-kind').value==='queries'?'H':'KV'}${h}`,ox+width/2,oy+22,17,'center');
  text(c,`r: ${compact(range.min)}–${compact(range.max)}${settings.auto?' · 99th percentile':''}`,ox+width/2,oy+44,12,'center','#586a64');
  c.save();c.beginPath();c.rect(left,top,plotW,plotH);c.clip();
  c.strokeStyle='#a6aaa5';c.lineWidth=.8;c.setLineDash([1,4]);
  for(let i=0;i<fractions.length;i++){
    const phi=fractions[i]*Math.PI;c.beginPath();
    if(mode==='xy'){
      if(i===fractions.length-1)continue;
      const co=Math.cos(phi),si=Math.sin(phi),r=range.max/Math.max(Math.abs(co),Math.abs(si));
      c.moveTo(X(0),Y(0));c.lineTo(X(r*co),Y(r*si));
    }else{c.moveTo(X(phi),top);c.lineTo(X(phi),top+plotH);}
    c.stroke();
  }
  c.setLineDash([]);c.globalAlpha=.32;let radial=0,angular=0,shown=0;
  // Every saved point is drawn individually; no sampling or density aggregation.
  for(let j=0;j<64;j++){
    c.fillStyle=palette[j];
    for(let k=j;k<n;k+=64){
      const r=radii[k],phi=phases[k];
      if(r<range.min||r>range.max){radial++;continue;}
      if(mode==='phase'&&(phi<xmin||phi>xmax+(settings.pmax===2?1e-6:0))){angular++;continue;}
      const x=mode==='xy'?r*Math.cos(phi):phi,y=mode==='xy'?r*Math.sin(phi):r;
      c.fillRect(X(x)-.55,Y(y)-.55,1.1,1.1);shown++;
    }
  }
  c.restore();c.strokeStyle='#7d8b80';c.lineWidth=1;c.strokeRect(left,top,plotW,plotH);
  if(mode==='xy'){
    for(const f of [-1,-.5,0,.5,1]){tick(c,f*range.max,X(f*range.max),top+plotH+20);tick(c,f*range.max,left-8,Y(f*range.max)+4,'right');}
    fractions.slice(0,-1).forEach((f,i)=>{const phi=f*Math.PI,co=Math.cos(phi),si=Math.sin(phi),r=.88*range.max/Math.max(Math.abs(co),Math.abs(si));text(c,labels[i],X(r*co),Y(r*si),11,'center','#666f66');});
    text(c,'x before RoPE',left+plotW/2,top+plotH+44,13,'center');
  }else{
    fractions.forEach((f,i)=>{if(f<settings.pmin||f>settings.pmax)return;c.save();c.translate(X(f*Math.PI),top+plotH+13);c.rotate(-Math.PI/3);text(c,labels[i],0,0,11,'right');c.restore();});
    for(const f of [0,.25,.5,.75,1]){const r=range.min+f*(range.max-range.min);tick(c,r,left-8,Y(r)+4,'right');}
    text(c,'Phase φ (radians)',left+plotW/2,top+plotH+62,13,'center');
  }
  c.save();c.translate(ox+22,top+plotH/2);c.rotate(-Math.PI/2);text(c,mode==='xy'?'y before RoPE':'Radius r',0,0,13,'center');c.restore();
  text(c,`${shown.toLocaleString()} / ${n.toLocaleString()} shown · ${((radial+angular)/n*100).toFixed(2)}% outside view`,ox+width/2,oy+height-12,12,'center','#586a64');
  return {shown,total:n,radial,angular,min:range.min,max:range.max};
}
async function render() {
  const id=++revision,kind=$('plot-kind').value,mode=$('plot-view').value,start=Number($('plot-page').value);
  pmin.disabled=pmax.disabled=mode==='xy';
  const settings={auto:auto.checked,min:Number(rmin.value||0),max:rmax.value===''?null:Number(rmax.value),pmin:Number(pmin.value),pmax:Number(pmax.value)};
  if(mode==='phase'&&!(settings.pmin>=0&&settings.pmax<=2&&settings.pmin<settings.pmax)) {note.textContent='Use 0 ≤ phase from < phase to ≤ 2 (in units of π).';return;}
  note.textContent='Loading saved points…';
  try{
    const layers=Array.from({length:kind==='queries'?1:4},(_,i)=>start+i);
    const all=await Promise.all(layers.map(l=>layer(kind,l)));if(id!==revision)return;
    const entries=[];
    all.forEach((data,i)=>{for(let h=0;h<data.amplitude.shape[0];h++)if(head.value==='all'||Number(head.value)===h)entries.push({l:layers[i],h,data});});
    const cols=Math.min(entries.length,kind==='queries'?4:2),w=entries.length===1?900:500;
    const h=mode==='xy'?w+95:420;
    const staging=document.createElement('canvas');staging.width=cols*w;staging.height=55+Math.ceil(entries.length/cols)*h+65;
    const c=staging.getContext('2d');c.fillStyle='#fff';c.fillRect(0,0,staging.width,staging.height);
    text(c,`Qwen ${kind} · pre-RoPE ${mode==='xy'?'x/y':'phase / radius'} · ${settings.auto?'99th-percentile r per head':'manual / full r range'}`,staging.width/2,29,18,'center');
    const counts=entries.map((e,i)=>drawPanel(c,e,i,cols,w,h,mode,settings));
    const cy=staging.height-30;
    text(c,'Pair index',45,cy,12);
    palette.forEach((color,j)=>{c.fillStyle=color;c.fillRect(120+j*(staging.width-260)/64,cy-13,(staging.width-260)/64+1,12);});
    text(c,'0 = fast',120,cy+18,11);text(c,'63 = slow',staging.width-140,cy+18,11,'right');
    if(id!==revision)return;
    canvas.width=staging.width;canvas.height=staging.height;canvas.getContext('2d').drawImage(staging,0,0);canvas.hidden=false;image.hidden=true;
    const sums=counts.reduce((a,b)=>({shown:a.shown+b.shown,total:a.total+b.total,radial:a.radial+b.radial,angular:a.angular+b.angular}),{shown:0,total:0,radial:0,angular:0});
    note.textContent=`${sums.shown.toLocaleString()} of ${sums.total.toLocaleString()} points shown. Outside r range: ${sums.radial.toLocaleString()} (${(sums.radial/sums.total*100).toFixed(2)}%); outside phase window: ${sums.angular.toLocaleString()}.`;
    $('observation-status').textContent=`${kind==='queries'?`Layer ${start}`:`Layers ${start}–${start+3}`} · ${head.value==='all'?'all heads':(kind==='queries'?'H':'KV')+head.value} · ${entries.length} panel${entries.length===1?'':'s'}`;
    canvas.setAttribute('aria-label',`Interactive ${kind} ${mode} plots. ${note.textContent}`);
    canvas.dataset.summary=JSON.stringify({kind,mode,auto:settings.auto,settings,counts});
    caption.textContent=`${settings.auto?'The r limit is the 99th percentile of all saved radii in each head (all 64 pairs and sampled tokens).':'Manual r limits are shared across shown heads; an empty r max uses each head’s full range.'} ${mode==='xy'?'Equal x/y scales are preserved; points outside the selected radial interval are omitted, even if they would fit in the square corners.':'Phase limits are in multiples of π.'} Dotted gray guides mark multiples of π/4 and π/3. Every retained saved point is drawn; there is no subsampling.`;
    $('observation-full').textContent='Save current view (PNG)';
  }catch(error){if(id!==revision)return;canvas.hidden=true;image.hidden=false;caption.textContent='Fixed-scale archive image shown; the selected range could not be drawn.';$('observation-full').textContent='Open fixed-scale image';note.textContent=`${error.message.startsWith('Use ')?error.message:'The interactive plot could not be loaded.'} The fixed-scale plot and PDFs are still available.`;}
}
function schedule(){revision++;clearTimeout(timer);timer=setTimeout(render,100);}
$('plot-kind').addEventListener('change',()=>{heads();schedule();});
controls.addEventListener('change',schedule);
controls.addEventListener('click',e=>{if(e.target.closest('#plot-next,#plot-previous'))schedule();});
for(const field of [rmin,rmax])field.addEventListener('input',()=>{auto.checked=false;schedule();});
for(const field of [pmin,pmax])field.addEventListener('input',schedule);
auto.addEventListener('change',()=>{if(auto.checked){rmin.value='0';rmax.value='';}schedule();});
$('scale-full').addEventListener('click',()=>{auto.checked=false;rmin.value='0';rmax.value='';pmin.value='0';pmax.value='2';schedule();});
$('observation-full').addEventListener('click',e=>{
  if(canvas.hidden)return;e.preventDefault();canvas.toBlob(blob=>{const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`rope-${$('plot-kind').value}-${$('plot-view').value}-scaled.png`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
});
heads();render();
