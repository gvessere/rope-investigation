import {readNPZ} from './scatter-data.mjs';
import {pairMaxima} from './head-pair-data.mjs';
const $=id=>document.getElementById(id),canvas=$('head-pair-canvas');
if(canvas){
 const ctx=canvas.getContext('2d'),controls=$('head-pair-controls'),fallback=$('head-pair-fallback'),status=$('head-pair-status');
 const layer=$('head-pair-layer'),range=$('head-pair-range'),group=$('head-pair-group'),size=$('head-pair-size');
 const cache=new Map();let revision=0,palette;
 for(let l=0;l<28;l++)layer.add(new Option(`Layer ${l}`,String(l)));layer.value='14';
 async function getLayer(l){
  if(cache.has(l))return cache.get(l);
  const pending=(async()=>{const r=await fetch(`data/observations/queries_layer_${String(l).padStart(2,'0')}.npz`);if(!r.ok)throw Error('Saved queries unavailable');const d=await readNPZ(await r.arrayBuffer());if(d.amplitude.shape.join(',')!==d.phase.shape.join(',')||d.amplitude.shape[0]!==12||d.amplitude.shape[2]!==64)throw Error('Unexpected saved query shape');return d;})();
  cache.set(l,pending);while(cache.size>3)cache.delete(cache.keys().next().value);
  try{return await pending;}catch(e){cache.delete(l);throw e;}
 }
 function text(s,x,y,n=12,color='#172e2b',align='left'){ctx.fillStyle=color;ctx.font=`${n}px system-ui,sans-serif`;ctx.textAlign=align;ctx.fillText(s,x,y);}
 async function draw(){
  const version=++revision,l=Number(layer.value),start=Number(range.value),count=16;
  status.textContent='Loading saved query pairs…';
  try{
   const d=await getLayer(l);if(version!==revision)return;
   const limits=pairMaxima(d.amplitude,start,count),tokens=d.amplitude.shape[1];
   const heads=group.value==='all'?Array.from({length:12},(_,h)=>h):Array.from({length:6},(_,h)=>h+Number(group.value)*6);
   const W=150,H=98,left=94,top=84;
   canvas.width=left+count*W+18;canvas.height=top+heads.length*H+44;
   ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);
   text(`Layer ${l} · query phase and radius · pairs ${start}–${start+15}`,left,26,18);
   text('Rows: query heads · columns: frequency pairs · common radius scale within each column',left,49,13,'#586a64');
   for(let c=0;c<count;c++){text(`Pair ${start+c}`,left+c*W+W/2,top-19,13,'#172e2b','center');text(`r: 0–${limits[c].toPrecision(3)}`,left+c*W+W/2,top-3,11,'#586a64','center');}
   heads.forEach((h,row)=>{
    const y0=top+row*H,kv=Math.floor(h/6);
    ctx.fillStyle=kv===0?'#edf5ee':'#f0eff8';ctx.fillRect(0,y0,canvas.width,H);
    text(`H${h} · KV${kv}`,8,y0+42,13,kv===0?'#176452':'#65568a');
    for(let c=0;c<count;c++){
     const j=start+c,x0=left+c*W+7,y=y0+8,pw=W-18,ph=H-25;
     ctx.fillStyle='#fff';ctx.fillRect(x0,y,pw,ph);
     ctx.strokeStyle='#c5cbc4';ctx.lineWidth=.6;ctx.setLineDash([1,3]);
     for(let k=0;k<=4;k++){ctx.beginPath();ctx.moveTo(x0+k*pw/4,y);ctx.lineTo(x0+k*pw/4,y+ph);ctx.stroke();}ctx.setLineDash([]);
     ctx.save();ctx.beginPath();ctx.rect(x0,y,pw,ph);ctx.clip();ctx.globalAlpha=.45;ctx.fillStyle=palette[j];
     for(let t=0;t<tokens;t++){
      const ix=(h*tokens+t)*64+j,r=d.amplitude.values[ix],p=d.phase.values[ix];
      ctx.fillRect(x0+p/(2*Math.PI)*pw-.7,y+ph-r/limits[c]*ph-.7,1.4,1.4);
     }
     ctx.restore();ctx.strokeStyle='#aab4aa';ctx.strokeRect(x0,y,pw,ph);
     if(row===heads.length-1){text('0',x0,y+ph+15,10);text('π',x0+pw/2,y+ph+15,10,'#586a64','center');text('2π',x0+pw,y+ph+15,10,'#586a64','right');}
    }
    if(h===6&&row>0){ctx.strokeStyle='#65568a';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(0,y0);ctx.lineTo(canvas.width,y0);ctx.stroke();}
   });
   text('x: pre-RoPE phase (radians) · y: radius r · all saved points shown',left,canvas.height-9,13,'#586a64');
   canvas.hidden=false;fallback.hidden=true;controls.hidden=false;
   canvas.style.width=size.value==='fit'?'100%':`${canvas.width}px`;
   status.textContent=`Layer ${l} · pairs ${start}–${start+15} · ${heads.length} heads · ${(heads.length*count*tokens).toLocaleString()} saved points. H0–H5 share KV0; H6–H11 share KV1. Each column’s full radius range uses all 12 heads, including when one group is selected.`;
  }catch(e){if(version!==revision)return;status.textContent=`Interactive view unavailable: ${e.message}. The saved layer-14 sheet remains below.`;canvas.hidden=true;fallback.hidden=false;}
 }
 for(const el of [layer,range,group,size])el.addEventListener('change',draw);
 $('head-pair-save').addEventListener('click',()=>{canvas.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob),a=document.createElement('a');a.download=`query-pairs-L${layer.value}-${range.value}.png`;a.href=url;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});});
 try{const r=await fetch('data/observations/palette.json');if(!r.ok)throw Error('Palette unavailable');palette=await r.json();await draw();}catch(e){status.textContent=`Interactive view unavailable: ${e.message}. The saved layer-14 sheet remains below.`;}
}
