import {buildPairScores,editAttention} from './navigator-contributions.mjs';
const gains=new Float64Array(64).fill(1);let owner=null,query=-1,pairs=null,baseScores=null,baseProb=null,result=null,revision=0,lastRevision=-1;
const el=id=>document.getElementById(id),num=x=>Number(x).toFixed(4),prob=x=>(100*x).toFixed(3)+'%';
for(let j=0;j<64;j++)el('editPair').add(new Option('Pair '+j,j));
function ensure(){
 if(!D||qi===null)return false;
 if(owner!==D||query!==qi){owner=D;query=qi;gains.fill(1);revision++;pairs=buildPairScores(D.q.subarray(qi*128,qi*128+128),D.k,D.positions[qi],D.freq);const count=D.positions[qi]+1;baseScores=Float64Array.from(D.scores.subarray(qi*D.n,qi*D.n+count));baseProb=Float64Array.from(D.probs.subarray(qi*D.n,qi*D.n+count));lastRevision=-1}
 if(revision!==lastRevision){result=editAttention(baseScores,baseProb,pairs,gains);lastRevision=revision}
 return true;
}
function showBars(){
 const canvas=el('bars'),{c,w,h}=surface(canvas,240),left=45,right=w-12,top=15,bottom=h-30,zero=(top+bottom)/2,bw=(right-left)/64;
 const choice=el('pairBars').value;
 const original=Array.from({length:64},(_,j)=>{const pre=(D.q[qi*128+j]*D.k[ki*128+j]+D.q[qi*128+j+64]*D.k[ki*128+j+64])/Math.sqrt(128);return choice==='pre'?pre:choice==='delta'?C[j]-pre:C[j]});
 const edited=original.map((x,j)=>x*gains[j]),max=Math.max(.01,...original.map(Math.abs),...edited.map(Math.abs));
 c.font='12px system-ui';c.strokeStyle='#cdd7df';c.beginPath();c.moveTo(left,zero);c.lineTo(right,zero);c.stroke();c.fillStyle='#526674';c.fillText(max.toFixed(2),0,top+8);c.fillText('0',20,zero+4);c.fillText((-max).toFixed(2),0,bottom);
 for(let j=0;j<64;j++){
  const nativeHeight=original[j]/max*(bottom-top)/2,height=edited[j]/max*(bottom-top)/2,x=left+j*bw;
  c.strokeStyle='#7d888e';c.lineWidth=1;c.strokeRect(x,Math.min(zero,zero-nativeHeight),bw-1,Math.max(1,Math.abs(nativeHeight)));
  c.fillStyle=colors[j];c.globalAlpha=j===pair?1:.7;c.fillRect(x+1,Math.min(zero,zero-height),Math.max(1,bw-3),Math.max(1,Math.abs(height)));
  if(j===pair){c.globalAlpha=1;c.strokeStyle='#193247';c.lineWidth=2;c.strokeRect(x-1,Math.min(zero,zero-height)-3,bw+1,Math.abs(height)+6)}
 }
 c.globalAlpha=1;c.textAlign='center';for(const j of [0,16,32,48,63])c.fillText(j,left+(j+.5)*bw,bottom+18);canvas.geom={left,right};
}
function showProfile(){
 const canvas=el('editedProfile'),{c,w,h}=surface(canvas,230),left=62,right=w-15,top=12,bottom=h-35,age=D.positions[qi],lx=Math.log1p(age);
 const X=k=>left+Math.log1p(age-k)/lx*(right-left),Y=p=>bottom-(Math.log10(Math.max(1e-8,p))+8)/8*(bottom-top);
 c.font='12px system-ui';c.fillStyle='#526674';c.strokeStyle='#dce3e8';
 for(const exponent of [-8,-6,-4,-2,0]){const y=Y(10**exponent);c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke();c.textAlign='right';c.fillText((10**(exponent+2)).toPrecision(1)+'%',left-7,y+4)}
 for(const n of [0,1,4,16,64,256,1024])if(n<=age){const x=left+Math.log1p(n)/lx*(right-left);c.textAlign='center';c.fillText(n,x,bottom+17)}
 for(const [values,color] of [[baseProb,'#087fbd'],[result.probabilities,'#9b427e']]){c.strokeStyle=color;c.lineWidth=1.2;c.globalAlpha=.8;c.beginPath();for(let k=age;k>=0;k--){if(k===age)c.moveTo(X(k),Y(values[k]));else c.lineTo(X(k),Y(values[k]))}c.stroke();c.globalAlpha=1;c.fillStyle=color;c.beginPath();c.arc(X(ki),Y(values[ki]),4,0,2*Math.PI);c.fill()}
 c.textAlign='center';c.fillStyle='#526674';c.fillText('Tokens before query (logarithmic spacing)',(left+right)/2,h-2);canvas.geom={left,right,lx,age};
}
function pick(k){mode='Q';anchor=qi;ki=k;pinned=true;rank=D.nativeTop[qi];calc();matches();render()}
function refresh(){
 const ready=ensure();for(const id of ['editPair','pairGain','pairGainNumber','mutePair','restorePair','resetPairs'])el(id).disabled=!ready;
 if(!ready){el('editedSummary').textContent='Choose a query with causal keys to try a pair edit.';el('editedKeys').replaceChildren();return}
 el('editPair').value=pair;el('pairGain').value=gains[pair];el('pairGainNumber').value=Number(gains[pair].toFixed(4));
 const original=pairs[ki*64+pair],adjusted=original*gains[pair],baselineOrder=Array.from({length:baseScores.length},(_,k)=>k).sort((a,b)=>baseScores[b]-baseScores[a]||a-b),baseRank=baselineOrder.indexOf(ki)+1;
 el('editedSummary').innerHTML='<b>Q '+esc(tok(D.positions[qi]))+' @ '+D.positions[qi]+' → K '+esc(tok(ki))+' @ '+ki+'</b><br>Pair '+pair+': contribution <b>'+num(original)+' → '+num(adjusted)+'</b> (gain '+num(gains[pair])+').<br>Total Q–K score: <b>'+num(baseScores[ki])+' → '+num(result.scores[ki])+'</b> · attention: <b>'+prob(baseProb[ki])+' → '+prob(result.probabilities[ki])+'</b> · key rank: <b>'+baseRank+' → '+result.ranks[ki]+'</b>.<br>'+result.active.length+' edited pair'+(result.active.length===1?'':'s')+' · top key: '+esc(tok(baselineOrder[0]))+' @ '+baselineOrder[0]+' → '+esc(tok(result.order[0]))+' @ '+result.order[0]+'.';
 el('editedKeys').innerHTML=result.order.slice(0,10).map((k,i)=>'<tr data-k="'+k+'"><td>'+(i+1)+'. '+esc(tok(k))+' @ '+k+'</td><td class="num">'+num(baseScores[k])+' → '+num(result.scores[k])+'</td><td class="num">'+prob(baseProb[k])+' → '+prob(result.probabilities[k])+'</td></tr>').join('');
 for(const row of el('editedKeys').querySelectorAll('tr'))row.onclick=()=>pick(+row.dataset.k);
 showBars();showProfile();
}
function setGain(value){if(!ensure()||!Number.isFinite(value))return;gains[pair]=Math.max(-5,Math.min(5,value));revision++;pinned=true;render()}
el('editPair').onchange=()=>{if(!ensure())return;pair=+el('editPair').value;el('pair').value=pair;pinned=true;render()};
el('pairGain').oninput=()=>setGain(+el('pairGain').value);el('pairGainNumber').oninput=()=>{if(el('pairGainNumber').value!=='')setGain(+el('pairGainNumber').value)};
el('mutePair').onclick=()=>setGain(0);el('restorePair').onclick=()=>setGain(1);el('resetPairs').onclick=()=>{if(!ensure())return;gains.fill(1);revision++;pinned=true;render()};
el('editedProfile').onclick=e=>{const g=el('editedProfile').geom;if(!g||!ensure())return;const x=e.clientX-el('editedProfile').getBoundingClientRect().left,n=Math.round(Math.expm1(Math.max(0,Math.min(1,(x-g.left)/(g.right-g.left)))*g.lx));pick(Math.max(0,g.age-n))};
const previousRender=render;render=function(){previousRender();refresh()};if(D)render();else refresh();
