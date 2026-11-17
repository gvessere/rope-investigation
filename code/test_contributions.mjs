import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {buildPairScores,editAttention} from '../navigator-contributions.mjs';
const close=(a,b,tol=1e-9)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} differs from ${b}`);
function explicit(query,key,age,freq,gains){
 // Rotate modified query and key separately, then take their ordinary dot product.
 let score=0;
 for(let j=0;j<64;j++){
  const t=age*freq[j],g=gains[j],qx=query[j]*g,qy=query[j+64]*g;
  score+=(qx*Math.cos(t)-qy*Math.sin(t))*key[j]+(qx*Math.sin(t)+qy*Math.cos(t))*key[j+64];
 }
 return score/Math.sqrt(128);
}
const one=new Float64Array(64).fill(1),zero=new Float64Array(64),freq=Float64Array.from({length:64},(_,j)=>1e6**(-j/64));
const q=Float64Array.from({length:128},(_,i)=>Math.sin(i*1.7)),keys=Float64Array.from({length:5*128},(_,i)=>Math.cos(i*.31));
const pairs=buildPairScores(q,keys,4,freq),scores=Float64Array.from({length:5},(_,k)=>explicit(q,keys.subarray(k*128,k*128+128),4-k,freq,one)+.003),p=Float64Array.from(scores,x=>Math.exp(x));let z=p.reduce((a,b)=>a+b);for(let k=0;k<p.length;k++)p[k]/=z;
const savedScores=scores.slice(),savedPairs=pairs.slice();
assert.deepEqual(editAttention(scores,p,pairs,one).scores,scores);assert.deepEqual(editAttention(scores,p,pairs,one).probabilities,p);
for(const gain of [0,-1,2.5]){
 const g=one.slice();g[13]=gain;const r=editAttention(scores,p,pairs,g);
 for(let k=0;k<5;k++)close(r.scores[k],explicit(q,keys.subarray(k*128,k*128+128),4-k,freq,g)+.003);
 close(r.probabilities.reduce((a,b)=>a+b),1);assert.equal(r.order[0],r.scores.indexOf(Math.max(...r.scores)));
}
const removed=editAttention(scores,p,pairs,zero);for(let k=0;k<5;k++){close(removed.scores[k],.003);close(removed.probabilities[k],.2)}
assert.deepEqual(scores,savedScores);assert.deepEqual(pairs,savedPairs);
const huge=editAttention(new Float64Array([10000,-10000]),new Float64Array([1,0]),new Float64Array(128),zero);assert.deepEqual(huge.probabilities,new Float64Array([1,0]));
for(const file of ['d10-l4-h0.json.gz','d39-l8-h3.json.gz']){
 const d=JSON.parse(gunzipSync(readFileSync(new URL('../data/navigator/'+file,import.meta.url))));
 const decode=s=>{const b=Buffer.from(s,'base64');return new Float32Array(b.buffer,b.byteOffset,b.byteLength/4)};
 const allq=decode(d.q),keys=decode(d.k),allScores=decode(d.scores),allp=decode(d.probs);
 for(const qi of [0,127]){
  const pos=d.positions[qi],count=pos+1,q=allq.subarray(qi*128,qi*128+128),pairs=buildPairScores(q,keys,pos,d.freq),s=allScores.subarray(qi*d.n,qi*d.n+count),p=allp.subarray(qi*d.n,qi*d.n+count),g=one.slice();g[5]=0;g[32]=2;g[63]=-1;
  const r=editAttention(s,p,pairs,g);close(r.probabilities.reduce((a,b)=>a+b),1);
  for(const k of [0,pos-3,pos]){
   const key=keys.subarray(k*128,k*128+128),delta=explicit(q,key,pos-k,d.freq,g)-explicit(q,key,pos-k,d.freq,one);
   close(r.scores[k],s[k]+delta,1e-9);
  }
 }
}
console.log('Pair edits match explicitly rotated query vectors; reset, muting, sign reversal, softmax stability and captured-data checks pass.');
