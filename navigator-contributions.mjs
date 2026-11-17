// Frozen-vector pair weighting: gains scale both coordinates of a query pair.
export function buildPairScores(query,keys,queryPosition,freq){
 if(query.length!==128||freq.length!==64||keys.length%128)throw Error('Invalid pair-vector shape');
 const count=queryPosition+1;if(count>keys.length/128)throw Error('Query lies outside captured keys');
 const out=new Float64Array(count*64),scale=1/Math.sqrt(128);
 for(let k=0;k<count;k++)for(let j=0;j<64;j++){
  const qx=query[j],qy=query[j+64],kx=keys[k*128+j],ky=keys[k*128+j+64],t=freq[j]*(queryPosition-k);
  out[k*64+j]=((qx*kx+qy*ky)*Math.cos(t)+(qx*ky-qy*kx)*Math.sin(t))*scale;
 }
 return out;
}
export function editAttention(scores,probabilities,pairs,gains){
 const count=scores.length;if(probabilities.length!==count||pairs.length!==count*64||gains.length!==64)throw Error('Invalid score shape');
 if(!Array.from(gains).every(Number.isFinite))throw Error('Pair gains must be finite');
 const edited=Float64Array.from(scores),p=new Float64Array(count),active=[];
 for(let j=0;j<64;j++)if(gains[j]!==1)active.push(j);
 if(!active.length)p.set(probabilities);
 else {
  for(let k=0;k<count;k++)for(const j of active)edited[k]+=(gains[j]-1)*pairs[k*64+j];
  let max=-Infinity;for(const s of edited)max=Math.max(max,s);
  let sum=0;for(let k=0;k<count;k++){p[k]=Math.exp(edited[k]-max);sum+=p[k]}
  for(let k=0;k<count;k++)p[k]/=sum;
 }
 const order=Array.from({length:count},(_,k)=>k).sort((a,b)=>edited[b]-edited[a]||a-b),ranks=new Int32Array(count);
 order.forEach((k,i)=>ranks[k]=i+1);
 return {scores:edited,probabilities:p,order,ranks,active};
}
