// Shared radius limits for a pair, always using all heads in the selected layer.
export function pairMaxima(amplitude,start,count){
  const [heads,tokens,pairs]=amplitude.shape;
  if(heads!==12||pairs!==64||!Number.isInteger(start)||!Number.isInteger(count)||start<0||count<1||start+count>64)throw Error('Invalid query-pair range');
  const limits=new Float64Array(count);
  for(let h=0;h<heads;h++)for(let t=0;t<tokens;t++)for(let c=0;c<count;c++){
    const r=amplitude.values[(h*tokens+t)*pairs+start+c];
    if(!Number.isFinite(r)||r<0)throw Error('Invalid saved radius');
    limits[c]=Math.max(limits[c],r);
  }
  return Array.from(limits,v=>Math.max(v*1.035,1e-9));
}
