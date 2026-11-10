import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {readNPZ,percentile,selectRange} from '../scatter-data.mjs';
const root=new URL('../',import.meta.url);
const fixtures=JSON.parse(await readFile(new URL('code/viewer-fixtures.json',root),'utf8'));
assert.ok(Math.abs(percentile([0,1,2,3,100]) - 96.12)<1e-10);
assert.throws(()=>selectRange([1,2,3],{auto:false,min:5,max:2}));
for(const kind of ['queries','keys']){
  const bytes=await readFile(new URL(`data/observations/${kind}_layer_00.npz`,root));
  const data=await readNPZ(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
  const fixture=fixtures[kind];assert.deepEqual(data.amplitude.shape,fixture.shape);
  const a=data.amplitude.values,p=data.phase.values,n=fixture.shape[1]*64;
  for(const [field,values] of [['amplitude',a],['phase',p]]){
    [0,Math.floor(values.length/2),values.length-1].forEach((index,i)=>assert.equal(values[index],fixture[field+'_samples'][i]));
  }
  for(let h=0;h<fixture.shape[0];h++){
    const values=a.subarray(h*n,(h+1)*n),ph=p.subarray(h*n,(h+1)*n),target=fixture.heads[h];
    const range=selectRange(values);assert.ok(Math.abs(range.max-target.r99)<1e-5);
    const full=selectRange(values,{auto:false,max:null});assert.equal(full.max,target.max);
    let automatic=0,manual=0,window=0;
    for(let i=0;i<n;i++){
      if(values[i]<=range.max){automatic++;if(ph[i]>=Math.PI/4&&ph[i]<=3*Math.PI/4)window++;}
      if(values[i]>=2&&values[i]<=8)manual++;
    }
    assert.equal(automatic,target.auto_shown);assert.equal(manual,target.manual_shown);assert.equal(window,target.window_shown);
  }
}
console.log('NPZ decoding, per-head 99th percentiles, full/manual limits, and phase-window counts match NumPy for all first-layer query and key heads.');
