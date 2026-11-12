import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {readNPZ} from '../scatter-data.mjs';
import {pairMaxima} from '../head-pair-data.mjs';
const root=new URL('../',import.meta.url);
const fixtures=JSON.parse(await readFile(new URL('code/head-pair-fixtures.json',root),'utf8'));
for(const layer of [0,14]){
 const b=await readFile(new URL(`data/observations/queries_layer_${String(layer).padStart(2,'0')}.npz`,root));
 const d=await readNPZ(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
 for(const start of [0,16,32,48]){
  const limits=pairMaxima(d.amplitude,start,16);
  limits.forEach((v,c)=>assert.ok(Math.abs(v-fixtures[layer][start+c])<1e-10));
 }
}
assert.throws(()=>pairMaxima({shape:[12,1,64],values:new Float32Array(768)},60,16));
console.log('All 64 per-pair column scales match NumPy across all 12 heads in layers 0 and 14.');
