import assert from 'node:assert/strict';
import fs from 'node:fs';
import {profileValues, axisBounds, chartSvg} from '../all-head-distance.mjs';
const data=JSON.parse(fs.readFileSync(new URL('../data/decay/all-head-distance.json',import.meta.url)));
assert.equal(data.heads.length,336);
assert.deepEqual(data.bins.slice(0,8),Array.from({length:8},(_,i)=>[i,i]));
assert.deepEqual(data.bins.at(-1),[2048,2060]);
assert.equal(data.query_count,2560);
for(const [i,record] of data.heads.entries()) {
  assert.equal(record.layer,Math.floor(i/12));assert.equal(record.head,i%12);
  assert.equal(record.per_key.length,17);
  const mass=profileValues(record,'paired');
  assert.ok(Math.abs(mass.reduce((a,b)=>a+b,0)-100)<1e-8);
  profileValues(record,'mean').forEach((v,j)=>assert.ok(Math.abs(v*data.effective_widths[j]-mass[j])<1e-10));
  for(const mode of ['mean','paired'])for(const scale of ['auto','shared','linear']) {
    const values=profileValues(record,mode),[lo,hi]=axisBounds(values,scale);
    assert.ok(values.every(v=>v>=lo&&v<=hi),'Autoscale must show every native point');
    const svg=chartSvg(data,record,mode,true,scale);
    assert.ok(!svg.includes('NaN')&&!svg.includes('Infinity'));
    assert.equal((svg.match(/<circle /g)||[]).length,17);
    assert.ok(svg.includes('stroke-dasharray'));
    assert.equal(record.reference_mass[0],null);
  }
}
// The prior distant-tail normalization hid a dominant near-token peak here.
const peak=data.heads[13*12+4];
assert.ok(Math.abs(peak.mass[1]-.7466246704856285)<1e-12);
assert.ok(peak.mass.slice(8,15).reduce((a,b)=>a+b,0)<.005);
// Last-layer trends should no longer be flattened by an unrelated 1e-12 global floor.
const last=profileValues(data.heads[27*12],'mean'), bounds=axisBounds(last);
assert.ok(bounds[1]/bounds[0]<1e5);
assert.throws(()=>profileValues(data.heads[0],'historical'));
console.log('All 336 heads: 100% mass, self/recent/far coverage, per-key opportunity accounting and unclipped autoscale verified.');
