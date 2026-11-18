import assert from 'node:assert/strict';
import fs from 'node:fs';
import {profileValues, chartSvg} from '../all-head-distance.mjs';
const data = JSON.parse(fs.readFileSync(new URL('../data/decay/all-head-distance.json', import.meta.url)));
assert.equal(data.heads.length, 336);
for (const [i, record] of data.heads.entries()) {
  assert.equal(record.layer, Math.floor(i/12)); assert.equal(record.head, i%12);
  assert.equal(record.per_key.length, 7);
  assert.ok(record.range_mass > 0 && record.range_mass <= 1);
  const mass = profileValues(record, 'paired');
  assert.ok(Math.abs(mass.reduce((a,b) => a+b,0)-record.range_mass*100) < 1e-10);
  profileValues(record, 'mean').forEach((value, j) => {
    assert.ok(Math.abs(value*data.widths[j]*100-mass[j]) < 1e-10);
  });
  for (const mode of ['mean','paired']) {
    const svg = chartSvg(data, record, mode);
    assert.ok(!svg.includes('NaN') && !svg.includes('Infinity'));
    assert.equal((svg.match(/<circle /g)||[]).length, 7);
    assert.equal(svg.includes('stroke-dasharray'), mode === 'mean');
  }
}
// A tiny tail must remain visibly labelled as a tiny part of full attention.
assert.ok(data.heads[13*12+4].range_mass < .005);
assert.ok(profileValues(data.heads[13*12+4], 'paired').reduce((a,b)=>a+b,0) < .5);
// Check absolute mass against the saved native softmax values, not a rescaled shape.
assert.ok(Math.abs(data.heads[6*12+7].per_key[0] - 0.0004551336610241955) < 1e-10);
assert.ok(data.heads[6*12+9].per_key.at(-1) > data.heads[6*12+9].per_key[0]);
assert.throws(() => profileValues(data.heads[0], 'historical'));
console.log('All 336 profiles: full attention units, finite plots and tiny-tail scope verified.');
