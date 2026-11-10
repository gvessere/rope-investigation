import fs from 'node:fs';
import path from 'node:path';
import katex from 'katex';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'dist');fs.rmSync(out,{recursive:true,force:true});fs.mkdirSync(out,{recursive:true});
const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const json=p=>JSON.parse(read(p));
const chapters=[
 ['overview','How I started exploring RoPE','The geometry, frequency ladder, phase conventions, and attention as a mixer.','Start here'],
 ['fourier','Can attention build a shape?','A finite sinusoidal basis, recovered constructions, and a numerical approximation comparison.','Expressivity'],
 ['measurements','How much phase information matters?','Original theta/r evidence, including zero-, two-, and three-bit interventions.','Measurements'],
 ['decay','Following attention into the past','Matched-history measurements and the gap between geometric scales and a power-law mechanism.','Theory & evidence'],
 ['reproduce','Code, figures, and experiment notes','Tested portable code, original measurements, and the boundaries of verification.','Code & data']
];
const nav='<nav aria-label="Primary"><a href="index.html">Start here</a><a href="reproduce.html">Code & data</a><a href="https://gvessere.github.io/">About Gery ↗</a></nav>';
function shell(title,body){return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="${esc(title)} — Gery Vessere’s research on rotary position embeddings, with original figures and reproducible code."><title>${esc(title)} · RoPE investigation</title><link rel="stylesheet" href="assets/style.css"></head><body><a class="skip" href="#main">Skip to content</a><div class="shell"><header class="masthead"><a class="brand" href="index.html">RoPE investigation<span>Gery Vessere · Research notes</span></a>${nav}</header>${body}<footer class="site-footer"><span>Gery Vessere · Research notes · November 2026</span><span><a href="https://github.com/gvessere/rope-investigation">Source on GitHub ↗</a> · <a href="reproduce.html">Provenance & reproduction</a></span></footer></div></body></html>`}
function math(body){return body.replace(/\\\[([\s\S]*?)\\\]/g,(_,tex)=>katex.renderToString(tex,{output:'mathml',displayMode:true,throwOnError:true})).replace(/\\\(([\s\S]*?)\\\)/g,(_,tex)=>katex.renderToString(tex,{output:'mathml',displayMode:false,throwOnError:true}));}
function code(body){return body.replace(/\{\{code:([^:}]+):([^}]+)\}\}/g,(_,p,fn)=>{const lines=read(p).split('\n');const start=lines.findIndex(x=>x.startsWith(`def ${fn}(`));if(start<0)throw Error(fn);let end=lines.findIndex((x,i)=>i>start&&x.startsWith('def '));if(end<0)end=lines.length;return `<pre><code>${esc(lines.slice(start,end).join('\n').trim())}</code></pre>`;});}
function table(headers,rows){return `<div class="table-wrap"><table><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map((x,i)=>`<td${i?' class="num"':''}>${x}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
const slow=json('data/slow_actual_key_phases/results.json').results;
const slowRows=[['zero_all','0 bits · all slow pairs'],['zero_small','0 bits · slow r&lt;2.5'],['pi2_all','2 bits · all slow pairs'],['pi2_small','2 bits · slow r&lt;2.5'],['pi4_all','3 bits · all slow pairs'],['pi4_small','3 bits · slow r&lt;2.5']].map(([key,label])=>[label,slow[key].loss_increase.toFixed(5),(100*slow[key].changed_prediction_fraction).toFixed(2)+'%']);
const comp=json('data/key_quantization_small_radius/comparison.json');
const codecRows=['hybrid_smallp3r4_largep6r6','hybrid_smallp4r4_largep6r6','prefix14_p6r4'].map(key=>{const x=comp.find(x=>x.name===key);if(!x)throw Error(key);return [esc(x.label),x.stored_K_bytes_per_key.toFixed(2),x.loss_increase.toFixed(5),(100*x.changed_prediction_fraction).toFixed(2)+'%']});
for(const [i,[slug,title]] of chapters.entries()){
 let body=read(`content/${slug}.html`).replace('{{slow_table}}',table(['Phase intervention','Δ NLL, nats/token','Top-one changed'],slowRows)).replace('{{codec_table}}',table(['Codec','Bytes/key','Δ NLL','Top-one changed'],codecRows));body=math(code(body));body=body.replaceAll('</figcaption>',' · <a href="reproduce.html#figures">Regenerate this figure</a></figcaption>');if(body.includes('{{'))throw Error('Unresolved template');
 const headings=[...body.matchAll(/<h2 id="([^"]+)">([\s\S]*?)<\/h2>/g)];
 const toc=`<aside class="toc" aria-label="On this page"><p>On this page</p>${headings.map(h=>`<a href="#${h[1]}">${h[2]}</a>`).join('')}<div class="series"><p>The series</p>${chapters.map(([s,t],j)=>`<a href="${s}.html"${s===slug?' aria-current="page"':''}>0${j+1} · ${t}</a>`).join('')}</div></aside>`;
 fs.writeFileSync(path.join(out,slug+'.html'),shell(title,`<main id="main"><header class="article-header"><p class="eyebrow">Exploring RoPE · Part ${i+1}</p><h1>${title}</h1><p class="meta">Gery Vessere · November 9, 2026</p></header><div class="article-layout">${toc}<article>${body}</article></div></main>`));
}
// Begin with the story itself, rather than a directory of articles.
fs.copyFileSync(path.join(out,'overview.html'),path.join(out,'index.html'));
for(const folder of ['assets','data','code','tests'])fs.cpSync(path.join(root,folder),path.join(out,folder),{recursive:true,filter:p=>!p.includes('__pycache__')});
fs.copyFileSync(path.join(root,'requirements.txt'),path.join(out,'requirements.txt'));fs.writeFileSync(path.join(out,'.nojekyll'),'');
console.log(`Built ${chapters.length+1} pages in dist/ with native MathML.`);
