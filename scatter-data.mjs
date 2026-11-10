// Read NumPy's saved NPZ arrays using the browser's native gzip decoder.
// ZIP deflate payloads are wrapped in gzip with their recorded CRC and size.
export async function readNPZ(buffer) {
  const bytes = new Uint8Array(buffer), v = new DataView(buffer);
  let end = bytes.length - 22;
  while (end >= Math.max(0, bytes.length - 65557) && v.getUint32(end, true) !== 0x06054b50) end--;
  if (end < 0 || v.getUint32(end, true) !== 0x06054b50) throw Error('Invalid NPZ directory');
  const count = v.getUint16(end+10,true); let cursor = v.getUint32(end+16,true);
  const result = {};
  for (let i=0;i<count;i++) {
    if (v.getUint32(cursor,true)!==0x02014b50) throw Error('Invalid ZIP entry');
    const method=v.getUint16(cursor+10,true), crc=v.getUint32(cursor+16,true);
    const packed=v.getUint32(cursor+20,true), size=v.getUint32(cursor+24,true);
    const names=v.getUint16(cursor+28,true), extra=v.getUint16(cursor+30,true), comment=v.getUint16(cursor+32,true);
    const local=v.getUint32(cursor+42,true);
    const name=new TextDecoder().decode(bytes.subarray(cursor+46,cursor+46+names));
    const start=local+30+v.getUint16(local+26,true)+v.getUint16(local+28,true);
    let raw;
    if (method===8) {
      const gzip=new Uint8Array(10+packed+8);
      gzip.set([31,139,8,0,0,0,0,0,0,255]); gzip.set(bytes.subarray(start,start+packed),10);
      const footer=new DataView(gzip.buffer);footer.setUint32(10+packed,crc,true);footer.setUint32(14+packed,size,true);
      raw=await new Response(new Blob([gzip]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    } else if (method===0) raw=bytes.slice(start,start+packed).buffer;
    else throw Error('Unsupported ZIP compression');
    const data=new Uint8Array(raw), dv=new DataView(raw);
    if (data[0]!==147 || new TextDecoder().decode(data.subarray(1,6))!=='NUMPY') throw Error('Invalid NPY data');
    const version=data[6], headerStart=version===1?10:12;
    const length=version===1?dv.getUint16(8,true):dv.getUint32(8,true);
    const header=new TextDecoder().decode(data.subarray(headerStart,headerStart+length));
    if (!/['"]descr['"]\s*:\s*['"]<f4['"]/.test(header) || !/['"]fortran_order['"]\s*:\s*False/.test(header)) throw Error('Expected C-order float32 arrays');
    const shape=header.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/)[1].split(',').map(s=>s.trim()).filter(Boolean).map(Number);
    const values=new Float32Array(raw.slice(headerStart+length));
    if (shape.reduce((a,b)=>a*b,1)!==values.length) throw Error('Array shape mismatch');
    result[name.replace(/\.npy$/,'')]={shape,values};
    cursor+=46+names+extra+comment;
  }
  return result;
}

export function percentile(values, fraction=.99) {
  if (!values.length) throw Error('Empty percentile input');
  const sorted=Float32Array.from(values).sort(), index=(sorted.length-1)*fraction;
  const lower=Math.floor(index), weight=index-lower;
  return sorted[lower]*(1-weight)+sorted[Math.min(lower+1,sorted.length-1)]*weight;
}

export function selectRange(values,{auto=true,min=0,max=null}={}) {
  let upper=auto?percentile(values):max;
  if (upper===null || upper==='') {
    upper=0;for (const value of values) upper=Math.max(upper,value);
  }
  const lower=auto?0:Number(min);
  if (!Number.isFinite(lower)||!Number.isFinite(upper)||lower<0||upper<=lower) throw Error('Use 0 ≤ r min < r max.');
  return {min:lower,max:upper};
}
