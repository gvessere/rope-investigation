"""Export exact local navigator API responses for static GitHub Pages hosting.
Start the full local navigator, then run this script. No model inference.
python code/export_navigator.py --source http://127.0.0.1:8770
"""
from pathlib import Path
import argparse,json,urllib.request,gzip,hashlib
ROOT=Path(__file__).resolve().parents[1]
HEADS=[(2,5),(4,0),(8,3),(23,4)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',default='http://127.0.0.1:8770');args=ap.parse_args()
 d=ROOT/'data/navigator';d.mkdir(exist_ok=True)
 docs=json.load(urllib.request.urlopen(args.source+'/api/meta'))
 (d/'meta.json').write_text(json.dumps(docs,indent=2)+'\n');records=[]
 for doc in docs:
  for layer,head in HEADS:
   name=f'd{doc["id"]}-l{layer}-h{head}.json.gz';p=d/name
   if not p.exists():p.write_bytes(urllib.request.urlopen(f'{args.source}/api/head?doc={doc["id"]}&layer={layer}&head={head}',timeout=120).read())
   b=p.read_bytes();x=json.loads(gzip.decompress(b));assert (x['doc'],x['layer'],x['head'])==(doc['id'],layer,head)
   records.append(dict(file=name,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),audit=x['audit']))
  print('Exported passage',doc['id'],flush=True)
 (d/'manifest.json').write_text(json.dumps(dict(model='Qwen/Qwen2.5-1.5B-Instruct',scope='Four head selections across all 20 held-out passages; all 64 pairs, 128 queries and all causal keys retained. Exact API export, without quantization or subsampling.',heads=[dict(layer=l,head=h) for l,h in HEADS],documents=docs,source_note=x['source'],files=records),indent=2)+'\n')
 print('Exported',len(records),'head/passage payloads;',sum(x['bytes'] for x in records),'bytes')
if __name__=='__main__':main()
