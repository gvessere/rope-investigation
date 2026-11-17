from pathlib import Path
import os,json,base64,gzip,functools
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import argparse
import numpy as np
from tokenizers import Tokenizer
ap=argparse.ArgumentParser(description='Serve the full local RoPE navigator from original captures.')
ap.add_argument('--captures',type=Path,required=True,help='Original rope_decay_mechanisms directory')
ap.add_argument('--tokenizer',type=Path,required=True,help='Local Qwen tokenizer.json')
ap.add_argument('--port',type=int,default=8770)
args=ap.parse_args()
P=args.captures
UI=Path(__file__).resolve().parents[1]/'navigator-local.html'
tok=Tokenizer.from_file(str(args.tokenizer))
manifest=json.loads((P/'manifest.json').read_text());docs=[dict(id=d['id'],title=d['title'],genre=d['genre']) for d in manifest if d['split']=='heldout'];freq=np.load(P/'inv_freq.npy').astype(float)
def enc(a):return base64.b64encode(np.ascontiguousarray(a,dtype='<f4').tobytes()).decode()
@functools.lru_cache(maxsize=2)
def capture(doc):
 f=P/f'N2048_doc{doc:02}'
 with np.load(f/'qk_pre.npz') as a,np.load(f/'rows.npz') as b:
  q=a['q'];k=a['k'];rows={n:b[n] for n in ['input_ids','query_positions','logZ','top_ages','top_probabilities']}
 text=[tok.decode([int(i)],skip_special_tokens=False) for i in rows['input_ids']]
 S=np.load(f/'scores_age0_1023.npy',mmap_mode='r')
 return q,k,rows,text,S
@functools.lru_cache(maxsize=3)
def payload(doc,l,h):
 q,k,rows,text,saved=capture(doc);q=q[l,h].astype(float);k=k[l,h//6].astype(float);positions=rows['query_positions'];n=len(k)
 def rotate(v,pos):
  a=np.asarray(pos)[:,None]*freq;c=np.cos(a);s=np.sin(a);x=v[:,:64];y=v[:,64:];return np.concatenate((x*c-y*s,x*s+y*c),axis=1)
 qr=rotate(q,positions);kr=rotate(k,np.arange(n));s=qr@kr.T/np.sqrt(128);causal=np.arange(n)[None,:]<=positions[:,None]
 ki=positions[:,None]-np.arange(1024)[None,:];qi=np.arange(len(q))[:,None];error=float(abs(s[qi,ki]-saved[l,h]).max());s[qi,ki]=saved[l,h]
 top=positions[:,None]-rows['top_ages'][l,h];s[qi,top]=np.log(rows['top_probabilities'][l,h])+rows['logZ'][l,h,:,None]
 probs=np.where(causal,np.exp(s-rows['logZ'][l,h,:,None]),0);s[~causal]=-1e9
 result=dict(doc=doc,layer=l,head=h,kv=h//6,n=n,nq=len(q),positions=positions.tolist(),tokens=text,freq=freq.tolist(),q=enc(q),k=enc(k),scores=enc(s),probs=enc(probs),nativeTop=top.tolist(),audit=dict(max_reconstruction_error_recent=error,max_row_mass_error=float(abs(probs.sum(-1)-1).max())),source='Recent 1024-key scores and native top-16 probabilities are saved native measurements. Other older scores are reconstructed from captured pre-RoPE vectors; pair contributions use that reconstruction.')
 return gzip.compress(json.dumps(result,separators=(',',':')).encode())
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  u=urlparse(self.path)
  try:
   if u.path=='/api/meta':data=json.dumps(docs).encode();typ='application/json';z=False
   elif u.path=='/api/head':
    p=parse_qs(u.query);d=int(p['doc'][0]);l=int(p['layer'][0]);h=int(p['head'][0]);assert d in [x['id'] for x in docs] and 0<=l<28 and 0<=h<12
    data=payload(d,l,h);typ='application/json';z=True
   elif u.path in ['/','/explorer.html']:data=UI.read_bytes();typ='text/html';z=False
   else:self.send_error(404);return
   self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-cache')
   if z:self.send_header('Content-Encoding','gzip')
   self.end_headers();self.wfile.write(data)
  except Exception as e:self.send_error(500,str(e));print(repr(e),flush=True)
print(f'Pair explorer: http://127.0.0.1:{args.port}',flush=True)
ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
