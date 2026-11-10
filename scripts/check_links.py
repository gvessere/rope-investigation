"""Check local file targets and fragment links in the built static site."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parents[1]/'dist'
class Parser(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        for k in ['href','src']:
            if k in a:self.links.append(a[k])
parsed={}
for p in ROOT.glob('*.html'):
    parser=Parser();parser.feed(p.read_text());parsed[p.resolve()]=parser
errors=[]
for p,doc in parsed.items():
    for link in doc.links:
        u=urlsplit(link)
        if u.scheme or u.netloc:continue
        target=(p.parent/unquote(u.path)).resolve() if u.path else p
        if not target.exists():errors.append((p.name,link,'missing file'))
        elif u.fragment and target in parsed and unquote(u.fragment) not in parsed[target].ids:errors.append((p.name,link,'missing fragment'))
if errors:raise SystemExit(str(errors))
print('All local files and fragments resolve across',len(parsed),'pages.')
