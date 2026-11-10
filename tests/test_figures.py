"""Enforce the user requirement: every displayed figure has a runnable recipe."""
from pathlib import Path
from html.parser import HTMLParser
import json
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]


class Images(HTMLParser):
    def __init__(self):super().__init__();self.sources=set()
    def handle_starttag(self,tag,attrs):
        if tag=='img':self.sources.add(dict(attrs)['src'])


class FigureTests(unittest.TestCase):
    def test_all_displayed_figures_regenerate(self):
        recipes=json.loads((ROOT/'data/figure_recipes.json').read_text())
        if any(not (ROOT/r['regenerated']).exists() for r in recipes):
            subprocess.run([sys.executable,str(ROOT/'code/figures.py')],check=True,cwd=ROOT)
        expected=set()
        for page in (ROOT/'content').glob('*.html'):
            parser=Images();parser.feed(page.read_text());expected |= parser.sources
        mapped={r['original'] for r in recipes}
        self.assertEqual(expected,mapped)
        self.assertEqual(len(expected),11)
        for recipe in recipes:
            for relative in recipe['inputs']+[recipe['code'],recipe['regenerated']]:
                self.assertTrue((ROOT/relative).is_file(),relative)
            self.assertGreater((ROOT/recipe['regenerated']).stat().st_size,10000)


if __name__=='__main__':unittest.main()
