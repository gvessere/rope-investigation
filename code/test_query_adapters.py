import unittest
try:
 import torch
 from query_adapters import QueryAdapter
except ImportError:torch=None
@unittest.skipIf(torch is None,'Optional model dependencies not installed')
class AdapterTest(unittest.TestCase):
 def test_identity_and_geometry(self):
  x=torch.randn(2,3,8);q=torch.randn_like(x);omega=torch.tensor([1.,.1])
  for mode in ['phase','amplitude','combined','q_lora6','q_lora12','permuted']:
   a=QueryAdapter(1,2,8,omega,mode)
   torch.testing.assert_close(a(0,x,q),q)
   if mode in ['phase','permuted']:
    with torch.no_grad():a.phase.normal_(std=.1)
    z=a(0,x,q).reshape(2,3,2,4);before=q.reshape(2,3,2,4)
    torch.testing.assert_close(z[...,:2]**2+z[...,2:]**2,before[...,:2]**2+before[...,2:]**2)
   if mode=='amplitude':
    with torch.no_grad():a.B.normal_(std=.1)
    z=a(0,x,q);before=q.reshape(2,3,2,4);after=z.reshape_as(before)
    torch.testing.assert_close(after[...,:2]*before[...,2:],after[...,2:]*before[...,:2])
