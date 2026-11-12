"""Query-only adapters applied before native RoPE; no changes to key projections."""
import math
import torch
from torch import nn
from torch.nn import functional as F
class QueryAdapter(nn.Module):
    def __init__(self,layers,heads,dim,omega,mode,seed=1101):
        super().__init__();self.heads=heads;self.dim=dim;self.dh=dim//heads;self.mode=mode
        gen=torch.Generator().manual_seed(seed)
        self.register_buffer('omega',omega.detach().clone())
        if mode in ['phase','permuted','combined']:
            self.phase=nn.Parameter(torch.zeros(layers,heads,dim))
        if mode=='permuted':
            perm=torch.randperm(len(omega),generator=torch.Generator().manual_seed(seed+700000))
            self.omega=self.omega[perm.to(omega.device)]
        if mode in ['amplitude','combined','q_lora6','q_lora12']:
            rank=8 if mode in ['amplitude','combined'] else int(mode[6:])
            outputs=heads*self.dh//2 if mode in ['amplitude','combined'] else dim
            self.A=nn.Parameter((torch.rand(layers,rank,dim,generator=gen)*2-1)/math.sqrt(dim))
            self.B=nn.Parameter(torch.zeros(layers,outputs,rank))
    def forward(self,layer,x,out):
        if self.mode.startswith('q_lora'):
            return out+F.linear(F.linear(x,self.A[layer]),self.B[layer])
        q=out.reshape(*out.shape[:-1],self.heads,self.dh)
        if self.mode in ['phase','permuted','combined']:
            delta=2*torch.tanh(F.linear(x,self.phase[layer])/2)
            angle=-delta.unsqueeze(-1)*self.omega
            a,b=q.chunk(2,-1);q=torch.cat([a*angle.cos()-b*angle.sin(),a*angle.sin()+b*angle.cos()],-1)
        if self.mode in ['amplitude','combined']:
            z=F.linear(F.linear(x,self.A[layer]),self.B[layer]).reshape(*x.shape[:-1],self.heads,self.dh//2)
            gain=(2*torch.tanh(z/2)).exp()
            q=q*torch.cat([gain,gain],-1)
        return q.reshape_as(out)
