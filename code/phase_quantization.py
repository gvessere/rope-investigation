"""The original small-radius, slow-key intervention, with portable backends.

Flat projected keys have shape [batch,tokens,256] = 2 KV heads × 128 dims.
Pairs are split-half (j,j+64); only j=49..63 are eligible. Phase coding preserves
radii and changes no query/value. `identity` reconstructs all slow pairs.
"""
import numpy as np


def round_keys_numpy(output,bits,protect_prefix=0):
    if bits not in (0,2,3,'identity'):raise ValueError('Use 0, 2, 3, or identity')
    if output.ndim!=3 or output.shape[-1]!=256:raise ValueError('Expected [batch,tokens,256] keys')
    z=output.reshape(output.shape[0],output.shape[1],2,128)
    x=z[...,49:64];y=z[...,113:128];r=np.sqrt(x*x+y*y);phi=np.arctan2(y,x)
    mask=np.ones_like(r,dtype=bool) if bits=='identity' else r<2.5
    mask[:,:protect_prefix]=False
    if bits==0:nx=r;ny=np.zeros_like(r)
    else:
        angle=phi if bits=='identity' else np.round(phi/(2*np.pi/2**bits))*(2*np.pi/2**bits)
        nx=r*np.cos(angle);ny=r*np.sin(angle)
    result=z.copy();result[...,49:64]=np.where(mask,nx,x);result[...,113:128]=np.where(mask,ny,y)
    return result.reshape(output.shape),int(mask.sum()),int(mask.size)


def round_keys_torch(output,bits,protect_prefix=0):
    import torch
    if bits not in (0,2,3,'identity'):raise ValueError('Use 0, 2, 3, or identity')
    if output.ndim!=3 or output.shape[-1]!=256:raise ValueError('Expected [batch,tokens,256] keys')
    z=output.reshape(output.shape[0],output.shape[1],2,128)
    x=z[...,49:64];y=z[...,113:128];r=torch.sqrt(x*x+y*y);phi=torch.atan2(y,x)
    mask=torch.ones_like(r,dtype=torch.bool) if bits=='identity' else r<2.5
    mask[:,:protect_prefix]=False
    if bits==0:nx=r;ny=torch.zeros_like(r)
    else:
        angle=phi if bits=='identity' else torch.round(phi/(2*np.pi/2**bits))*(2*np.pi/2**bits)
        nx=r*angle.cos();ny=r*angle.sin()
    result=z.clone();result[...,49:64]=torch.where(mask,nx,x);result[...,113:128]=torch.where(mask,ny,y)
    return result.reshape_as(output),int(mask.sum().cpu()),mask.numel()
