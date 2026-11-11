"""Remove only small learned key-bias pairs in the slow band."""
import numpy as np

def remove_small_slow_biases(biases):
    z=np.asarray(biases)
    if z.shape[-1]!=64 or not np.iscomplexobj(z):
        raise ValueError('Expected complex coordinate pairs with final dimension 64')
    mask=np.zeros(z.shape,dtype=bool)
    mask[...,49:64]=np.abs(z[...,49:64])<2.5
    edited=z.copy();edited[mask]=0
    return edited,mask
