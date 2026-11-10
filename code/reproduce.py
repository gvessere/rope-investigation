"""Deterministic theory illustration, visibly separate from original empirical figures."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rope import frequencies, fit_curve, design

ROOT=Path(__file__).resolve().parents[1]


def main():
    # 16 real parameters for each approximation: DC + 7 sin/cos pairs + Nyquist.
    train=np.arange(128);test=np.arange(256)
    target=lambda x: np.exp(-.5*((x-80)/8)**2)
    y=target(train)
    def dft_basis(x):
        a=2*np.pi*np.asarray(x)[:,None]*np.arange(1,8)[None]/128
        return np.column_stack((np.ones(len(x)),np.cos(a),np.sin(a),np.cos(np.pi*np.asarray(x))))
    fourier=np.linalg.lstsq(dft_basis(train),y,rcond=1e-10)[0]
    omega=frequencies(16)
    coeff,rank,singular=fit_curve(train,y,omega)
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    reports={}
    for ax,xx,title in zip(axs,[train,test],['Fit interval','Extrapolation (shaded)']):
        ax.plot(xx,target(xx),color='black',label='Gaussian target, center 80')
        ax.plot(xx,np.einsum('ij,j->i',dft_basis(xx),fourier,optimize=False),label='16-column Fourier basis')
        ax.plot(xx,np.einsum('ij,j->i',design(xx,omega),coeff,optimize=False),label='16-column geometric RoPE basis')
        if len(xx)>128:
            ax.axvspan(128,255,color='#d1c6b8',alpha=.3)
            ax.set_yscale('symlog',linthresh=.1)
            ax.set_ylabel('Score (symmetric log scale; linear near zero)')
        ax.set(title=title,xlabel='Relative displacement (samples)',ylabel='Score (arbitrary units)')
        ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.suptitle('New deterministic illustration · fixed coefficients · B=10,000 · SVD cutoff 1e−10')
    folder=ROOT/'assets/illustrations';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'dft_rope.png',dpi=160);plt.close(fig)
    for name,basis,c in [('fourier',dft_basis,fourier),('rope',lambda x:design(x,omega),coeff)]:
        reports[name]=dict(train_rmse=float(np.sqrt(np.mean((np.einsum('ij,j->i',basis(train),c,optimize=False)-y)**2))),
                           extrapolation_rmse=float(np.sqrt(np.mean((np.einsum('ij,j->i',basis(test[128:]),c,optimize=False)-target(test[128:]))**2))))
    reports['rope'].update(effective_rank=rank,singular_values=singular.tolist(),coefficient_l2=float(np.linalg.norm(coeff)))
    (ROOT/'data/illustration_metrics.json').write_text(json.dumps(reports,indent=2)+'\n')
    return reports


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
