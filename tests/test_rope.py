import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from rope import frequencies, rotate, score, quantize_phase, softmax, fit_curve, design, fit_decay


class RoPETests(unittest.TestCase):
    def test_relative_rotation_and_sine_sign(self):
        rng = np.random.default_rng(20261109)
        q, k = rng.normal(size=(2,64,2))
        w = frequencies()
        for m,n in [(0,1),(17,3),(-5,40),(2048,8192)]:
            direct = (rotate(q,m,w)*rotate(k,n,w)).sum()/np.sqrt(128)
            self.assertAlmostEqual(direct, float(score(q,k,n-m,w)), places=10)
        # q=(1,0),k=(0,1): dot(q,R(delta)k)=-sin(delta).
        self.assertAlmostEqual(float(score(np.array([[1.,0]]),np.array([[0.,1]]),.3,np.array([1.]))),-np.sin(.3)/np.sqrt(2))

    def test_phase_rounding_seam_radius_and_selection(self):
        angles = np.linspace(-np.pi,np.pi,1001)
        r = np.linspace(.01,5,1001)
        p = r[:,None]*np.stack((np.cos(angles),np.sin(angles)),1)
        for bits in [0,2,3]:
            z = quantize_phase(p,bits,2.5)
            np.testing.assert_allclose(np.linalg.norm(z,axis=1),r,atol=1e-14)
            np.testing.assert_array_equal(z[r>=2.5],p[r>=2.5])
            if bits:
                err = np.angle(np.exp(1j*(np.arctan2(z[:,1],z[:,0])-angles)))
                self.assertLessEqual(abs(err[r<2.5]).max(),np.pi/2**bits+1e-14)
            else:
                np.testing.assert_allclose(z[r<2.5,1],0,atol=1e-14)

    def test_dft_exactness_and_restricted_basis(self):
        rng=np.random.default_rng(9);y=rng.normal(size=64)
        np.testing.assert_allclose(np.fft.ifft(np.fft.fft(y)).real,y,atol=1e-13)
        x=np.arange(64);w=frequencies(16)
        coef,rank,_=fit_curve(x,y,w)
        self.assertLessEqual(rank,16)
        self.assertGreater(np.linalg.norm(np.einsum('ij,j->i',design(x,w),coef,optimize=False)-y),1)

    def test_softmax_mixing_and_output_differential(self):
        s=np.array([2.,1.9,-1.]);v=np.array([[1.,4.],[3.,0.],[-2.,1.]])
        p=softmax(s);np.testing.assert_allclose(p,softmax(s+1000))
        e=np.array([.3,-.8,.6]);h=1e-6;o=p@v
        derivative=((softmax(s+h*e)@v)-(softmax(s-h*e)@v))/(2*h)
        np.testing.assert_allclose(derivative,(p*e)@(v-o),atol=1e-9)

    def test_known_power_laws_with_bin_exposure(self):
        lo=2**np.arange(3,10);hi=2*lo-1
        for exponent in [0.,1.,2.]:
            mass=np.array([(np.arange(a,b+1,dtype=float)**(-exponent)).sum() for a,b in zip(lo,hi)])
            fitted,r2=fit_decay(mass,lo,hi)
            self.assertAlmostEqual(float(fitted),exponent,delta=.002)


if __name__ == '__main__':
    unittest.main()
