import unittest
import numpy as np
from phase_quantization import round_keys_numpy

class PhaseRoundingChecks(unittest.TestCase):
    def test_radii_scope_and_prefix_are_preserved(self):
        raw=np.random.default_rng(11).normal(size=(1,24,256)).astype('float32')
        z=raw.reshape(1,24,2,128);r=np.hypot(z[...,49:64],z[...,113:128])
        for bits in [0,2,3,'identity']:
            output,_,_=round_keys_numpy(raw,bits,14);v=output.reshape(z.shape)
            np.testing.assert_array_equal(output[:,:14],raw[:,:14])
            np.testing.assert_array_equal(v[...,:49],z[...,:49]);np.testing.assert_array_equal(v[...,64:113],z[...,64:113])
            np.testing.assert_allclose(np.hypot(v[...,49:64],v[...,113:128]),r,rtol=2e-6,atol=1e-6)
            if bits!='identity':
                mask=r>=2.5
                np.testing.assert_array_equal(v[...,49:64][mask],z[...,49:64][mask])
                np.testing.assert_array_equal(v[...,113:128][mask],z[...,113:128][mask])
    def test_identity_and_zero_have_distinct_meanings(self):
        x=np.ones((1,2,256),dtype='float32')
        identity,_,_=round_keys_numpy(x,'identity')
        np.testing.assert_allclose(identity,x,atol=2e-7)
        zero,_,_=round_keys_numpy(x,0)
        self.assertEqual(zero.reshape(1,2,2,128)[0,0,0,113],0)
        self.assertAlmostEqual(zero.reshape(1,2,2,128)[0,0,0,49],np.sqrt(2),places=6)

if __name__=='__main__':unittest.main()
