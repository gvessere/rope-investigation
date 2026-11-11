import unittest
import numpy as np
from small_bias_removal import remove_small_slow_biases
class BiasRemovalTest(unittest.TestCase):
    def test_band_threshold_and_copy(self):
        z=np.ones((2,64),dtype=complex);z[:,50]=2.5;z[:,51]=3j
        edited,mask=remove_small_slow_biases(z)
        self.assertEqual(int(mask.sum()),26)
        np.testing.assert_array_equal(edited[:,:49],z[:,:49])
        np.testing.assert_array_equal(edited[:,50:52],z[:,50:52])
        self.assertTrue(np.all(edited[mask]==0));self.assertTrue(np.all(z[:,49]==1))
