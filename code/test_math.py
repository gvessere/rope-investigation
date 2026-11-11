"""Independent checks of the mathematical claims illustrated in the post."""
import unittest
import numpy as np
from figures import basis, frequencies, fit_rope


class FourierChecks(unittest.TestCase):
    def test_real_dft_recovers_arbitrary_samples(self):
        y = np.random.default_rng(7).normal(size=128)
        spectrum = np.fft.rfft(y)
        self.assertEqual(len(spectrum), 65)
        np.testing.assert_allclose(np.fft.irfft(spectrum, n=128), y, atol=1e-14)

    def test_rope_expansion_matches_rotated_dot_products(self):
        rng = np.random.default_rng(2)
        q, k = rng.normal(size=(2, 64, 2))
        omega = frequencies()
        a = np.sum(q * k, axis=1)
        b = q[:, 0] * k[:, 1] - q[:, 1] * k[:, 0]
        zq = q[:, 0] + 1j*q[:, 1]
        zk = k[:, 0] + 1j*k[:, 1]
        np.testing.assert_allclose(zq*np.conj(zk), a-1j*b, atol=1e-14)
        def rotate(v, p):
            c, s = np.cos(p * omega), np.sin(p * omega)
            return np.column_stack((c*v[:, 0] - s*v[:, 1],
                                    s*v[:, 0] + c*v[:, 1]))
        for m, n in ((0, 0), (17, 97), (97, 17), (117, 197)):
            expected = np.sum(rotate(q, m) * rotate(k, n)) / np.sqrt(128)
            actual = np.sum(basis([m - n], omega)[0] * np.r_[a, b]) / np.sqrt(128)
            complex_score = np.real(np.sum(zq*np.conj(zk)*np.exp(1j*omega*(m-n)))) / np.sqrt(128)
            polar_score = np.sum(abs(zq)*abs(zk)*np.cos(np.angle(zq)-np.angle(zk)+omega*(m-n))) / np.sqrt(128)
            self.assertAlmostEqual(polar_score, expected, places=12)
            self.assertAlmostEqual(complex_score, expected, places=12)
            self.assertAlmostEqual(actual, expected, places=12)

    def test_coordinated_phases_translate_the_whole_score(self):
        rng = np.random.default_rng(12)
        omega = frequencies()
        a, b = rng.normal(size=(2, len(omega)))
        positions = np.arange(256)
        for shift in (-32, 0, 17, 32):
            c, s = np.cos(omega * shift), np.sin(omega * shift)
            shifted_coeff = np.r_[a*c-b*s, a*s+b*c]
            direct = np.einsum("ij,j->i", basis(positions-shift, omega), np.r_[a,b])
            rephased = np.einsum("ij,j->i", basis(positions, omega), shifted_coeff)
            np.testing.assert_allclose(rephased, direct, atol=1e-12)

    def test_frequency_ratio_and_regularized_fit(self):
        omega = frequencies()
        np.testing.assert_allclose(omega[1:] / omega[:-1], 10000**(-2/128))
        x = np.arange(256)
        y = np.exp(-.5 * ((x - 80) / 8)**2)
        prediction, coeff = fit_rope(x, y)
        self.assertLess(np.sqrt(np.mean((prediction-y)**2)), .03)
        self.assertTrue(np.all(np.isfinite(coeff)))
        design = basis(x, omega)
        # First-order optimality for the stated regularized least squares fit.
        residual = prediction - y
        gradient = np.einsum("ij,i->j", design, residual) + 1e-3 * coeff
        np.testing.assert_allclose(gradient, 0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
