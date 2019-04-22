import unittest

import hypothesis as hp, hypothesis.strategies as hps
import numpy as np
import scipy as sp, scipy.special
import fsph

def Ylm_scipy(phis, thetas, lmax):
    result = []
    for l in range(lmax + 1):
        for m in range(l + 1):
            result.append(sp.special.sph_harm(m, l, thetas, phis)*(-1)**m)
        for m in range(1, l + 1):
            result.append(sp.special.sph_harm(-m, l, thetas, phis))

    return np.transpose(result)

class TestScipy(unittest.TestCase):
    @hp.given(hps.integers(0, 64), hps.floats(0, np.pi, False, False),
              hps.floats(0, 2*np.pi, False, False))
    def test_values(self, lmax, phi, theta):
        phis = np.array([phi])
        thetas = np.array([theta])

        Ys_scipy = Ylm_scipy(phis, thetas, lmax)
        Ys_fsph = fsph.pointwise_sph(phis, thetas, lmax)

        np.testing.assert_allclose(Ys_fsph, Ys_scipy, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
