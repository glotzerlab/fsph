import unittest

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
    def test_values(self):
        N = 8
        lmax = 12

        np.random.seed(13)

        phis = np.random.uniform(0, np.pi, size=(N,))
        thetas = np.random.uniform(0, 2*np.pi, size=(N,))

        Ys_scipy = Ylm_scipy(phis, thetas, lmax)
        Ys_fsph = fsph.pointwise_sph(phis, thetas, lmax)

        self.assertTrue(np.allclose(Ys_scipy, Ys_fsph))

if __name__ == '__main__':
    unittest.main()
