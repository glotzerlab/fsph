import functools
import unittest

import hypothesis as hp, hypothesis.strategies as hps
import hypothesis.extra.numpy as hpn
import numpy as np
import tensorflow as tf
import fsph, fsph.tf_ops

class TestTensorflow(unittest.TestCase):
    @hp.settings(deadline=None, print_blob=True)
    @hp.given(hps.integers(0, 64), hps.booleans(),
              hpn.arrays(np.float32, hpn.array_shapes(max_dims=1),
                         hps.floats(0, np.float32(np.pi), False, False, width=32)),
              hpn.arrays(np.float32, hpn.array_shapes(max_dims=1),
                         hps.floats(0, np.float32(2*np.pi), False, False, width=32)))
    def test_basic(self, lmax, negative_m, phis, thetas):
        phis = phis[:min(len(phis), len(thetas))]
        thetas = thetas[:min(len(phis), len(thetas))]

        Ys_fsph = fsph.pointwise_sph(phis, thetas, lmax, negative_m)

        inputs = np.array([phis, thetas]).T

        Ys_tf = fsph.tf_ops.spherical_harmonic_series(inputs, lmax, negative_m)

        self.assertEqual(Ys_fsph.shape, Ys_tf.shape)
        np.testing.assert_allclose(Ys_fsph, Ys_tf, atol=1e-4)

    @hp.settings(deadline=None, print_blob=True)
    @hp.given(hps.integers(0, 12), hps.booleans(),
              hpn.arrays(np.float32, hpn.array_shapes(max_dims=1),
                         hps.floats(np.float32(.1),
                                    np.float32(np.pi - .1), False, False, width=32)),
              hpn.arrays(np.float32, hpn.array_shapes(max_dims=1),
                         hps.floats(np.float32(.1),
                                    np.float32(2*np.pi - .1), False, False, width=32)))
    def test_numeric_gradient(self, lmax, negative_m, phis, thetas):
        phis = phis[:min(len(phis), len(thetas))]
        thetas = thetas[:min(len(phis), len(thetas))]

        Y0 = fsph.pointwise_sph(phis, thetas, lmax, negative_m)
        grad_numeric = []
        for dim in range(2):
            dx = 1e-3
            if dim == 0:
                Y = fsph.pointwise_sph(phis + dx, thetas, lmax, negative_m)
            else:
                Y = fsph.pointwise_sph(phis, thetas + dx, lmax, negative_m)

            dY = Y - Y0
            grad_numeric.append(dY/dx)

        grad_numeric = np.transpose(grad_numeric, (1, 2, 0))

        inputs = np.array([phis, thetas]).T

        grad_tf = fsph.tf_ops.spherical_harmonic_series_grad(inputs, lmax, negative_m)

        np.testing.assert_allclose(grad_numeric, grad_tf, atol=5e-2)

    @hp.settings(deadline=None, print_blob=True)
    @hp.given(hps.integers(0, 12), hps.booleans(),
              hpn.arrays(np.float32, hpn.array_shapes(max_dims=1),
                         hps.floats(np.float32(.1),
                                    np.float32(np.pi - .1), False, False, width=32)),
              hpn.arrays(np.float32, hpn.array_shapes(max_dims=1),
                         hps.floats(np.float32(.1),
                                    np.float32(2*np.pi - .1), False, False, width=32)))
    def test_tf_gradient(self, lmax, negative_m, phis, thetas):
        phis = phis[:min(len(phis), len(thetas))]
        thetas = thetas[:min(len(phis), len(thetas))]
        inputs = np.array([phis, thetas]).T

        test_fun = functools.partial(
            fsph.tf_ops.spherical_harmonic_series,
            lmax=lmax, negative_m=negative_m)

        (exact, numeric) = tf.test.compute_gradient(test_fun, [inputs])

        np.testing.assert_allclose(exact, numeric, atol=5e-2)

if __name__ == '__main__':
    unittest.main()
