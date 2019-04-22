import unittest

import hypothesis as hp, hypothesis.strategies as hps
import numpy as np
import tensorflow as tf
import fsph, fsph.tf_ops

class TestTensorflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = tf.Session()

    @hp.given(hps.integers(0, 64), hps.booleans(),
              hps.floats(0, np.pi, False, False),
              hps.floats(0, 2*np.pi, False, False))
    def test_basic(self, lmax, negative_m, phi, theta):
        phis = np.array([phi], dtype=np.float32)
        thetas = np.array([theta], dtype=np.float32)

        Ys_fsph = fsph.pointwise_sph(phis, thetas, lmax, negative_m)

        inputs = np.array([phis, thetas]).T

        Ys_tf = self.session.run(
            fsph.tf_ops.spherical_harmonic_series(inputs, lmax, negative_m))

        self.assertEqual(Ys_fsph.shape, Ys_tf.shape)
        np.testing.assert_allclose(Ys_fsph, Ys_tf, atol=1e-4)

    @hp.given(hps.integers(0, 12), hps.booleans(),
              hps.floats(.1, np.pi - .1, False, False),
              hps.floats(.1, 2*np.pi - .1, False, False))
    def test_numeric_gradient(self, lmax, negative_m, phi, theta):
        phis = np.array([phi])
        thetas = np.array([theta])

        Y0 = fsph.pointwise_sph(phis, thetas, lmax, negative_m)
        grad_numeric = []
        for dim in range(2):
            dx = 1e-5
            if dim == 0:
                Y = fsph.pointwise_sph(phis + dx, thetas, lmax, negative_m)
            else:
                Y = fsph.pointwise_sph(phis, thetas + dx, lmax, negative_m)

            dY = Y - Y0
            grad_numeric.append(dY/dx)

        grad_numeric = np.transpose(grad_numeric, (1, 2, 0))

        inputs = np.array([phis, thetas]).T

        grad_tf = self.session.run(
            fsph.tf_ops.spherical_harmonic_series_grad(inputs, lmax, negative_m))

        np.testing.assert_allclose(grad_numeric, grad_tf, atol=1e-2)

if __name__ == '__main__':
    unittest.main()
