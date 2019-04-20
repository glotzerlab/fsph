import unittest

import numpy as np
import tensorflow as tf
import fsph, fsph.tf_ops

class TestTensorflow(unittest.TestCase):
    def test_basic(self):
        N = 8
        lmax = 8

        np.random.seed(13)

        for negative_m in [False, True]:
            phis = np.random.uniform(0, np.pi, size=(N,)).astype(np.float32)
            thetas = np.random.uniform(0, 2*np.pi, size=(N,)).astype(np.float32)

            Ys_fsph = fsph.pointwise_sph(phis, thetas, lmax, negative_m)

            inputs = np.array([phis, thetas]).T

            with tf.Session() as ses:
                Ys_tf = fsph.tf_ops.spherical_harmonic_series(inputs, lmax, negative_m).eval()

            self.assertEqual(Ys_fsph.shape, Ys_tf.shape)
            self.assertTrue(np.allclose(Ys_fsph, Ys_tf))

    def test_numeric_gradient(self):
        N = 8
        lmax = 8

        np.random.seed(13)

        for negative_m in [False, True]:
            phis = np.random.uniform(0, np.pi - .1, size=(N,))
            thetas = np.random.uniform(0, 2*np.pi - .1, size=(N,))

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

            with tf.Session() as ses:
                grad_tf = fsph.tf_ops.spherical_harmonic_series_grad(inputs, lmax, negative_m).eval()

            self.assertTrue(np.allclose(grad_numeric, grad_tf, atol=1e-2, rtol=1e-2))

if __name__ == '__main__':
    unittest.main()
