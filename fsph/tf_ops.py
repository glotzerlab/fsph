
import tensorflow as tf
from tensorflow.python.framework import ops

from . import _fsph
so_name = _fsph.__file__.replace('/_fsph.', '/_tf_ops.')

all_ops = tf.load_op_library(so_name)

spherical_harmonic_series = all_ops.spherical_harmonic_series
spherical_harmonic_series_grad = all_ops.spherical_harmonic_series_grad

@ops.RegisterGradient('SphericalHarmonicSeries')
def _spherical_harmonic_series_grad(op, grad):
    # input_grad:: (..., Nsphs, 2)
    input_grad = spherical_harmonic_series_grad(*op.inputs)

    # grad:: (..., Nsphs) -> (..., Nsphs, 1)
    grad = tf.expand_dims(grad, -1)

    # result::(..., 2, 1)
    result = (tf.linalg.matmul(tf.math.conj(input_grad), grad, transpose_a=True) +
              tf.linalg.matmul(input_grad, tf.math.conj(grad), transpose_a=True))
    # result -> (..., 2)
    result = tf.math.real(tf.squeeze(result, -1))*0.5

    return [result, None, None]
