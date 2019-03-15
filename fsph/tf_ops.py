
import tensorflow as tf
from tensorflow.python.framework import ops

from . import _fsph
so_name = _fsph.__file__.replace('/_fsph.', '/_tf_ops.')

all_ops = tf.load_op_library(so_name)

spherical_harmonic_series = all_ops.spherical_harmonic_series
spherical_harmonic_series_grad = all_ops.spherical_harmonic_series_grad
