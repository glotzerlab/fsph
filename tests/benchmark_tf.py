import itertools
import timeit
import numpy as np
import matplotlib, matplotlib.pyplot as pp

setup = """
import fsph, fsph.tf_ops
import numpy as np
import tensorflow as tf

N = {N}
lmax = {lmax}
negative_m = {negative_m}

if {use_gpu}:
    device_name = '/device:GPU:0'
else:
    device_name = '/cpu:0'

phis = np.random.uniform(0, np.pi, size=(N,))
thetas = np.random.uniform(0, 2*np.pi, size=(N,))

inputs = np.array([phis, thetas]).T

with tf.device(device_name):
    placeholder = tf.placeholder('float32', inputs.shape)
    if {compute_grad}:
        sphs = fsph.tf_ops.spherical_harmonic_series_grad(placeholder, lmax, negative_m)
    else:
        sphs = fsph.tf_ops.spherical_harmonic_series(placeholder, lmax, negative_m)

session = tf.Session()
"""

statement = 'session.run(sphs, feed_dict={placeholder: inputs})'

dsets = {}

lmaxs = [12, 32, 64]
Ns = np.linspace(1024, 16384, 6).astype(int)
compute_grad = True

for (lmax, N) in itertools.product(lmaxs, Ns):
    time_cpu = timeit.timeit(
        stmt=statement, number=5,
        setup=setup.format(use_gpu=False, N=N, lmax=lmax, negative_m=True, compute_grad=compute_grad))
    dsets[(N, lmax, 'cpu')] = time_cpu

    time_gpu = timeit.timeit(
        stmt=statement, number=5,
        setup=setup.format(use_gpu=True, N=N, lmax=lmax, negative_m=True, compute_grad=compute_grad))
    dsets[(N, lmax, 'gpu')] = time_gpu

for lmax in lmaxs:
    xs = Ns
    ys = [dsets[(N, lmax, 'cpu')]/dsets[(N, lmax, 'gpu')] for N in Ns]

    pp.plot(xs, ys, label='l={}'.format(lmax))

pp.xlabel('N_points')
pp.ylabel('speedup')
pp.legend()
pp.savefig('/tmp/fsph_cpu_gpu_benchmark.png')
