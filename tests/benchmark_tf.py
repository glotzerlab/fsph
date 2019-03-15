import argparse
import itertools
import timeit

import fsph, fsph.tf_ops
import matplotlib, matplotlib.pyplot as pp
import numpy as np
import tensorflow as tf

class Benchmark:
    def __init__(self, N, lmax, negative_m, compute_grad):
        self.N = N
        self.lmax = lmax
        self.negative_m = negative_m
        self.compute_grad = compute_grad

        self.phis = np.random.uniform(0, np.pi, size=(N,))
        self.thetas = np.random.uniform(0, 2*np.pi, size=(N,))

        self.inputs = np.array([self.phis, self.thetas], dtype=np.float32).T

class BenchmarkTF(Benchmark):
    _device_map = dict(cpu='/cpu:0', gpu='/device:GPU:0')

    def __init__(self, N, lmax, negative_m, compute_grad, device):
        super().__init__(N, lmax, negative_m, compute_grad)
        self.device = self._device_map[device]

    @staticmethod
    @tf.function
    def compute(x, lmax, negative_m, compute_grad):
        if compute_grad:
            return fsph.tf_ops.spherical_harmonic_series_grad(x, lmax, negative_m)
        else:
            return fsph.tf_ops.spherical_harmonic_series(x, lmax, negative_m)

    def __call__(self):
        with tf.device(self.device):
            return self.compute(self.inputs, self.lmax, self.negative_m, self.compute_grad)

class BenchmarkFSPH(Benchmark):
    def __call__(self):
        return fsph.pointwise_sph(self.phis, self.thetas, self.lmax, self.negative_m)

parser = argparse.ArgumentParser('Run tensorflow benchmarks')
parser.add_argument('-l', nargs='*', default=[12, 32, 64],
    help='Spherical harmonic l values to use')
parser.add_argument('--n-min', type=int, default=1024,
    help='Minimum number of points to use')
parser.add_argument('--n-max', type=int, default=16384,
    help='Maximum number of points to use')
parser.add_argument('--num-n', type=int, default=6,
    help='Number of values between --n-min and --n-max to sample')
parser.add_argument('-r', '--replicas', type=int, default=3,
    help='Number of repeats to do for benchmarking purposes')
parser.add_argument('--negative-m', action='store_true',
    help='Compute spherical harmonics with negative m values')
parser.add_argument('-g', '--compute-gradient', action='store_true',
    help='Compute the gradient, rather than the raw spherical harmonic values')
parser.add_argument('-o', '--output',
    help='Output location')

def main(l, n_min, n_max, num_n, replicas, negative_m, compute_gradient, output):
    dsets = {}

    lmaxs = list(map(int, l))
    Ns = np.linspace(n_min, n_max, num_n).astype(int)

    for (lmax, N) in itertools.product(lmaxs, Ns):
        if not compute_gradient:
            time_fsph = timeit.timeit(
                stmt=BenchmarkFSPH(N, lmax, negative_m, compute_gradient), number=replicas)
            dsets[(N, lmax, 'fsph')] = time_fsph

        time_cpu = timeit.timeit(
            stmt=BenchmarkTF(N, lmax, negative_m, compute_gradient, 'cpu'), number=replicas)
        dsets[(N, lmax, 'cpu')] = time_cpu

        time_gpu = timeit.timeit(
            stmt=BenchmarkTF(N, lmax, negative_m, compute_gradient, 'gpu'), number=replicas)
        dsets[(N, lmax, 'gpu')] = time_gpu

    for (lmax, color) in zip(lmaxs, pp.rcParams['axes.prop_cycle'].by_key()['color']):
        xs = Ns

        if (N, lmax, 'fsph') in dsets:
            ys = [dsets[(N, lmax, 'fsph')]/dsets[(N, lmax, 'cpu')] for N in Ns]
            pp.plot(xs, ys, '--', color=color)
            ys = [dsets[(N, lmax, 'fsph')]/dsets[(N, lmax, 'gpu')] for N in Ns]
            pp.plot(xs, ys, '-', label='l={}'.format(lmax), color=color)
        else:
            ys = [dsets[(N, lmax, 'cpu')]/dsets[(N, lmax, 'gpu')] for N in Ns]
            pp.plot(xs, ys, label='l={}'.format(lmax), color=color)

    pp.xlabel('N_points')
    pp.ylabel('speedup')
    pp.legend()

    if output:
        pp.savefig(output)
    else:
        pp.show()

if __name__ == '__main__': main(**vars(parser.parse_args()))
