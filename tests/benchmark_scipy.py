import argparse
import itertools
import timeit

import fsph
import matplotlib, matplotlib.pyplot as pp
import numpy as np
import scipy as sp, scipy.special

class Benchmark:
    def __init__(self, N, lmax, negative_m):
        self.N = N
        self.lmax = lmax
        self.negative_m = negative_m

        self.phis = np.random.uniform(0, np.pi, size=(N,))
        self.thetas = np.random.uniform(0, 2*np.pi, size=(N,))

class BenchmarkScipy(Benchmark):
    def __call__(self):
        result = []
        for l in range(self.lmax + 1):
            for m in range(l + 1):
                result.append(sp.special.sph_harm(m, l, self.thetas, self.phis)*(-1)**m)
            if self.negative_m:
                for m in range(1, l + 1):
                    result.append(sp.special.sph_harm(-m, l, self.thetas, self.phis))

        return result

class BenchmarkFSPH(Benchmark):
    def __call__(self):
        return fsph.pointwise_sph(self.phis, self.thetas, self.lmax, self.negative_m)

parser = argparse.ArgumentParser('Run scipy benchmarks')
parser.add_argument('-l', nargs='*', default=[12, 32, 64],
    help='Spherical harmonic l values to use')
parser.add_argument('-n', nargs='*', default=[1024, 2048, 4096],
    help='Number of points to compute for')
parser.add_argument('-r', '--replicas', type=int, default=3,
    help='Number of repeats to do for benchmarking purposes')
parser.add_argument('--negative-m', action='store_true',
    help='Compute spherical harmonics with negative m values')
parser.add_argument('-x', '--x-axis', default='lmax', choices=['n', 'lmax'],
    help='X-axis value to use')
parser.add_argument('--absolute', action='store_true',
    help='Plot absolute speed, rather than relative speedups')
parser.add_argument('-o', '--output',
    help='Output location')

def main(l, n, replicas, negative_m, x_axis, absolute, output):
    dsets = {}

    lmaxs = list(map(int, l))
    Ns = list(map(int, n))

    for (lmax, N) in itertools.product(lmaxs, Ns):
        time_scipy = timeit.timeit(
            stmt=BenchmarkScipy(N, lmax, negative_m), number=replicas)
        dsets[(N, lmax, 'scipy')] = time_scipy

        time_fsph = timeit.timeit(
            stmt=BenchmarkFSPH(N, lmax, negative_m), number=replicas)
        dsets[(N, lmax, 'fsph')] = time_fsph

    colors = pp.rcParams['axes.prop_cycle'].by_key()['color']

    if x_axis.lower() == 'n':
        for (lmax, color) in zip(lmaxs, colors):
            xs = Ns
            ys_fsph = [dsets[(N, lmax, 'fsph')] for N in xs]
            ys_scipy = [dsets[(N, lmax, 'scipy')] for N in xs]

            if absolute:
                pp.plot(xs, ys_fsph, label='$fsph_{{{}}}$'.format(lmax), color=color)
                pp.plot(xs, ys_scipy, '--', label='$scipy_{{{}}}$'.format(lmax), color=color)
            else:
                ys = np.array(ys_scipy)/np.array(ys_fsph)
                pp.plot(xs, ys, label='$lmax={}$'.format(lmax), color=color)

        pp.xlabel('N')
        if absolute:
            pp.ylabel('t/s')
        else:
            pp.ylabel('speedup')
        pp.legend(loc='best')

    elif x_axis.lower() == 'lmax':
        for (N, color) in zip(Ns, colors):
            xs = lmaxs
            ys_fsph = [dsets[(N, lmax, 'fsph')] for lmax in xs]
            ys_scipy = [dsets[(N, lmax, 'scipy')] for lmax in xs]

            if absolute:
                pp.plot(xs, ys_fsph, label='$fsph_{{{}}}$'.format(N), color=color)
                pp.plot(xs, ys_scipy, '--', label='$scipy_{{{}}}$'.format(N), color=color)
            else:
                ys = np.array(ys_scipy)/np.array(ys_fsph)
                pp.plot(xs, ys, label='$N={}$'.format(N), color=color)

        pp.xlabel('$l_{max}$')
        if absolute:
            pp.ylabel('t/s')
        else:
            pp.ylabel('speedup')
        pp.legend(loc='best')
    else:
        raise NotImplementedError(x_axis)

    if output:
        pp.savefig(output)
    else:
        pp.show()

if __name__ == '__main__': main(**vars(parser.parse_args()))
