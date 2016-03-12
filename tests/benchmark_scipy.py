from collections import defaultdict
import timeit
import matplotlib, matplotlib.pyplot as pp

setup = """
import numpy as np
import scipy as sp, scipy.special
import fsph

N = {N}
lmax = {lmax}

phis = np.random.uniform(-np.pi, np.pi, size=(N,))
thetas = np.random.uniform(0, 2*np.pi, size=(N,))

def Ylm_scipy(phis, thetas, lmax):
    result = []
    for l in range(lmax):
        for m in range(-l, l+1):
            result.append(sp.special.sph_harm(m, l, phis, thetas))

    return result

def Ylm_fsph(phis, thetas, lmax):
    result = fsph.pointwise_sph(phis, thetas, lmax)
    return result
"""

dsets = {}

lmaxs = [4, 6, 8, 10, 12, 14,16, 20, 24, 32, 48, 64]
Ns = [1000]

for lmax in lmaxs:
    for N in Ns:
        time_scipy = timeit.timeit(stmt='Ylm_scipy(phis, thetas, lmax)', setup=setup.format(N=N, lmax=lmax), number=10)
        time_fsph = timeit.timeit(stmt='Ylm_fsph(phis, thetas, lmax)', setup=setup.format(N=N, lmax=lmax), number=10)

        dsets[(N, lmax, 'scipy')] = time_scipy
        dsets[(N, lmax, 'fsph')] = time_fsph

mode = 'x_lmax'
if mode == 'x_N':
    for lmax in lmaxs:
        xs = list(sorted({N for (N, lmax_, typ) in dsets if lmax_ == lmax and typ == 'fsph'}))
        ys = [dsets[(N, lmax, 'fsph')] for N in xs]
        pp.plot(xs, ys, label='$fsph_{{{}}}$'.format(lmax))

        xs = list(sorted({N for (N, lmax_, typ) in dsets if lmax_ == lmax and typ == 'scipy'}))
        ys = [dsets[(N, lmax, 'scipy')] for N in xs]
        pp.plot(xs, ys, '--', label='$scipy_{{{}}}$'.format(lmax))

    pp.xlabel('N')
    pp.ylabel('t/s')
    pp.legend(loc='best')
elif mode == 'x_lmax':
    for N in Ns:
        xs = list(sorted({lmax for (N_, lmax, typ) in dsets if N_ == N and typ == 'fsph'}))
        ys = [dsets[(N, lmax, 'fsph')] for lmax in xs]
        pp.plot(xs, ys, label='$fsph_{{{}}}$'.format(N))

        xs = list(sorted({lmax for (N_, lmax, typ) in dsets if N_ == N and typ == 'scipy'}))
        ys = [dsets[(N, lmax, 'scipy')] for lmax in xs]
        pp.plot(xs, ys, '--', label='$scipy_{{{}}}$'.format(N))

    pp.xlabel('$l_{max}$')
    pp.ylabel('t/s')
    pp.legend(loc='best')

pp.savefig('/tmp/fsph_benchmark.png')
