# distutils: language = c++
# cython: embedsignature=True

cimport cython
from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as np
from cpython cimport PyObject, Py_INCREF
from libcpp cimport bool, complex

cimport cpp

np.import_array()

@cython.boundscheck(False)
def pointwise_sph(phi, theta, lmax, negative_m=True):
    phi = np.ascontiguousarray(phi).reshape((-1,))
    theta = np.ascontiguousarray(theta).reshape((-1,))
    lmax = int(lmax)

    if phi.dtype in [np.int32, np.int64]:
        phi = phi.astype(np.float32)
    if theta.dtype in [np.int32, np.int64]:
        theta = theta.astype(np.float32)

    assert phi.shape == theta.shape and phi.dtype == theta.dtype
    assert phi.dtype in [np.float32, np.float64] and theta.dtype in [np.float32, np.float64]

    sphCount = (lmax + 1)*(lmax + 2)//2 + (lmax*(lmax + 1)//2 if negative_m else 0)
    resultShape = (phi.size, sphCount)

    cdef float[:] phi_f;
    cdef float[:] theta_f;
    cdef float complex[:] result_f;

    cdef double[:] phi_d;
    cdef double[:] theta_d;
    cdef double complex[:] result_d;

    if phi.dtype == np.float32:
        result = np.empty(resultShape, dtype=np.complex64)
        phi_f = phi.ravel()
        theta_f = theta.ravel()
        result_f = result.ravel()
        cpp.evaluate_SPH[float](&result_f[0], lmax, &phi_f[0], &theta_f[0], phi.size, negative_m)
    else:
        result = np.empty(resultShape, dtype=np.complex128)
        phi_d = phi.ravel()
        theta_d = theta.ravel()
        result_d = result.ravel()
        cpp.evaluate_SPH[double](&result_d[0], lmax, &phi_d[0], &theta_d[0], phi.size, negative_m)

    return result
