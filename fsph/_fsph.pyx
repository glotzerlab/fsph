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
@cython.wraparound(False)
def pointwise_sph(phi, theta, lmax, negative_m=True):
    """Evaluate a series of spherical harmonics on an array of spherical coordinates.

    The array objects phi and theta should have the same length and
    can hold single- or double-precision floating point numbers. The
    resulting array will be of length (N_coordinates, N_sphs) where
    N_coordinates is the length of the given coordinate arrays.

    To map the columns of the result array to particular (l, m)
    values, see :py:func:`get_LMs`.

    :param phi: Array-like object of polar angles in [0, pi]
    :param theta: Array-like object of azimuthal angles in [0, 2*pi]
    :param lmax: Integer maximum spherical harmonic degree to compute (inclusive)
    :param negative_m: Set to False to disable the negative-m spherical harmonics

    """
    phi = np.ascontiguousarray(phi)
    theta = np.ascontiguousarray(theta)
    lmax = int(lmax)

    if phi.dtype in [np.int32, np.int64]:
        phi = phi.astype(np.float32)
    if theta.dtype in [np.int32, np.int64]:
        theta = theta.astype(np.float32)

    assert phi.shape == theta.shape and phi.dtype == theta.dtype
    assert phi.dtype in [np.float32, np.float64] and theta.dtype in [np.float32, np.float64]

    sphCount = (lmax + 1)*(lmax + 2)//2 + (lmax*(lmax + 1)//2 if negative_m else 0)
    resultShape = phi.shape + (sphCount,)

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

@cython.boundscheck(False)
@cython.wraparound(False)
def pointwise_sph_grads(phi, theta, lmax, negative_m=True, return_sphs=True):
    """Evaluate a spherical harmonic gradients on an array of spherical coordinates.

    The array objects phi and theta should have the same length and
    can hold single- or double-precision floating point numbers. The
    resulting array will be of length (N_coordinates, N_sphs, 2) where
    N_coordinates is the length of the given coordinate arrays and the
    last dimension corresponds to (phi_gradient, theta_gradient).

    To map the columns of the result array to particular (l, m)
    values, see :py:func:`get_LMs`.

    :param phi: Array-like object of polar angles in [0, pi]
    :param theta: Array-like object of azimuthal angles in [0, 2*pi]
    :param lmax: Integer maximum spherical harmonic degree to compute (inclusive)
    :param negative_m: Set to False to disable the negative-m spherical harmonics
    :param return_sphs: If True, return the spherical harmonics (as from :py:func:`pointwise_sph`) in addition to the gradients as a (gradients, spherical_harmonics) tuple

    """
    phi = np.ascontiguousarray(phi)
    theta = np.ascontiguousarray(theta)
    lmax = int(lmax)

    if phi.dtype in [np.int32, np.int64]:
        phi = phi.astype(np.float32)
    if theta.dtype in [np.int32, np.int64]:
        theta = theta.astype(np.float32)

    assert phi.shape == theta.shape and phi.dtype == theta.dtype
    assert phi.dtype in [np.float32, np.float64] and theta.dtype in [np.float32, np.float64]

    sphCount = (lmax + 1)*(lmax + 2)//2 + (lmax*(lmax + 1)//2 if negative_m else 0)
    resultShape = phi.shape + (sphCount, 2)
    sphShape = resultShape[:len(resultShape) - 1]

    cdef float[:] phi_f;
    cdef float[:] theta_f;
    cdef float complex[:] result_f;
    cdef float complex[:] sph_f;
    cdef float complex *sph_target_f = NULL;

    cdef double[:] phi_d;
    cdef double[:] theta_d;
    cdef double complex[:] result_d;
    cdef double complex[:] sph_d;
    cdef double complex *sph_target_d = NULL;

    sphs = None

    if phi.dtype == np.float32:
        result = np.empty(resultShape, dtype=np.complex64)
        if return_sphs:
            sphs = np.empty(sphShape, dtype=np.complex64)
            sph_f = sphs.ravel()
            sph_target_f = &sph_f[0]
        phi_f = phi.ravel()
        theta_f = theta.ravel()
        result_f = result.ravel()
        cpp.evaluate_SPH_with_grads[float](&result_f[0], sph_target_f, lmax, &phi_f[0], &theta_f[0], phi.size, negative_m)
    else:
        result = np.empty(resultShape, dtype=np.complex128)
        if return_sphs:
            sphs = np.empty(sphShape, dtype=np.complex128)
            sph_d = sphs.ravel()
            sph_target_d = &sph_d[0]
        phi_d = phi.ravel()
        theta_d = theta.ravel()
        result_d = result.ravel()
        cpp.evaluate_SPH_with_grads[double](&result_d[0], sph_target_d, lmax, &phi_d[0], &theta_d[0], phi.size, negative_m)

    if return_sphs:
        return result, sphs

    return result

def get_LMs(lmax, negative_m=True):
    """Returns the (l, m) indices in the order that they are exposed by fsph.

    Creates a (N_sphs, 2) array where the first column corresponds to
    the l values and the second column corresponds to the m values for
    any index in the series.

    """
    ls = []
    ms = []

    for l in range(lmax + 1):
        ls.extend((l + 1)*[l])
        ms.extend(range(l + 1))

        if negative_m:
            ls.extend(l*[l])
            ms.extend([-m for m in range(1, l + 1)])

    return np.array([ls, ms], dtype=np.int64).T
