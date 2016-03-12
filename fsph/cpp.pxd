# distutils: language = c++

cimport libcpp
from libcpp cimport bool

cdef extern from "../src/spherical_harmonics.hpp" namespace "fsph":
    cdef void evaluate_SPH[T](void*, unsigned int, T*, T*, unsigned int, bool) nogil
