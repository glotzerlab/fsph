# fsph

[![PyPI](https://img.shields.io/pypi/v/fsph.svg?style=flat)](https://pypi.org/project/fsph/)
[![ReadTheDocs](https://img.shields.io/readthedocs/fsph.svg?style=flat)](https://fsph.readthedocs.io/en/latest/)

fsph is a library to efficiently compute series of spherical harmonics (i.e. all of Y<sub>l</sub><sup>m</sup> for a set of l).

It is based on math enumerated by Martin J. Mohlenkamp at http://www.ohio.edu/people/mohlenka/research/uguide.pdf.

fsph uses portions of the cuda_complex project, located at
https://github.com/jtravs/cuda_complex.git . Its code and license are
located in the cuda_complex subdirectory.