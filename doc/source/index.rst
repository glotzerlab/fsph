.. fsph documentation master file, created by
   sphinx-quickstart on Tue Sep 11 14:27:06 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fsph's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

fsph is a library to compute series of complex spherical harmonics.

.. note::

   The Condon-Shortley phase is not included, although it is
   often used by many other libraries; to incorporate it, multiply the
   positive-`m` spherical harmonics by :math:`(-1)^m`.

Installation
============

Install from PyPI::

  pip install --no-build-isolation fsph

Or from source::

  git clone https://github.com/glotzerlab/fsph.git
  cd fsph
  python setup.py install

API Reference
=============

.. automodule:: fsph
   :members: pointwise_sph, get_LMs

Tensorflow Operations
---------------------

As of version 0.2, fsph can also compute spherical harmonic series of
points on the CPU and GPU using tensorflow. This module is
automatically built when tensorflow is found while installing
fsph. GPU support is enabled when CUDA (specifically, the `nvcc`
binary) is found while installing fsph.

.. py:function:: fsph.tf_ops.spherical_harmonic_series(inputs, lmax, negative_m)

   Compute a spherical harmonic series for a set of input points.

   :param inputs: (..., 2) array of (phi, theta) values
   :param lmax: Maximum spherical harmonic l to compute
   :param negative_m: If True, compute for negative as well as positive m values

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
