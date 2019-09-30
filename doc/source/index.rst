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

  pip install fsph

Or from source::

  git clone https://github.com/glotzerlab/fsph.git
  cd fsph
  python setup.py install

API Reference
=============

.. automodule:: fsph
   :members: pointwise_sph, get_LMs

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
