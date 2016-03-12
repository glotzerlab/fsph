#!/usr/bin/env python

import os, subprocess, sys
from distutils.command.build_ext import build_ext
from distutils.core import Extension, setup
import numpy

macros = []
extra_args = []
sources = []

if '--cython' in sys.argv:
    from Cython.Build import cythonize
    sys.argv.remove('--cython')

    def myCythonize(macros, *args, **kwargs):
        result = cythonize(*args, **kwargs)
        for r in result:
            r.define_macros.extend(macros)
            r.include_dirs.append(numpy.get_include())
            r.extra_compile_args.extend(extra_args)
            r.extra_link_args.extend(extra_args)
            r.sources.extend(sources)

        return result

    modules = myCythonize(macros, 'fsph/_fsph.pyx')
else:
    sources.append('fsph/_fsph.cpp')
    modules = [Extension('fsph._fsph', sources=sources,
                         define_macros=macros, extra_compile_args=extra_args,
                         extra_link_args=extra_args, include_dirs=[numpy.get_include()])]

setup(name='fsph',
      version='0.1',
      description='Fast sequential spherical harmonics calculation',
      author='Matthew Spellings',
      author_email='mspells@umich.edu',
      url='',
      packages=['fsph'],
      ext_modules=modules
)
