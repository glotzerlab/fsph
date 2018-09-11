#!/usr/bin/env python

import os, subprocess, sys
from distutils.command.build_ext import build_ext
from distutils.core import Extension, setup
import numpy

with open('fsph/version.py') as version_file:
    exec(version_file.read())

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
      version=__version__,
      description='Fast sequential spherical harmonics calculation',
      author='Matthew Spellings',
      author_email='mspells@umich.edu',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Mathematics',
      ],
      ext_modules=modules,
      install_requires=['numpy'],
      packages=['fsph'],
      project_urls={
          'Documentation': 'http://fsph.readthedocs.io/',
          'Source': 'https://bitbucket.org/glotzer/fsph'
          },
      url='http://fsph.readthedocs.io',
)
