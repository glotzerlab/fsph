#!/usr/bin/env python

import distutils
from distutils.dep_util import newer
from distutils.command.build_ext import build_ext
from distutils.core import Extension, setup
from distutils import log
import os
import shutil
import subprocess
import sys
import numpy

with open('fsph/version.py') as version_file:
    exec(version_file.read())

macros = []
extra_args = []
sources = []

CYTHONIZE = False
if '--cython' in sys.argv:
    from Cython.Build import cythonize
    sys.argv.remove('--cython')
    CYTHONIZE = True

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

class CustomBuildCommand(build_ext):
    """Custom build command that compiles the CUDA tensorflow module and incorporates it into the final extension.

    This command should not change default behavior if tensorflow is
    not enabled or nvcc is not found. The nvcc binary can be set via
    the NVCC environment variable.

    If enabling cuda is undesirable but would otherwise be enabled by
    default, set the NVCC environment variable to a name that does not
    exist on $PATH.
    """
    def run(self, *args, **kwargs):
        NVCC = os.environ.get('NVCC', 'nvcc')
        found_nvcc = distutils.spawn.find_executable(NVCC) is not None

        for ext in self.extensions:
            if '_tf_ops' in ext.name and found_nvcc:
                src_name = 'src/tensorflow_op_gpu.cu'
                output_location = os.path.join(self.build_temp, 'tensorflow_op_gpu.cu.o')

                if newer(src_name, output_location):
                    os.makedirs(self.build_temp, 0o755, exist_ok=True)
                    command = [NVCC, '-c', '-o', output_location, src_name,
                               '-D', 'GOOGLE_CUDA=1', '-Xcompiler', '-fPIC']
                    command.extend(ext.extra_compile_args)

                    if self.verbose:
                        command.append('--resource-usage')

                    log.info(' '.join(command))
                    subprocess.check_call(command)

                    ext.extra_objects.append(output_location)

        super().run(*args, **kwargs)

try:
    import tensorflow as tf

    ext = Extension('fsph._tf_ops',
                    sources=['src/tensorflow_op.cpp'],
                    extra_compile_args=tf.sysconfig.get_compile_flags(),
                    extra_link_args=tf.sysconfig.get_link_flags())
    modules.append(ext)
except ImportError:
    # skip building tensorflow component
    pass

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
      cmdclass={
        'build_ext': CustomBuildCommand,
      },
      ext_modules=modules,
      install_requires=['numpy'],
      packages=['fsph'],
      project_urls={
          'Documentation': 'http://fsph.readthedocs.io/',
          'Source': 'https://github.com/glotzerlab/fsph'
          },
      url='http://fsph.readthedocs.io',
)
