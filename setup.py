from setuptools       import setup, Extension
from Cython.Build     import cythonize
from Cython.Distutils import build_ext

perslocsig = Extension(name="statmapper",
                       sources=["statmapper/perslocsig.pyx"],
                       language="c++",
                       extra_compile_args=["-Wmaybe-uninitialized", "-Wunused-but-set-variable", "-lboost_filesystem", "-std=c++11"])

setup(name="statmapper",
      author="Mathieu Carriere",
      author_email="mathieu.carriere3@gmail.com",
      ext_modules = cythonize([perslocsig]),
      cmdclass = {"build_ext": build_ext})
