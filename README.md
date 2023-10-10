# statmapper

**Warning**: this code is no longer maintained as it is now part of the Gudhi library. See https://gudhi.inria.fr/python/latest/cover_complex_sklearn_isk_ref.html and https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-cover-complex.ipynb.

This package provides code for various statistics to be run on the Mapper. Install with:


`git clone https://github.com/MathieuCarriere/statmapper.git && cd statmapper && pip install .`

The functions included in this package allow to compute representative nodes for the different topological features of a Mapper computed with [`sklearn_tda`](https://github.com/MathieuCarriere/sklearn-tda), as well as an evaluation of their significance. A complete example is provided in the notebook `example/statmapper.ipynb`. This package has the same dependencies as [`sklearn_tda`](https://github.com/MathieuCarriere/sklearn-tda).
