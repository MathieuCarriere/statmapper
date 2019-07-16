# statmapper

This package provides code for various statistics to be run on the Mapper. Install with:


`git clone https://github.com/MathieuCarriere/statmapper.git && cd statmapper && pip install .`

The functions included in this package allow to compute representative nodes for the different topological features of a Mapper computed with [sklearn_tda](https://github.com/MathieuCarriere/sklearn-tda). A complete example is provided in the notebook `statmapper.ipynb`. This package has the same dependencies as [sklearn_tda](https://github.com/MathieuCarriere/sklearn-tda) and also requires a path to the ouput of the compilation of the [homology localization code](https://github.com/pxiangwu/Persistent-Homology-Localization-Algorithms). We include such a compiled file (for Ubuntu 18.04) in the external_functions` folder.
