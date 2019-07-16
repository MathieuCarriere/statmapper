from distutils.core import setup

setup(
    name                           = "statmapper",
    author                         = "Mathieu Carriere",
    author_email                   = "mathieu.carriere3@gmail.com",
    packages                       = ["statmapper"],
    description                    = "A set of functions for the statistical analysis of Mapper",
    long_description_content_type  = "text/markdown",
    long_description               = open("README.md", "r").read(),
    url                            = "https://github.com/MathieuCarriere/statmapper/",
    classifiers                    = ("Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License", "Operating System :: OS Independent"),
)
