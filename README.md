# Nideconv
nideconv is a Python module for fast and easy estimating of event-related signals.

# Installation
Currently, nideconv can be installed using the GitHub repository:

### Make Conda environment (optional but highly recomended, especially for Windows)
I highly recommend to first make a dedicated [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)-environment:

`conda create --name nideconv`

Then activate that environment

`conda activate nideconv`

Install requirements

`conda install numpy scikit-learn pystan`

And install nideconv itself

`pip install git+https://github.com/VU-Cog-Sci/nideconv`

> Note: Due to the dependency on `pystan` for Bayesian analyses, which is currently not supported for Windows on Python versions >= 3.8.16, you will be able to install `nideconv` on Windows with Python versions > 3.8.16 but *won't* be able to use the Bayesian analysis functionality on higher Python versions (it will throw an error).

### Install nideconv without Anaconda
Whatever you do, it is very important that you have Cython and numpy before you install `nideconv`, so maybe do

`pip install cython numpy`

And then install nideconv itself

`pip install git+https://github.com/VU-Cog-Sci/nideconv`

# Documentation

The latest documentation can be found on http://nideconv.readthedocs.io/en/latest/
