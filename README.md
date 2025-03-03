# Nideconv
Nideconv is a package that allows you to perform impulse response shape fitting on time series data, in order to estimate event-related signals.

Example use cases are fMRI and pupil size analysis.
The package performs the linear least squares analysis using `numpy.linalg` as a backend, but can switch between different backends, such as `statsmodels` (which is not yet implemented).
For very collinear design matrices ridge regression is implemented through the sklearn RidgeCV function.

It is possible to add covariates to the events to estimate not just the impulse response function, but also correlation timecourses with secondary variables. Furthermore, one can add the duration each event should have in the designmatrix, for designs in which the durations of the events vary.

In neuroscience, the inspection of the event-related signals such as those estimated by nideconv is essential for a thorough understanding of one's data.
Researchers may overlook essential patterns in their data when blindly running GLM analyses without looking at the impulse response shapes.

# Installation
Currently, nideconv can be installed using the GitHub repository:

### Make Conda environment (optional but highly recomended, especially for Windows)
I highly recommend to first make a dedicated [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)-environment:

`conda create --name nideconv`

Then activate that environment

`conda activate nideconv`

Install nideconv

`pip install git+https://github.com/VU-Cog-Sci/nideconv`

> Note: Due to the dependency on `pystan` for Bayesian analyses, which is currently not supported for Windows on Python versions >= 3.8.16, you will be able to install `nideconv` on Windows with Python versions > 3.8.16 but *won't* be able to use the Bayesian analysis functionality on higher Python versions (it will throw an error).

# Documentation

The latest documentation can be found on http://nideconv.readthedocs.io/en/latest/
