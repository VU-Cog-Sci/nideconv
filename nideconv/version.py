from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "nideconv: a package to fit response time-courses"
# Long description will go up on the pypi page
long_description = """

Nideconv
========
Nideconv is a package that allows you to perform 
impulse response shape fitting on time series data, 
in order to estimate event-related signals. 


Example use cases are fMRI and pupil size analysis. 
The package performs the linear least squares analysis 
using numpy.linalg as a backend, but can switch 
between different backends, such as statsmodels (which is not yet implemented). 
For very collinear design matrices ridge regression is 
implemented through the sklearn RidgeCV function. 


It is possible to add covariates to the events to estimate 
not just the impulse response function, but also correlation 
timecourses with secondary variables. Furthermore, one can add 
the duration each event should have in the designmatrix, 
for designs in which the durations of the events vary. 


In neuroscience, the inspection of the event-related signals 
such as those estimated by nideconv is essential 
for a thorough understanding of one's data. Researchers may 
overlook essential patterns in their data when blindly 
running GLM analyses without looking at the impulse response shapes.

To get started using these components in your own software, please go to the
repository README_.

.. _README: https://github.com/VuCogSci/nideconv/blob/master/README.md

License
=======
``nideconv`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2017--, Gilles de Hollander & Tomas Knapen, 
Vrije Universiteit & Spinoza Centre for Neuroimaging, Amsterdam.
"""

NAME = "nideconv"
MAINTAINER = "Gilles de Hollander"
MAINTAINER_EMAIL = "gilles.de.hollander@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/VuCogSci/nideconv"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Tomas Knapen"
AUTHOR_EMAIL = "tknapen@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'nideconv': [pjoin('data', '*')]}
REQUIRES = ["numpy","scipy","sklearn"]
