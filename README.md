# response_fytter
response_fytter is a package that allows you to perform impulse response shape fitting on time series data, in order to estimate event-related signals. 


Example use cases are fMRI and pupil size analysis. The package performs the linear least squares analysis using numpy.linalg as a backend, but can switch between different backends, such as statsmodels (which is not yet implemented). For very collinear design matrices ridge regression is implemented through the sklearn RidgeCV function. 


It is possible to add covariates to the events to estimate not just the impulse response function, but also correlation timecourses with secondary variables. Furthermore, one can add the duration each event should have in the designmatrix, for designs in which the durations of the events vary. 


In neuroscience, the inspection of the event-related signals such as those estimated by response_fytter is essential for a thorough understanding of one's data. Researchers may overlook essential patterns in their data when blindly running GLM analyses without looking at the impulse response shapes. 


The test notebook explains how the package can be used for data analysis, by creating toy signals and then using response_fytter to fit the impulse response functions from the toy data. 


## Dependencies
numpy, scipy, statsmodels, sklearn

TODO
- temporal autocorrelation correction





[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.46216.svg)](http://dx.doi.org/10.5281/zenodo.46216)


