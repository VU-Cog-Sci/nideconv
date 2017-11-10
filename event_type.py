#!/usr/bin/env python
# encoding: utf-8
"""
response

"""

import numpy as np
import scipy as sp

def _create_fir_basis(timepoints, nr_regressors):
    """"""
    return np.eye(nr_regressors)

def _create_fourier_basis(timepoints, nr_regressors):
    """"""
    L_fourier = np.zeros((nr_regressors, len(timepoints)))
    L_fourier[0,:] = 1

    for r in range(int(nr_regressors/2)):
        x = np.linspace(0, 2.0*np.pi*(r+1), len(time_points))
    #     sin_regressor 
        L_fourier[1+r,:] = np.sqrt(2) * np.sin(x)
    #     cos_regressor 
        L_fourier[1+r+int(nr_regressors/2),:] = np.sqrt(2) * np.cos(x)

    return L_fourier

def _create_legendre_basis(timepoints, nr_regressors):
    """"""
    x = np.linspace(-1, 1, len(timepoints), endpoint=True)
    L_legendre = np.polynomial.legendre.legval(x=x, c=np.eye(nr_regressors))

    return L_legendre



class EventType(object):
    """docstring for EventType"""
    def __init__(self, 
                fitter, 
                basis_set='fir', 
                interval=[0,10], 
                nr_regressors=0, 
                onset_times=None, 
                durations=None, 
                covariates=None):
        """
        """
        super(EventType, self).__init__()

        self.fitter = fitter
        self.basis_set = basis_set
        self.interval = interval
        self.nr_regressors = nr_regressors
        self.onset_times = onset_times
        self.durations = durations
        self.covariates = covariates

        self.timepoints = np.range(self.interval[0], self.interval[1], 
                                    self.fitter.input_sample_duration)

        if self.covariates == None: # single dict of one-valued covariates
            self.covariates = {'int': np.ones(self.onset_times.shape)}

        # only for fir, the nr of regressors is dictated by the interval and sample frequency
        if basis_set == 'fir':
            self.nr_regressors = int((interval[1] - interval[0]) 
                                    / self.fitter.input_sample_frequency)
        # legendre and fourier basis sets should be odd
        elif self.basis_set in ('fourier', 'legendre'):
            if (self.nr_regressors %2 ) == 0:
                self.nr_regressors += 1

        if self.basis_set == 'fir':
            self.L = _create_fir_basis(self.timepoints, self.nr_regressors)
        elif self.basis_set == 'fourier':
            self.L = _create_fourier_basis(self.timepoints, self.nr_regressors)
        elif self.basis_set == 'legendre':
            self.L = _create_legendre_basis(self.timepoints, self.nr_regressors)

        # create empty design matrix
        self.X = np.zeros((self.nr_regressors * len(self.covariates), self.fitter.input_signal.shape[1]))

        # perhaps for covariance matrix fitting, later:
        self.C = self.C_I = np.eye(self.L.shape[0])

    def event_timecourse(self, covariate = None):
        """
        event_timecourse creates a timecourse of events 
        of nr_samples by nr_regressors, which has to be converted 
        to the basis of choice.
        """
        event_timepoints = np.zeros((self.fitter.input_signal.shape[1], 
                                            len(self.nr_regressors)))
        mean_dur = self.durations.mean() / self.fitter.input_sample_frequency # check this

        if covariate == None:
            covariate = np.ones(self.onset_times.shape)

        for e,d,c in zip(self.onset_times, self.durations, covariate):
            et = int((e+interval[0]) * self.fitter.input_sample_frequency), 
            dt =  int(d*self.fitter.input_sample_frequency)
            event_timepoints[et:et+dt,:] = c/mean_dur

        return event_timepoints
    
    def create_design_matrix(self):
        """"""
        self.covariate_indices = {}
        for i,key in enumerate(self.covariates.iterkeys()):
            event_timepoints = self.event_timecourse(covariate=self.covariates[key])
            for r in range(self.nr_regressors):
                X[i * nr_regressors + r] = \
                sp.signal.fftconvolve(event_timepoints, self.L[r])[:input_data.shape[0]]
            self.covariate_indices.update{key: np.arange(i * nr_regressors,nr_regressors)}

    def betas_to_timecourses(self):
        """"""
        assert hasattr(self, 'betas'), 'no betas found, please run regression before rsq'

        self.covariate_timecourses = {}
        for key in self.covariates.iterkeys():
            cov_betas = self.betas[self.covariate_indices[key]]
            self.covariate_timecourses.update({key:
                                                np.dot(cov_betas, self.L)})

