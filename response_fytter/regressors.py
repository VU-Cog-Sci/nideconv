#!/usr/bin/env python
# encoding: utf-8
"""
response

"""

import numpy as np
import scipy as sp
from scipy import signal
import pandas as pd
import warnings
from .utils import get_proper_interval

def _create_fir_basis(timepoints, n_regressors):
    """"""
    return np.eye(n_regressors)

def _create_fourier_basis(timepoints, n_regressors):
    """"""
    L_fourier = np.zeros((n_regressors, len(timepoints)))
    L_fourier[0,:] = 1

    for r in range(int(n_regressors/2)):
        x = np.linspace(0, 2.0*np.pi*(r+1), len(timepoints))
    #     sin_regressor 
        L_fourier[1+r,:] = np.sqrt(2) * np.sin(x)
    #     cos_regressor 
        L_fourier[1+r+int(n_regressors/2),:] = np.sqrt(2) * np.cos(x)

    return L_fourier

def _create_legendre_basis(timepoints, n_regressors):
    """"""
    x = np.linspace(-1, 1, len(timepoints), endpoint=True)
    L_legendre = np.polynomial.legendre.legval(x=x, c=np.eye(n_regressors))

    return L_legendre

class Regressor(object):

    def __init__(self,
                 name,
                 fitter):

        self.name = name
        self.fitter= fitter


    def create_design_matrix():
        pass

class Confound(Regressor):

    def __init__(self, name, fitter, confounds):
        
        super(Confound, self).__init__(name, fitter)

        if confounds.ndim == 1:
            self.confounds = confounds[np.newaxis, :]
        else:
            self.confounds = confounds

        self.confounds = pd.DataFrame(self.confounds)

    def create_design_matrix(self):
        self.X = self.confounds
        self.X.columns = pd.MultiIndex.from_product([['confounds'], [self.name], self.X.columns],
                                                    names=['event_type', 'covariate', 'regressor'])
        self.X.set_index(self.fitter.input_signal_time_points, inplace=True)
        self.X.index.rename('t', inplace=True)

class Intercept(Confound):

    def __init__(self,
                 name,
                 fitter):
        confound = pd.DataFrame(np.ones(fitter.input_signal.shape[0]), columns=['intercept'])
        super(Intercept, self).__init__(name, fitter, confound)


class Event(Regressor):
    """Event is a class that encapsulates the creation and conversion
    of design matrices and resulting beta weights for specific event_types. 
    Design matrices for an event_type can be built up of different basis sets,
    and one can choose the time interval over which to fit the response. """
    def __init__(self, 
                name, 
                fitter,
                basis_set='fir', 
                interval=[0,10], 
                n_regressors=0, 
                onset_times=None, 
                durations=None, 
                covariates=None):
        """ Initialize a ResponseFitter object.

        Parameters
        ----------
        fitter : ResponseFitter object
            the response fitter object needed to feed the Event its
            parameters.

        basis_set : string ['fir', 'fourier', 'legendre'] or
                    np.array (1D)
            basis set to use in the fitting. 

        interval : list (2)
            the minimum and maximum timepoints relative to the event onset times
            that delineate the interval for which to estimate the response
            time-course

        n_regressors : int
            for fourier and legendre basis sets, this argument determines the 
            number of regressors to use. More regressors adds more precision, 
            either in terms of added, higher, frequencies (fourier) or 
            higher polynomial order (legendre)

        onset_times : np.array (1D)
            onset times, in seconds, of all the events to estimate the response
            to

        durations : np.array (1D), optional
            durations of each of the events in onset_times. 

        covariates : dict, optional
            dictionary of covariates for each of the events in onset_times. 
            that is, the keys are the names of the covariates, the values
            are 1D numpy arrays of length identical to onset_times; these
            are the covariate values of each of the events in onset_times. 

        """        
        super(Event, self).__init__(name, fitter)

        self.basis_set = basis_set
        self.interval = interval
        self.n_regressors = n_regressors
        self.onset_times = onset_times

        if durations is None:
            self.durations = np.ones_like(self.onset_times) * self.fitter.input_sample_duration
        else:
            self.durations = durations

        self.interval_duration = self.interval[1] - self.interval[0]
        self.sample_duration = self.fitter.input_sample_duration
        self.sample_frequency = self.fitter.input_sample_frequency

        # Check whether the interval is proper
        if ~np.isclose(self.interval_duration % self.sample_duration, 0):
            old_interval = self.interval
            self.interval = get_proper_interval(old_interval, self.sample_duration)
            self.interval_duration = self.interval[1] - self.interval[0]

            warning = '\nWARNING: The duration of the interval %s is not a multiple of ' \
                      'the sample duration %s.\n\r' \
                      'Interval is now automatically set to %s.' \
                       % (old_interval, self.sample_duration, self.interval)

            warnings.warn(warning)

        self.timepoints = np.arange(self.interval[0],
                                    self.interval[1] + self.sample_duration,
                                    self.sample_duration) 
        onset_index = np.arange(self.onset_times.shape[0])

        if covariates is None: # single dict of one-valued covariates
            self.covariates = pd.DataFrame({'intercept': np.ones(self.onset_times.shape[0])})
        else:
            self.covariates = pd.DataFrame(covariates)


        # only for fir, the nr of regressors is dictated by the interval and sample frequency
        if type(basis_set) is str:
            if basis_set == 'fir':
                self.n_regressors = int((self.interval[1] - self.interval[0]) / self.sample_duration) + 1
            # legendre and fourier basis sets should be odd
            elif self.basis_set in ('fourier', 'legendre'):
                if (self.n_regressors %2 ) == 0:
                    self.n_regressors += 1

            if self.basis_set == 'fir':
                self.L = _create_fir_basis(self.timepoints, self.n_regressors)
                self.regressor_labels = ['fir_%d' % i for i in np.arange(self.timepoints.shape[0])]
            elif self.basis_set == 'fourier':
                self.L = _create_fourier_basis(self.timepoints, self.n_regressors)
                self.regressor_labels = ['fourier_intercept']
                self.regressor_labels += ['fourier_sin_%d_period' % period for period in np.arange(1, self.n_regressors//2 + 1)]
                self.regressor_labels += ['fourier_cos_%d_period' % period for period in np.arange(1, self.n_regressors//2 + 1)]

            elif self.basis_set == 'legendre':
                self.L = _create_legendre_basis(self.timepoints, self.n_regressors)
                self.regressor_labels = ['legendre_%d' % poly for poly in np.arange(1, self.n_regressors + 1)]
        else:
            if len(basis_set) != len(self.timepoints):
                raise Exception('Basis set should be exactly %d timepoints long, ' \
                                'current basis has is %d timepoints' % (len(basis_set), len(self.timepoints)))
            if basis_set.ndim == 1:
                basis_set = basis_set[np.newaxis, :]

                
            self.L = basis_set
            self.n_regressors = self.L.shape[0]
            self.regressor_labels = ['custom_basis_set_%d' % i for i in range(1, self.L.shape[0] + 1)]

        # perhaps for covariance matrix fitting, later:
        self.C = self.C_I = np.eye(self.L.shape[0])

    def event_timecourse(self, covariate = None):
        """
        event_timecourse creates a timecourse of events 
        of nr_samples by n_regressors, which has to be converted 
        to the basis of choice.

        Parameters
        ----------
        covariate : string, optional
            Name of the covariate that will be used in the regression. 
            Is set to ones if not providedt.h   

        Returns
        -------
        event_timepoints : np.array (n_regressors, n_timepoints)
            An array that depicts the occurrence of each of the events 
            in the time-space of the signal.

        """

        event_timepoints = np.zeros(self.fitter.input_signal.shape[0])
        mean_dur = self.durations.mean() * self.sample_frequency # check this

        if covariate is None:
            covariate = self.covariates['intercept']
        else:
            covariate = self.covariates[covariate]

        for e,d,c in zip(self.onset_times, self.durations, covariate):
            et = int((e+self.interval[0]) * self.sample_frequency) 
            dt =  int(d*self.sample_frequency)
            event_timepoints[et:et+dt] = c/mean_dur

        return event_timepoints
    
    def create_design_matrix(self):
        """
        create_design_matrix creates the design matrix for this event_type by
        iterating over covariates. 
        
        """

        # create empty design matrix
        self.X = np.zeros((self.fitter.input_signal.shape[0], self.n_regressors * self.covariates.shape[1] ))
        columns = pd.MultiIndex.from_product(([self.name], self.covariates.columns, self.regressor_labels), 
        names=['event_type', 'covariate', 'regressor'])
        self.X = pd.DataFrame(self.X, columns=columns, index=self.fitter.input_signal_time_points)
        self.X.index.rename('t', inplace=True)
        
        for covariate in self.covariates.columns:
            event_timepoints = self.event_timecourse(covariate=covariate)

            for r, regressor in enumerate(self.regressor_labels):
                self.X[self.name, covariate, regressor] = sp.signal.convolve(event_timepoints, self.L[r], 'full')[:self.fitter.input_signal.shape[0]]


    def get_timecourses(self):
        """
        takes betas, given from response_fitter object, and restructures the 
        beta weights to the interval that we're trying to fit, using the L
        basis function matrix. 
        
        """        
        assert hasattr(self, 'betas'), 'no betas found, please run regression before rsq'

        x = self.betas.to_frame().reset_index().rename(columns={0:'beta'})

        timecourses = x.groupby('covariate').beta.apply(lambda beta: pd.Series(beta.dot(self.L), index=self.timepoints)).reset_index()
        
        if timecourses.shape[1] == 2:
            assert(self.covariates.shape[1] == 1)
            timecourses.columns = ['t', 'signal']
            timecourses.insert(0, 'covariate', self.covariates.columns[0])
        else:
            timecourses.columns = ['covariate', 't', 'signal']

        return timecourses

