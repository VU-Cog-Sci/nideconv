#!/usr/bin/env python
# encoding: utf-8
"""
response

"""

import numpy as np
import scipy as sp
from scipy import signal, interpolate
import pandas as pd
import warnings
from .utils import (get_proper_interval,
                    double_gamma_with_d,
                    get_time_to_peak_from_timecourse,
                    double_gamma_with_d_time_derivative)

def _get_timepoints(interval, 
                    sample_rate,
                    oversample):

    total_length = interval[1] - interval[0]
    timepoints = np.linspace(interval[0],
                              interval[1],
                              total_length * sample_rate * oversample,
                              endpoint=False)
    return timepoints

def _create_fir_basis(interval, sample_rate, n_regressors, oversample=1):
    """"""

    regressor_labels = ['fir_%d' % i for i in np.arange(n_regressors)]

    total_length = interval[1] - interval[0]
    basis = np.eye(n_regressors)
    basis = np.vstack((basis, basis[-1]))

    orig_timepoints = np.linspace(interval[0],
                                  interval[1],
                                  n_regressors + 1, # Include endpoint to allow interpolation 
                                  endpoint=True)                  # below

    timepoints = _get_timepoints(interval, sample_rate, oversample)

    fir = interpolate.interp1d(orig_timepoints,
                               basis,
                               kind='nearest',
                               axis =0)(timepoints)

    return pd.DataFrame(fir,
                        index=timepoints,
                        columns=regressor_labels) \
                            .rename_axis('time') \
                            .rename_axis('basis function', axis=1)

def _create_canonical_hrf_basis(interval, sample_rate, n_regressors, oversample=1):
    timepoints = _get_timepoints(interval, sample_rate, oversample)
    basis_function =  double_gamma_with_d(timepoints)[:, np.newaxis]

    return pd.DataFrame(basis_function,
                        index=timepoints,
                        columns=['canonical HRF']) \
                        .rename_axis('time') \
                        .rename_axis('basis function', axis=1) 

def _create_canonical_hrf_with_time_derivative_basis(interval, sample_rate, n_regressors, oversample=1):
    timepoints = _get_timepoints(interval, sample_rate, oversample)
    
    hrf = double_gamma_with_d(timepoints)
    dt_hrf = double_gamma_with_d_time_derivative(timepoints)

    return pd.DataFrame(np.array([hrf, dt_hrf]).T,
                        index=timepoints,
                        columns=['HRF', 'HRF (derivative wrt time-to-peak)']) \
                        .rename_axis('time') \
                        .rename_axis('basis function', axis=1) 

def _create_fourier_basis(interval, sample_rate, n_regressors, oversample=1):
    """"""
    
    timepoints = _get_timepoints(interval, sample_rate, oversample)
    
    L_fourier = np.zeros((len(timepoints), n_regressors))
    
    L_fourier[:,0] = 1

    for r in range(int(n_regressors/2)):
        x = np.linspace(0, 2.0*np.pi*(r+1), len(timepoints))
        #     sin_regressor 
        L_fourier[:, 1+r] = np.sqrt(2) * np.sin(x)

        #     cos_regressor 
        L_fourier[:, 1+r+int(n_regressors/2)] = np.sqrt(2) * np.cos(x)

    regressor_labels = ['fourier_intercept']
    regressor_labels += ['fourier_sin_%d_period' % period for period in np.arange(1, n_regressors//2 + 1)]
    regressor_labels += ['fourier_cos_%d_period' % period for period in np.arange(1, n_regressors//2 + 1)]

    return pd.DataFrame(L_fourier,
                        index=timepoints,
                        columns=regressor_labels) \
                        .rename_axis('time') \
                        .rename_axis('basis function', axis=1) 

def _create_legendre_basis(interval, sample_rate, n_regressors, oversample=1):
    """"""

    regressor_labels = ['legendre_%d' % poly for poly in np.arange(1, self.n_regressors + 1)]
    x = np.linspace(-1, 1, int(np.diff(interval)) * oversample + 1, endpoint=True)
    L_legendre = np.polynomial.legendre.legval(x=x, c=np.eye(n_regressors)).T

    return pd.DataFrame(L_legendre,
                        index=timepoints,
                        columns=regressor_labels) \
                        .rename_axis('time') \
                        .rename_axis('basis function', axis=1) 


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
        self.confounds = pd.DataFrame(confounds)

    def create_design_matrix(self, oversample=1):
        self.X = self.confounds
        self.X.columns = pd.MultiIndex.from_product([['confounds'], [self.name], self.X.columns],
                                                    names=['event type', 'covariate', 'regressor'])
        self.X.set_index(self.fitter.input_signal.index, inplace=True)
        self.X.index.rename('time', inplace=True)

class Intercept(Confound):

    def __init__(self,
                 name,
                 fitter):

        confound = pd.DataFrame(np.ones(len(fitter.input_signal)),
                                columns=['intercept'])
        super(Intercept, self).__init__(name, fitter, confound)


class Event(Regressor):
    """Event is a class that encapsulates the creation and conversion
    of design matrices and resulting beta weights for specific event types. 
    Design matrices for an event type can be built up of different basis sets,
    and one can choose the time interval over which to fit the response. """
    def __init__(self, 
                name, 
                fitter,
                basis_set='fir', 
                interval=[0,10], 
                n_regressors=None, 
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
        self.durations = durations

        self.interval_duration = self.interval[1] - self.interval[0]
        self.sample_duration = self.fitter.sample_duration
        self.sample_rate = self.fitter.sample_rate

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


        if covariates is None:
            self.covariates = pd.DataFrame({'intercept': np.ones(self.onset_times.shape[0])})
        else:
            self.covariates = pd.DataFrame(covariates)

        if type(self.basis_set) is not str:
            self.n_regressors = self.basis_set.shape[1]

            self.basis_set = pd.DataFrame(self.basis_set,
                                          index=np.linspace(*self.interval,
                                                            num=len(self.basis_set),
                                                            endpoint=True))

        else:
            if self.basis_set == 'fir':
                if self.n_regressors is None:
                    self.n_regressors = int((self.interval[1] - self.interval[0]) / self.sample_duration)
                    warnings.warn('Number of FIR regressors has automatically been set to %d '
                                  'per covariate' % self.n_regressors)

            # legendre and fourier basis sets should be odd
            elif self.basis_set in ('fourier', 'legendre'):
                if self.n_regressors is None:
                    raise Exception('Please provide number of regressors!')
                elif (self.n_regressors % 2) == 0:
                    self.n_regressors += 1
                    warnings.warn('Number of {} regressors has to be uneven and has automatically ' 
                                  'been set to {} per covariate'.format(self.basis_set, self.n_regressors))
            elif self.basis_set == 'canonical_hrf':
                if (self.n_regressors is not None) and (self.n_regressors != 1):
                    warnings.warn('With the canonical HRF as a basis set, you can have only ONE '
                                  'regressors per covariate!')

                self.n_regressors = 1

            elif self.basis_set == 'canonical_hrf_with_time_derivative':
                if (self.n_regressors is not None) and (self.n_regressors != 2):
                    warnings.warn('With the canonical HRF with time derivative as a basis set,'
                                   'you can have only TWO '
                                  'regressors per covariate!')

                self.n_regressors = 2




    def event_timecourse(self, 
                         covariate=None,
                         oversample=1):
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

        if self.durations is None:
            durations = np.ones_like(self.onset_times) * self.sample_duration / oversample
        else:
            durations = self.durations
        event_timepoints = np.zeros(self.fitter.input_signal.shape[0] * oversample)

        if covariate is None:
            covariate = self.covariates['intercept']
        else:
            covariate = self.covariates[covariate]

        for e,d,c in zip(self.onset_times, durations, covariate):
            et = int((e + self.interval[0]) * self.sample_rate * oversample)
            dt =  np.max((d * self.sample_rate * oversample, 1), 0).astype(int)
            event_timepoints[et:et+dt] = c

        return event_timepoints
    
    def create_design_matrix(self, oversample=1):
        """
        create_design_matrix creates the design matrix for this event type by
        iterating over covariates. 
        
        """

        # create empty design matrix
        self.X = np.zeros((self.fitter.input_signal.shape[0] * oversample, 
                           self.n_regressors * self.covariates.shape[1]))

        L = self.get_basis_function(oversample)

        columns = pd.MultiIndex.from_product(([self.name], self.covariates.columns, L.columns),
                                             names=['event_type', 'covariate', 'regressor'])

        oversampled_timepoints = np.linspace(0, 
                                             self.fitter.input_signal.shape[0] * self.sample_duration, 
                                             self.fitter.input_signal.shape[0] * oversample,
                                             endpoint=False) 

        self.X = pd.DataFrame(self.X,
                              columns=columns,
                              index=oversampled_timepoints)
        

        for covariate in self.covariates.columns:
            event_timepoints = self.event_timecourse(covariate=covariate,
                                                     oversample=oversample)

            for regressor in L.columns:
                self.X[self.name, covariate, regressor] = sp.signal.convolve(event_timepoints,
                                                                             L[regressor],
                                                                             'full')[:len(self.X)]
        if oversample != 1:
            self.downsample_design_matrix()


    def get_timecourses(self, oversample=1):
        """
        takes betas, given from response_fitter object, and restructures the 
        beta weights to the interval that we're trying to fit, using the L
        basis function matrix. 
        
        """        
        assert hasattr(self, 'betas'), 'no betas found, please run regression before rsq'

        L = self.get_basis_function(oversample)

        return self.betas.groupby(level=['event type', 'covariate']).apply(_dotproduct_timecourse, L)

    def get_basis_function(self, oversample=1):


        # only for fir, the nr of regressors is dictated by the interval and sample rate
        if type(self.basis_set) is str:

            if self.basis_set == 'fir':
                L = _create_fir_basis(self.interval, self.sample_rate, self.n_regressors, oversample)

            elif self.basis_set == 'fourier':
                L = _create_fourier_basis(self.interval, self.sample_rate, self.n_regressors, oversample)

            elif self.basis_set == 'legendre':
                L = _create_legendre_basis(self.interval, self.sample_rate, self._regressors, oversample)

            elif self.basis_set == 'canonical_hrf':
                L = _create_canonical_hrf_basis(self.interval,
                                                self.sample_rate,
                                                1,
                                                oversample)
                regressor_labels = ['canonical_hrf']

            elif self.basis_set == 'canonical_hrf_with_time_derivative':
                L = _create_canonical_hrf_with_time_derivative_basis(self.interval,
                                                self.sample_rate,
                                                2,
                                                oversample)
                regressor_labels = ['canonical_hrf', 'canonical_hrf_time_derivative']

        else:
            regressor_labels = ['custom_basis_function_%d' % i for i in range(1, self.n_regressors+1)]        
            L = np.zeros((self.basis_set.shape[0] * oversample, self.n_regressors))

            interp = sp.interpolate.interp1d(self.basis_set.index, self.basis_set.values, axis=0)
            L = interp(timepoints)


        #L = pd.DataFrame(L,
                         #columns=pd.Index(regressor_labels, name='basis_function'),)
                         #index=L.index)

        return L
        
    def downsample_design_matrix(self):
        interp = sp.interpolate.interp1d(self.X.index, self.X.values, axis=0)
        X_ = interp(self.fitter.input_signal.index)
        self.X = pd.DataFrame(X_,
                              columns=self.X.columns,
                              index=self.fitter.input_signal.index)


    def get_time_to_peak(self, oversample=20, cutoff=1.0, negative_peak=False):
        return self.get_timecourses(oversample=oversample)\
                   .groupby(['event type', 'covariate'], as_index=False)\
                   .apply(get_time_to_peak_from_timecourse, 
                          negative_peak=negative_peak,
                          cutoff=cutoff)\
                   .reset_index(level=[ -1], drop=True)\
                   .pivot(columns='area', index='peak')
                   


def _dotproduct_timecourse(d, L):
    return L.dot(d.reset_index(level=['event type', 'covariate'], drop=True))
