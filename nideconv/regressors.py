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
from .utils import (
    get_proper_interval,
    double_gamma_with_d,
    get_time_to_peak_from_timecourse,
    double_gamma_with_d_time_derivative,
)
import logging


def _get_timepoints(
    interval,
    sample_rate,
    oversample,
    endpoint=False):

    total_length = interval[1] - interval[0]
    timepoints = np.linspace(
        interval[0],
        interval[1],
        int(total_length * sample_rate * oversample),
        endpoint=endpoint)
    
    return timepoints


def _create_fir_basis(interval, sample_rate, n_regressors, oversample=1):
    """"""

    regressor_labels = [f'fir_{d}' for d in np.arange(n_regressors)]

    basis = np.eye(n_regressors)
    basis = np.vstack((basis, basis[-1]))

    orig_timepoints = np.linspace(
        interval[0],
        interval[1],
        n_regressors + 1,  # Include endpoint to allow interpolation
        endpoint=True)                  # below

    timepoints = _get_timepoints(interval, sample_rate, oversample)

    fir = interpolate.interp1d(
        orig_timepoints,
        basis,
        kind='nearest',
        axis=0)(timepoints)

    return pd.DataFrame(
        fir,
        index=timepoints,
        columns=regressor_labels).rename_axis('time').rename_axis('basis function', axis=1)

def _create_canonical_hrf_basis(interval, sample_rate, n_regressors, oversample=1):
    timepoints = _get_timepoints(interval, sample_rate, oversample)
    basis_function = double_gamma_with_d(timepoints)[:, np.newaxis]

    return pd.DataFrame(
        basis_function,
        index=timepoints,
        columns=['canonical HRF']).rename_axis('time').rename_axis('basis function', axis=1)


def _create_canonical_hrf_with_time_derivative_basis(interval, sample_rate, n_regressors, oversample=1):
    timepoints = _get_timepoints(interval, sample_rate, oversample)

    hrf = double_gamma_with_d(timepoints)
    dt_hrf = double_gamma_with_d_time_derivative(timepoints)

    return pd.DataFrame(
        np.array([hrf, dt_hrf]).T,
        index=timepoints,
        columns=['HRF', 'HRF (derivative wrt time-to-peak)']).rename_axis('time').rename_axis('basis function', axis=1)

def _create_canonical_hrf_with_time_derivative_dispersion_basis(interval, sample_rate, n_regressors, oversample=1):
    timepoints = _get_timepoints(interval, sample_rate, oversample)

    # hrf = double_gamma_with_d(timepoints)
    # dt_hrf = double_gamma_with_d_time_derivative(timepoints)
    # disp_hrf = double_gamma_with_d_time_derivative_dispersion(timepoints)
    from nilearn.glm.first_level import hemodynamic_models
    tr = 1/sample_rate

    # for some reason, it's iffy when using 0 as starting point
    if interval[0] == 0:
        time_length = timepoints[-1]-timepoints[0]
    else:
        time_length = interval[1]-interval[0]

    hrf = hemodynamic_models.glover_hrf(
        tr, 
        oversampling=oversample, 
        time_length=time_length, 
        onset=abs(interval[0]))

    dt_hrf = hemodynamic_models.glover_time_derivative(
        tr, 
        oversampling=oversample, 
        time_length=time_length,
        onset=abs(interval[0]))    

    disp_hrf = hemodynamic_models.glover_dispersion_derivative(
        tr, 
        oversampling=oversample, 
        time_length=time_length, 
        onset=abs(interval[0]))        

    # ensure same shape
    if hrf.shape[0] != timepoints.shape[0]:
        timepoints = timepoints[:hrf.shape[0]]
        
    return pd.DataFrame(
        np.array([hrf, dt_hrf,disp_hrf]).T,
        index=timepoints,
        columns=['HRF', 'HRF (derivative wrt time-to-peak)','HRF (dispersion wrt time-to-peak)']).rename_axis('time').rename_axis('basis function', axis=1)

def _create_fourier_basis(interval, sample_rate, n_regressors, oversample=1):
    """"""

    timepoints = _get_timepoints(interval, sample_rate, oversample)

    L_fourier = np.zeros((len(timepoints), n_regressors))

    L_fourier[:, 0] = 1

    for r in range(int(n_regressors/2)):
        x = np.linspace(0, 2.0*np.pi*(r+1), len(timepoints))
        #     sin_regressor
        L_fourier[:, 1+r] = np.sqrt(2) * np.sin(x)

        #     cos_regressor
        L_fourier[:, 1+r+int(n_regressors/2)] = np.sqrt(2) * np.cos(x)

    regressor_labels = ['fourier_intercept']
    regressor_labels += [f'fourier_sin_{d}_period' for d in np.arange(1, n_regressors//2 + 1)]
    regressor_labels += [f'fourier_cos_{d}_period' for d in np.arange(1, n_regressors//2 + 1)]

    return pd.DataFrame(
        L_fourier,
        index=timepoints,
        columns=regressor_labels) \
        .rename_axis('time') \
        .rename_axis('basis function', axis=1)


def _create_legendre_basis(interval, sample_rate, n_regressors, oversample=1):
    """"""

    timepoints = _get_timepoints(interval, sample_rate, oversample)
    regressor_labels = [
        f'legendre_{poly}' for poly in np.arange(1, n_regressors + 1)
    ]

    # sync x to timepoints; otherwise mismatch
    x = np.linspace(
        -1,
        1,
        timepoints.shape[0]
    )

    L_legendre = np.polynomial.legendre.legval(
        x=x,
        c=np.eye(n_regressors)
    ).T

    return pd.DataFrame(
        L_legendre,
        index=timepoints,
        columns=regressor_labels) \
        .rename_axis('time') \
        .rename_axis('basis function', axis=1)


def _create_dct_basis(interval, sample_rate, n_regressors, oversample=1):
    """"""

    timepoints = _get_timepoints(interval, sample_rate, oversample)

    L_dct = np.zeros((len(timepoints), n_regressors))
    
    len_tim = len(timepoints)    
    n_times = np.arange(len_tim)
    
    nfct = np.sqrt(2.0 / len_tim)
    
    # L_dct[:, 0] = 1.0
    
    # regressor_labels = ['dct_intercept'] + [f'dct_{n}' for n in range(1, n_regressors)]
    regressor_labels = [f'dct_{n}' for n in range(0, n_regressors)]
    
    for k in range(0, n_regressors):
        L_dct[:, k] = nfct * np.cos((np.pi / len_tim) * (n_times + 0.5) * k)

    return pd.DataFrame(
        L_dct,
        index=timepoints,
        columns=regressor_labels).rename_axis('time').rename_axis('basis function', axis=1)


class Regressor():
    def __init__(self, name, fitter):
        self.name = name
        self.fitter = fitter

    def create_design_matrix(self):
        pass

class Confound(Regressor):

    def __init__(self, name, fitter, confounds):
        super().__init__(name, fitter)
        self.confounds = pd.DataFrame(confounds)

    def create_design_matrix(self, oversample=1):
        self.X = self.confounds.copy()

        self.X.columns = pd.MultiIndex.from_product(
            [['confounds'], [self.name], self.X.columns],
            names=['event type', 'covariate', 'regressor']
        )

        self.X.set_index(self.fitter.input_signal.index, inplace=True)
        self.X.index.rename('time', inplace=True)


class Intercept(Confound):
    def __init__(self, name, fitter):
        confound = pd.DataFrame(
            np.ones(len(fitter.input_signal)),
            columns=['Intercept']
        )
        super().__init__(name, fitter, confound)

class Event(Regressor):
    """Event is a class that encapsulates the creation and conversion
    of design matrices and resulting beta weights for specific event_types.
    Design matrices for an event_type can be built up of different basis sets,
    and one can choose the time interval over which to fit the response. """

    def __init__(
        self,
        name,
        fitter,
        basis_set='fir',
        interval=[0, 10],
        n_regressors=None,
        onsets=None,
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

        onsets : np.array (1D)
            onset times, in seconds, of all the events to estimate the response
            to

        durations : np.array (1D), optional
            durations of each of the events in onsets.

        covariates : dict, optional
            dictionary of covariates for each of the events in onsets.
            that is, the keys are the names of the covariates, the values
            are 1D numpy arrays of length identical to onsets; these
            are the covariate values of each of the events in onsets.

        """
        super().__init__(name, fitter)

        self.basis_set = basis_set
        self.interval = interval
        self.n_regressors = n_regressors
        self.onsets = pd.Series(onsets)

        if self.onsets.dtype != float:
            logging.warning('Onsets should be floats (currently {})! Converting...'.format(
                self.onsets.dtype))
            self.onsets = self.onsets.astype(float)

        self.durations = durations

        if durations is not None:
            self.durations = pd.Series(self.durations)
            if self.durations.dtype != float:
                logging.warning('Durations should be floats (currently {})! Converting...'.format(
                    self.durations.dtype))
                self.durations = self.durations.astype(float)

        self.interval_duration = self.interval[1] - self.interval[0]
        self.sample_duration = self.fitter.sample_duration
        self.sample_rate = self.fitter.sample_rate

        # Check whether the interval is proper
        if ~np.isclose(self.interval_duration % self.sample_duration,
                       0) and \
                ~np.isclose(self.interval_duration % self.sample_duration,
                            self.sample_duration):

            old_interval = self.interval
            self.interval = get_proper_interval(
                old_interval, self.sample_duration)
            self.interval_duration = self.interval[1] - self.interval[0]

            warning = f'\nWARNING: The duration of the interval {old_interval} is not a multiple of ' \
                      f'the sample duration {self.sample_duration}.\n\r' \
                      f'Interval is now automatically set to {self.interval}.'

            warnings.warn(warning)

        if covariates is None:
            self.covariates = pd.DataFrame(
                {'intercept': np.ones(self.onsets.shape[0])}
            )
        else:
            self.covariates = pd.DataFrame(covariates)

        if not isinstance(self.basis_set, str):
            self.n_regressors = self.basis_set.shape[1]
            self.basis_set = pd.DataFrame(
                self.basis_set,
                index=np.linspace(
                    *self.interval,
                    num=len(self.basis_set),
                    endpoint=True
                )
            )

        else:
            self.allowed_basissets = [
                "fir",
                "fourier",
                "dct",
                "legendre",
                "canonical_hrf",
                "canonical_hrf_with_time_derivative",
                "canonical_hrf_with_time_derivative_dispersion",
            ]

            if self.basis_set not in self.allowed_basissets:
                raise ValueError(f"Requested basis set '{self.basis_set}' not available. Must be one of {self.allowed_basissets}")


            if self.basis_set in ['fir', 'dct']:
                length_interval = self.interval[1] - self.interval[0]
                if self.n_regressors is None:
                    self.n_regressors = int(
                        length_interval / self.sample_duration)
                    warnings.warn(f'Number of FIR regressors has automatically been set to {self.n_regressors} per covariate')

                if self.n_regressors > (length_interval / self.sample_duration):
                    warnings.warn(f'Number of regressors ({self.n_regressors}) is larger than the number of timepoints in the interval ({int(length_interval / self.sample_rate)}). This model can only be fit using regularized methods.')

            # legendre and fourier basis sets should be odd
            elif self.basis_set in ('fourier', 'legendre'):
                if self.n_regressors is None:
                    raise Exception('Please provide number of regressors!')
                elif (self.n_regressors % 2) == 0:
                    self.n_regressors += 1
                    warnings.warn(f'Number of {self.basis_set} regressors has to be uneven and has automatically '
                                  f'been set to {self.n_regressors} per covariate')
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

            elif self.basis_set == 'canonical_hrf_with_time_derivative_dispersion':
                if (self.n_regressors is not None) and (self.n_regressors != 3):
                    warnings.warn('With the canonical HRF with time/ and dispersion derivative as a basis set, you can have only THREE regressors per covariate!')

                self.n_regressors = 3             
            else:
                raise f""
    def event_timecourse(
        self,
        covariate=None,
        oversample=1
        ):
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
            durations = np.ones_like(self.onsets) * \
                self.sample_duration / oversample
        else:
            durations = self.durations
        event_timepoints = np.zeros(
            self.fitter.input_signal.shape[0] * oversample)

        if covariate is None:
            covariate = self.covariates['intercept']
        else:
            covariate = self.covariates[covariate]

        for e, d, c in zip(self.onsets, durations, covariate):
            et = int((e + self.interval[0]) * self.sample_rate * oversample)
            dt = np.max((d * self.sample_rate * oversample, 1), 0).astype(int)
            event_timepoints[et:et+dt] = c

        return event_timepoints

    def create_design_matrix(self, oversample=1):
        """
        create_design_matrix creates the design matrix for this event_type by
        iterating over covariates.

        """

        # create empty design matrix
        self.X = np.zeros((self.fitter.input_signal.shape[0] * oversample,
                           self.n_regressors * self.covariates.shape[1]))

        self.L = self.get_basis_function(oversample)

        columns = pd.MultiIndex.from_product(
            ([self.name], self.covariates.columns, self.L.columns),
            names=['event type', 'covariate', 'regressor']
        )

        oversampled_timepoints = np.linspace(
            0,
            self.fitter.input_signal.shape[0] *
            self.sample_duration,
            self.fitter.input_signal.shape[0] *
            oversample,
            endpoint=False)

        self.X = pd.DataFrame(
            self.X,
            columns=columns,
            index=oversampled_timepoints)

        for covariate in self.covariates.columns:
            event_timepoints = self.event_timecourse(
                covariate=covariate,
                oversample=oversample)

            for regressor in self.L.columns:
                self.X[self.name, covariate, regressor] = sp.signal.convolve(event_timepoints,
                                                                             self.L[regressor],
                                                                             'full')[:len(self.X)]
        if oversample != 1:
            self.downsample_design_matrix()

    def get_timecourses(self, oversample=1):
        """
        takes betas, given from response_fitter object, and restructures the
        beta weights to the interval that we're trying to fit, using the L
        basis function matrix.

        """
        assert hasattr(
            self, 'betas'), 'no betas found, please run regression before rsq'

        L = self.get_basis_function(oversample)

        return self.betas.groupby(level=['event type', 'covariate']).apply(_dotproduct_timecourse, L)

    def get_basis_function(self, oversample=1):

        self.allowed_basissets = [
            "fir",
            "fourier",
            "dct",
            "legendre",
            "canonical_hrf",
            "canonical_hrf_with_time_derivative",
            "canonical_hrf_with_time_derivative_dispersion",
        ]

        if self.basis_set not in self.allowed_basissets:
            raise ValueError(f"Requested basis set '{self.basis_set}' not available. Must be one of {self.allowed_basissets}")

        # only for fir, the nr of regressors is dictated by the interval and sample rate
        if isinstance(self.basis_set, str):

            if self.basis_set == 'fir':
                L = _create_fir_basis(
                    self.interval, 
                    self.sample_rate, 
                    self.n_regressors, 
                    oversample
                )
            elif self.basis_set == 'fourier':
                L = _create_fourier_basis(                    
                    self.interval, 
                    self.sample_rate, 
                    self.n_regressors, 
                    oversample
                )
            elif self.basis_set == 'dct':
                L = _create_dct_basis(
                    self.interval, 
                    self.sample_rate, 
                    self.n_regressors, 
                    oversample
                )

            elif self.basis_set == 'legendre':
                L = _create_legendre_basis(
                    self.interval, 
                    self.sample_rate, 
                    self.n_regressors, 
                    oversample
                )                    

            elif self.basis_set == 'canonical_hrf':
                L = _create_canonical_hrf_basis(
                    self.interval,
                    self.sample_rate,
                    1,
                    oversample)
                
                regressor_labels = ['canonical_hrf']

            elif self.basis_set == 'canonical_hrf_with_time_derivative':
                L = _create_canonical_hrf_with_time_derivative_basis(
                    self.interval,
                    self.sample_rate,
                    2,
                    oversample)
                
                regressor_labels = [
                    'canonical_hrf',
                    'canonical_hrf_time_derivative'
                ]
                
            elif self.basis_set == 'canonical_hrf_with_time_derivative_dispersion':
                L = _create_canonical_hrf_with_time_derivative_dispersion_basis(
                    self.interval,
                    self.sample_rate,
                    3,
                    oversample)
                
                regressor_labels = [
                    'canonical_hrf',
                    'canonical_hrf_time_derivative',
                    'canonical_hrf_dispersion_derivative'
                ]

        else:
            L = np.zeros((self.basis_set.shape[0] * oversample, self.n_regressors))

            timepoints = _get_timepoints(self.interval, self.sample_rate, oversample)
            interp = sp.interpolate.interp1d(self.basis_set.index, self.basis_set.values, axis=0)
            L = interp(timepoints)

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
                   .reset_index(level=[-1], drop=True)\
                   .pivot(columns='area', index='peak')


def _dotproduct_timecourse(d, L):
    return L.dot(d.reset_index(level=['event type', 'covariate'], drop=True))
