from .regressors import (
    Event,
    Confound,
    Intercept
)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from .plotting import plot_timecourses, plot_design_matrix
from .utils import get_time_to_peak_from_timecourse


class ResponseFitter:
    """
    ResponseFitter performs deconvolution on an input signal using event times
    and optionally covariates. Each event type can use different basis function
    sets, configurable via `Event` objects.
    """

    def __init__(
        self,
        input_signal,
        sample_rate,
        oversample_design_matrix=20,
        add_intercept=True,
        **kwargs
        ):
        """
        Initialize a ResponseFitter object.

        Parameters
        ----------
        input_signal : np.ndarray or pd.DataFrame
            Input data of shape (n_timepoints, n_signals).
            This represents the signals to be deconvolved, sampled at the
            frequency specified by `sample_rate`.

        sample_rate : float
            Sampling frequency in Hz of the input data.

        oversample_design_matrix : int, optional
            Factor by which to oversample the design matrix (default = 20).

        add_intercept : bool, optional
            Whether to add an intercept regressor by default (default = True).

        **kwargs : dict
            Additional attributes to be stored in the ResponseFitter object.
        """
        self.input_signal = input_signal
        self.sample_rate = sample_rate
        self.oversample_design_matrix = oversample_design_matrix

        # Store any other passed keyword arguments as object attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.sample_rate = sample_rate
        self.sample_duration = 1.0/self.sample_rate

        self.oversample_design_matrix = oversample_design_matrix

        self.input_signal_time_points = np.linspace(
            0,
            input_signal.shape[0] *
            self.sample_duration,
            input_signal.shape[0],
            endpoint=False
        )

        self.input_signal = pd.DataFrame(input_signal)
        self.input_signal.index = pd.Index(
            self.input_signal_time_points,
            name='time'
        )

        self.X = pd.DataFrame(index=self.input_signal.index)

        if add_intercept:
            self.add_intercept()

        self.events = {}

    def add_intercept(self, name='intercept'):
        intercept = Intercept(name, self)
        self._add_regressor(intercept)

    def add_confounds(self, name, confound):
        """Add a timeseries or set of timeseries to the general
        design matrix as a confound

        Parameters
        ----------
        confound : array
            Confound of (n_timepoints) or (n_timepoints, n_confounds)

        """

        confound = Confound(name, self, confound)
        self._add_regressor(confound)

    def _add_regressor(self, regressor, oversample=None):

        if oversample is None:
            oversample = self.oversample_design_matrix

        regressor.create_design_matrix(oversample=oversample)

        if self.X.shape[1] == 0:
            self.X_list = [regressor.X, self.X]
        else:
            self.X_list = [self.X, regressor.X]

        self.X = pd.concat(self.X_list, axis=1)
        self.X.columns.names = regressor.X.columns.names

    def add_event(
        self,
        event_name,
        onsets=None,
        basis_set='fir',
        interval=[0, 10],
        n_regressors=None,
        durations=None,
        covariates=None,
        **kwargs):

        """
        create design matrix for a given event_type.

        Parameters
        ----------
        event_name : string
            Name of the event_type, used as key to lookup this event_type's
            characteristics

        **kwargs : dict
            keyward arguments to be internalized by the generated and
            internalized Event object. Needs to consist of the
            necessary arguments to create an Event object,
            see Event constructor method.

        """

        assert event_name not in self.X.columns.get_level_values(
            0), f"The event_name {event_name} is already in use"

        ev = Event(
            name=event_name,
            onsets=onsets,
            basis_set=basis_set,
            interval=interval,
            n_regressors=n_regressors,
            durations=durations,
            covariates=covariates,
            fitter=self,
            **kwargs
        )

        self._add_regressor(ev)

        self.events[event_name] = ev

    def fit(self, type='ols', cv=20, alphas=None, store_residuals=False):
        """Regress a created design matrix on the input_data.

        Creates internal variables betas, residuals, rank and s.
        The beta values are then injected into the event_type objects the
        ResponseFitter contains.

        Parameters
        ----------
        type : string, optional
            the type of fit to be done. Options are 'ols' for np.linalg.lstsq,
            'ridge' for CV ridge regression.

        """
        if type == 'ols':
            n, p = self.X.shape

            self.betas, self.sse, self.rank, self.s = \
                np.linalg.lstsq(self.X, self.input_signal, rcond=-1)
            self._send_betas_to_regressors()

            if self.rank < p:
                raise Exception('Design matrix is singular. Consider using less '
                                'regressors, basis functions, or try ridge regression.')

            self.sigma2 = self.sse / (n - (p+1))
            self.sigma2 = pd.Series(
                self.sigma2, index=self.input_signal.columns)

            if store_residuals:
                prediction = self.X.dot(self.betas)
                self._residuals = self.input_signal - prediction

        elif type == 'ridge':   # betas and residuals are internalized by ridge_regress
            if self.input_signal.shape[1] > 1:
                raise NotImplementedError(
                    'No support for multidimensional signals yet')
            self.ridge_regress(cv=cv, alphas=alphas,
                               store_residuals=store_residuals)

    def get_standard_errors_timecourse(self, melt=False, oversample=None):
        self._check_fitted()
        c = self.get_basis_functions(oversample=oversample)
        X_ = np.linalg.pinv(self.X.T.dot(self.X))
        sem = np.sqrt(
            (c.values.dot(X_) * c).sum(1).values[:, np.newaxis] * self.sigma2[np.newaxis, :])
        sem = pd.DataFrame(sem, index=c.index, columns=self.sigma2.index)

        if melt:
            sem = sem.reset_index().melt(id_vars=['event type',
                                                  'covariate',
                                                  'time'],
                                         var_name='roi')

        return sem

    def get_t_value_timecourses(self,
                                oversample=None,
                                melt=False):

        tc = self.get_timecourses(oversample=oversample,
                                  melt=melt)
        sem = self.get_standard_errors_timecourse(melt=melt,
                                                  oversample=oversample)
        t = tc / sem
        t = pd.concat([t], keys=['t'], names=['stat'], axis=1)
        return t

    def t_test(self, event_type1, event_type2, oversample=None):

        """
        Runs a t-test between two time courses, as defined by `condition 1` and `condition 2`
        and returns a t-value that takes into acount the variance and covariance of the
        estimates of condition 1 and 2 via

        SEM = \sqrt{c (X^TX)^{-1}\sigma^2}

        and t = \frac{c'\hat{beta}}{SEM}}

        where c is defined as the weighted sum of regressors describing the time course
        of `event_type1` minus the weighted sum of regressors describing the time
        course of `event_type2`.

        Parameters
        ----------
        event_type1 : str
            Should be a valid event type that occurs in the design

        event_type2 : str
            Should be a valid event type that occurs in the design

        oversample : int
            At what temporal resolution the resulting timecourses should be oversampled

        """
        bf = self.get_basis_functions(oversample=oversample)
        c = bf.loc[event_type1] - bf.loc[event_type2]

        X_ = np.linalg.pinv(self.X.T.dot(self.X))
        sem = np.sqrt(
            (c.values.dot(X_) * c).sum(1).values[:, np.newaxis] * self.sigma2[np.newaxis, :])

        c_dot_beta = c.dot(self.betas)

        sem = pd.DataFrame(sem, index=c.index, columns=self.sigma2.index)

        return c_dot_beta / sem


    def ridge_regress(self, cv=20, alphas=None, store_residuals=False):
        """
        run CV ridge regression instead of ols fit. Uses sklearn's RidgeCV class

        Parameters
        ----------
        cv : int
            number of cross-validation folds

        alphas : np.array
            the alpha/lambda values to try out in the CV ridge regression

        """

        if alphas is None:
            alphas = np.logspace(7, 0, 20)
        self.rcv = linear_model.RidgeCV(alphas=alphas,
                                        fit_intercept=False,
                                        cv=cv)
        self.rcv.fit(self.X, self.input_signal)

        self.betas = self.rcv.coef_.T

        if store_residuals:
            self._residuals = self.input_signal - self.rcv.predict(self.X)
            self.sse = np.sum(self._residuals**2)
        else:
            self.sse = np.sum(
                (self.input_signal - self.rcv.predict(self.X))**2)

        self._send_betas_to_regressors()

    def _send_betas_to_regressors(self):
        self.betas = pd.DataFrame(self.betas,
                                  index=self.X.columns,
                                  columns=self.input_signal.columns)

        for key in self.events:
            self.events[key].betas = self.betas.loc[[key]]

    def predict_from_design_matrix(self,
                                   X=None,
                                   melt=False):
        """
        predict a signal given a design matrix. Requires regression to have
        been run.

        Parameters
        ----------
        X : np.array, (timepoints, n_regressors)
            the design matrix for which to predict data.

        """
        # check if we have already run the regression - which is necessary
        if X is None:
            X = self.X

        assert hasattr(
            self, 'betas'), 'no betas found, please run regression before prediction'
        assert X.shape[1] == self.betas.shape[0], \
            """designmatrix needs to have the same number of regressors
                    as the betas already calculated"""

        prediction = self.X.dot(self.betas)
        if melt:
            prediction = prediction.reset_index()\
                                   .melt(var_name='roi',
                                         value_name='prediction',
                                         id_vars='time')

        else:
            prediction.columns = ['prediction for %s' %
                                  c for c in prediction.columns]

        return prediction

    def get_basis_functions(self, oversample=None):

        if oversample is None:
            oversample = self.oversample_design_matrix

        names = ['event type', 'covariate', 'time']
        row_index = pd.MultiIndex(names=names,
                                  levels=[[], [], []],
                                  codes=[[], [], []])

        bf = pd.DataFrame(columns=self.X.columns, index=row_index)

        for event_type, event in self.events.items():
            for covariate in event.covariates.columns:
                ev = event.get_basis_function(oversample=oversample)
                ev.index = pd.MultiIndex.from_product([[event_type], [covariate], ev.index],
                                                      names=names)
                ev.columns = pd.MultiIndex.from_product([[event_type], [covariate], ev.columns],
                                                        names=['event type', 'covariate', 'regressor'])
                bf = pd.concat((bf, ev), axis=0, )

        bf.fillna(0, inplace=True)

        return bf

    def get_timecourses(self,
                        oversample=None,
                        melt=False):
        assert hasattr(
            self, 'betas'), 'no betas found, please run regression before prediction'

        if oversample is None:
            oversample = self.oversample_design_matrix

        timecourses = pd.DataFrame()

        for event_type in self.events:
            tc = self.events[event_type].get_timecourses(oversample=oversample)
            timecourses = pd.concat((timecourses, tc), ignore_index=False)

        if melt:
            timecourses = timecourses.reset_index().melt(id_vars=['event type',
                                                                  'covariate',
                                                                  'time'],
                                                         var_name='roi')

        return timecourses

    def plot_timecourses(
        self,
        oversample=None,
        legend=True,
        *args,
        **kwargs):

        if oversample is None:
            oversample = 1

        tc = self.get_timecourses(melt=True, oversample=oversample)
        tc['subject'] = 'dummy'

        return plot_timecourses(
            tc,
            oversample=oversample,
            legend=legend,
            *args,
            **kwargs
        )

    def get_rsq(self):
        """
        calculate the rsq of a given fit.
        calls predict_from_design_matrix to predict the signal that has been fit
        """

        assert hasattr(self, 'betas'), \
            'no betas found, please run regression before rsq'

        rsq = 1 - (self.sse /
                   ((self.input_signal.values - self.input_signal.mean().values)**2).sum(0))

        return pd.DataFrame(rsq[np.newaxis, :], columns=self.input_signal.columns)

    def get_residuals(self):
        if not hasattr(self, '_residuals'):
            return self.input_signal - self.predict_from_design_matrix().values
        else:
            return self._residuals

    def get_epochs(self, onsets, interval, remove_incomplete_epochs=True):
        """
        Return a matrix corresponding to specific onsets, within a given
        interval. Matrix size is (n_onsets, n_timepoints_within_interval).

        Note that any events that are in the ResponseFitter-object will
        be regressed out before calculating the epochs.
        """

        # If no other events are defined, no need to regress them out
        if self.X.shape[1] == 0:
            signal = self.input_signal
        else:
            self.fit(store_residuals=True)
            signal = self._residuals

        onsets = np.array(onsets)

        indices = np.array([
            signal.index.get_indexer([onset + interval[0]], method='nearest')[0]
            for onset in onsets
        ])


        interval_duration = interval[1] - interval[0]
        interval_n_samples = int(interval_duration * self.sample_rate) + 1

        indices = np.tile(
            indices[:, np.newaxis], (1, interval_n_samples)) + np.arange(interval_n_samples)

        # Set elements in epochs that fall out of time series to nan
        indices[indices >= signal.shape[0]] = -1

        # Make dummy element to fill epochs with nans if they fall out of the timeseries
        signal = pd.concat((signal, pd.DataFrame(np.zeros((1, signal.shape[1])) * np.nan,
                                                 columns=signal.columns,
                                                 index=[np.nan])), axis=0)

        # Calculate epochs
        epochs = signal.values[indices].swapaxes(-1, -2)
        epochs = epochs.reshape((epochs.shape[0], np.prod(epochs.shape[1:])))
        columns = pd.MultiIndex.from_product([signal.columns,
                                              np.linspace(interval[0], interval[1], interval_n_samples)],
                                             names=['roi', 'time'])
        epochs = pd.DataFrame(epochs,
                              columns=columns,
                              index=pd.Index(onsets, name='onset'))

        # Get rid of incomplete epochs:
        if remove_incomplete_epochs:
            epochs = epochs[~epochs.isnull().any(axis=1)]

        return epochs

    def get_time_to_peak(self,
                         oversample=None,
                         cutoff=1.0,
                         negative_peak=False,
                         include_prominence=False):

        if oversample is None:
            oversample = self.oversample_design_matrix

        if include_prominence:
            ix = ['time peak', 'prominence']
        else:
            ix = ['time peak']

        return self.get_timecourses(oversample=oversample)\
                   .groupby(['event type', 'covariate'])\
                   .apply(get_time_to_peak_from_timecourse,
                          negative_peak=negative_peak,
                          cutoff=cutoff)\
            .loc[(slice(None), slice(None), ix), :]

    def get_original_signal(self, melt=False):
        if melt:
            return self.input_signal.reset_index()\
                                    .melt(var_name='roi',
                                          value_name='signals',
                                          id_vars='time')
        else:
            return self.input_signal

    def plot_model_fit(self,
                       xlim=None,
                       legend=True):

        n_rois = self.input_signal.shape[1]
        if n_rois > 24:
            raise Exception(
                'Are you sure you want to plot {} areas?!'.format(n_rois))

        signal = self.get_original_signal(melt=True)
        prediction = self.predict_from_design_matrix(melt=True)

        data = signal.merge(prediction)

        if n_rois < 4:
            col_wrap = n_rois
        else:
            col_wrap = 4

        fac = sns.FacetGrid(data,
                            col='roi',
                            col_wrap=col_wrap,
                            aspect=3)

        fac.map(plt.plot, 'time', 'signals', color='k', label='signal')
        fac.map(plt.plot, 'time', 'prediction', color='r',
                ls='--', lw=3, label='prediction')

        if xlim is not None:
            for ax in fac.axes.ravel():
                ax.set_xlim(*xlim)

        if legend:
            fac.add_legend()

        fac.set_ylabels('signal')
        fac.set_titles('{col_name}')

        return fac

    def _check_fitted(self):
        assert hasattr(self, 'betas'), \
            'no betas found, please run regression before rsq'

    def plot_design_matrix(self, palette=None):

        return plot_design_matrix(self.X,
                                  palette=palette)


class ConcatenatedResponseFitter(ResponseFitter):

    def __init__(self, response_fitters):

        self.response_fitters = response_fitters

        self.X = pd.concat([rf.X for rf in self.response_fitters]).fillna(0)

        for attr in ['sample_rate', 'oversample_design_matrix']:
            check_properties_response_fitters(self.response_fitters, attr)
            setattr(self, attr, getattr(self.response_fitters.iloc[0], attr))

        self.input_signal = pd.concat(
            [rf.input_signal for rf in self.response_fitters])

        self.events = {}
        for rf in self.response_fitters:
            self.events.update(rf.events)

    def add_intercept(self, *args, **kwargs):
        raise Exception('ConcatenatedResponseFitter does not allow for adding'
                        'intercepts anymore. Do this in the original response '
                        'fitters that get concatenated')

    def add_confounds(self, *args, **kwargs):
        raise Exception('ConcatenatedResponseFitter does not allow for adding '
                        'confounds. Do this in the original response '
                        'fytters that get concatenated')

    def add_event(self, *args, **kwargs):
        raise Exception('ConcatenatedResponseFitter does not allow for adding'
                        'events.')

    def plot_timecourses(self,
                         oversample=None,
                         *args,
                         **kwargs):

        tc = self.get_timecourses(melt=True,
                                  oversample=oversample)
        tc['subject'] = 'dummy'

        plot_timecourses(tc, *args, **kwargs)

    def get_epochs(self, onsets, interval, remove_incomplete_epochs=True):
        raise NotImplementedError()


def check_properties_response_fitters(response_fitters, attribute):

    attribute_values = [getattr(rf, attribute) for rf in response_fitters]

    assert(all([v == attribute_values[0] for v in attribute_values])
           ), "%s not equal across response fitters!" % attribute
