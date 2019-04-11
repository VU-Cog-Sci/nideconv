from .regressors import Event, Confound, Intercept
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy as sp
from .plotting import plot_timecourses
from nilearn import input_data, image
from nilearn._utils import load_niimg
from .utils import get_time_to_peak_from_timecourse

class ResponseFitter(object):
    """ResponseFitter takes an input signal and performs deconvolution on it. 
    To do this, it requires event times, and possible covariates.
    ResponseFitter can, for each event type, use different basis function sets,
    see Event."""
    def __init__(self,
                 input_signal,
                 sample_rate,
                 oversample_design_matrix=20,
                 add_intercept=True, **kwargs):
        """ Initialize a ResponseFitter object.

        Parameters
        ----------
        input_signal : numpy array, dimensions preferably (X, n)
            input data, of X timeseries of n timepoints 
            sampled at the frequency at which we would 
            like to conduct this analysis

        sample_rate : float
            frequency in Hz at which input data are sampled

        **kwargs : dict
            keyward arguments to be internalized by the ResponseFitter object
        """        
        super(ResponseFitter, self).__init__()
        self.__dict__.update(kwargs)

        self.sample_rate = sample_rate
        self.sample_duration = 1.0/self.sample_rate

        self.oversample_design_matrix = oversample_design_matrix

        self.input_signal_time_points = np.linspace(0, 
                                                    input_signal.shape[0] * self.sample_duration, 
                                                    input_signal.shape[0],
                                                    endpoint=False) 

        self.input_signal = pd.DataFrame(input_signal)
        self.input_signal.index = pd.Index(self.input_signal_time_points,
                                           name='time')


        self.X = pd.DataFrame(index=self.input_signal.index)

        if add_intercept:
            self.add_intercept()

        self.events =  {}


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
            self.X = pd.concat((regressor.X, self.X), 1)
        else:
            self.X = pd.concat((self.X, regressor.X), 1)


    def add_event(self,
                  event_name,
                  onset_times=None, 
                  basis_set='fir', 
                  interval=[0,10], 
                  n_regressors=None, 
                  durations=None, 
                  covariates=None,
                  **kwargs):
        """
        create design matrix for a given event type.

        Parameters
        ----------
        event_name : string
            Name of the event type, used as key to lookup this event type's
            characteristics

        **kwargs : dict
            keyward arguments to be internalized by the generated and 
            internalized Event object. Needs to consist of the 
            necessary arguments to create an Event object, 
            see Event constructor method.

        """

        assert event_name not in self.X.columns.get_level_values(0), "The event_name {} is already in use".format(str(event_name))

        ev = Event(name=event_name, 
                   onset_times=onset_times,
                   basis_set=basis_set,
                   interval=interval,
                   n_regressors=n_regressors,
                   durations=durations,
                   covariates=covariates,
                   fitter=self,
                   **kwargs)

        self._add_regressor(ev)

        self.events[event_name] = ev



    def regress(self, type='ols', cv=20, alphas=None, store_residuals=False):
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

            self.betas, self.ssquares, self.rank, self.s = \
                                np.linalg.lstsq(self.X, self.input_signal, rcond=-1)
            self._send_betas_to_regressors()

            if self.rank < p:
                raise Exception('Design matrix is singular. Consider using less '
                                'regressors, basis functions, or try ridge regression.')


            self.sigma2 = self.ssquares / (n -(p+1))
            self.sigma2 = pd.Series(self.sigma2, index=self.input_signal.columns)

            if store_residuals:
                prediction = self.X.dot(self.betas)
                self._residuals = self.input_signal - prediction

        elif type == 'ridge':   # betas and residuals are internalized by ridge_regress
            if self.input_signal.shape[1] > 1:
                raise NotImplementedError('No support for multidimensional signals yet')
            self.ridge_regress(cv=cv, alphas=alphas, store_residuals=store_residuals)


    def get_standard_errors_timecourse(self, melt=False, oversample=None):
        self._check_fitted()
        c = self.get_basis_functions(oversample=oversample)
        X_= np.linalg.pinv(self.X.T.dot(self.X))
        sem = np.sqrt((c.values.dot(X_) * c).sum(1).values[:, np.newaxis] * self.sigma2[np.newaxis, :])
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
            self.ssquares = np.sum(self._residuals**2)
        else:
            self.ssquares = np.sum((self.input_signal - self.rcv.predict(self.X))**2)

        self._send_betas_to_regressors()

    def _send_betas_to_regressors(self):
        self.betas = pd.DataFrame(self.betas, 
                                  index=self.X.columns,
                                  columns=self.input_signal.columns)
        #self.betas.index.set_names(['event_type','covariate', 'regressor'], inplace=True)

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

        assert hasattr(self, 'betas'), 'no betas found, please run regression before prediction'
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
            prediction.columns = ['prediction for {}'.format(str(c)) for c in prediction.columns]

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
                ev.index = pd.MultiIndex.from_product([[event_type], [covariate], ev.index.get_values()],
                                                       names=names)
                ev.columns = pd.MultiIndex.from_product([[event_type], [covariate], ev.columns.get_values()],
                                                       names=['event type', 'covariate', 'regressor'])
                bf = pd.concat((bf, ev), axis=0, )
                
        bf.fillna(0, inplace=True)
            
        return bf

    def get_timecourses(self, 
                        oversample=None,
                        melt=False):
        assert hasattr(self, 'betas'), 'no betas found, please run regression before prediction'

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

    def plot_timecourses(self,
                         oversample=None,
                         legend=True,
                         *args,
                         **kwargs):

        tc = self.get_timecourses(melt=True,
                                  oversample=oversample)
        tc['subject'] = 'dummy'

        return plot_timecourses(tc, *args, **kwargs)
        

    def get_rsq(self):
        """
        calculate the rsq of a given fit. 
        calls predict_from_design_matrix to predict the signal that has been fit
        """


        assert hasattr(self, 'betas'), \
                        'no betas found, please run regression before rsq'

        return 1 - (self.ssquares / ((self.input_signal - self.input_signal.mean())**2).sum())


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
            self.regress(store_residuals=True)
            signal = self._residuals
            
            
        onsets = np.array(onsets)
        
        indices = np.array([signal.index.get_loc(onset, method='nearest') for onset in onsets + interval[0]])
        
        interval_duration = interval[1] - interval[0]
        interval_n_samples = int(interval_duration * self.sample_rate) + 1
        
        indices = np.tile(indices[:, np.newaxis], (1, interval_n_samples)) + np.arange(interval_n_samples)
        
        # Set elements in epochs that fall out of time series to nan
        indices[indices >= signal.shape[0]] = -1

        # Make dummy element to fill epochs with nans if they fall out of the timeseries
        signal = pd.concat((signal, pd.DataFrame(np.zeros((1, signal.shape[1])) * np.nan,
                                                 columns=signal.columns,
                                                 index=[np.nan])), 
                           0)

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
            epochs = epochs[~epochs.isnull().any(1)]
        return epochs


    def get_time_to_peak(self, 
                         oversample=None, 
                         cutoff=1.0, 
                         negative_peak=False,
                         include_prominence=False):
        
        if oversample is None:
            oversample = self.oversample_design_matrix

        if include_prominence:
            cols = ['time to peak', 'prominence']
        else:
            cols = ['time to peak']


        return self.get_timecourses(oversample=oversample)\
                   .groupby(['event type', 'covariate'], as_index=False)\
                   .apply(get_time_to_peak_from_timecourse, 
                          negative_peak=negative_peak,
                          cutoff=cutoff)\
                   .reset_index(level=[ -1], drop=True)\
                   .pivot_table(columns='area', index='peak')[cols]
                   
    
    def get_original_signal(self, melt=False):
        if melt:
            return self.input_signal.reset_index()\
                                    .melt(var_name='roi',
                                          value_name='signal',
                                          id_vars='time')
        else:
            return self.input_signal

    def plot_model_fit(self,
                       xlim=None,
                       legend=True):
        

        n_rois = self.input_signal.shape[1]
        if n_rois > 24:
            raise Exception('Are you sure you want to plot {} areas?!'.format(n_rois))

        signal = self.get_original_signal(melt=True)
        prediction = self.predict_from_design_matrix(melt=True)

        data = signal.merge(prediction)
        
        if n_rois < 4:
            col_wrap = n_rois
        else:
            col_wrap = 4
       
        fac  = sns.FacetGrid(data, 
                             col='roi',
                             col_wrap=col_wrap,
                             aspect=3)

        fac.map(plt.plot, 'time', 'signal', color='k', label='signal')
        fac.map(plt.plot, 'time', 'prediction', color='r', lw=3, label='prediction')

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

class ConcatenatedResponseFitter(ResponseFitter):


    def __init__(self, response_fitters):

        self.response_fitters = response_fitters

        self.X = pd.concat([rf.X for rf in self.response_fitters]).fillna(0)

        for attr in ['sample_rate', 'oversample_design_matrix']:
            check_properties_response_fitters(self.response_fitters, attr)
            setattr(self, attr, getattr(self.response_fitters[0], attr))


        self.input_signal = pd.concat([rf.input_signal for rf in self.response_fitters])

        self.events =  {}
        for rf in self.response_fitters:
            self.events.update(rf.events)


    def add_intercept(self, *args, **kwargs):
        raise Exception('ConcatenatedResponseFitter does not allow for adding'\
                         'intercepts anymore. Do this in the original response '\
                         'fitters that get concatenated')


    def add_confounds(self, *args, **kwargs):
        raise Exception('ConcatenatedResponseFitter does not allow for adding '\
                         'confounds. Do this in the original response '\
                         'fytters that get concatenated')


    def add_event(self, *args, **kwargs):
        raise Exception('ConcatenatedResponseFitter does not allow for adding'\
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

    assert(all([v == attribute_values[0] for v in attribute_values])), "{} not equal across response fitters!".format(str(attribute))
