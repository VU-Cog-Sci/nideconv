from .regressors import Event, Confound, Intercept
import numpy as np
import pandas as pd
from sklearn import linear_model
import scipy as sp
from .plotting import plot_timecourses
from nilearn import input_data, image
from nilearn._utils import load_niimg

class ResponseFytter(object):
    """ResponseFytter takes an input signal and performs deconvolution on it. 
    To do this, it requires event times, and possible covariates.
    ResponseFytter can, for each event_type, use different basis function sets,
    see Event."""
    def __init__(self,
                 input_signal,
                 sample_rate,
                 oversample_design_matrix=20,
                 add_intercept=True, **kwargs):
        """ Initialize a ResponseFytter object.

        Parameters
        ----------
        input_signal : numpy array, dimensions preferably (X, n)
            input data, of X timeseries of n timepoints 
            sampled at the frequency at which we would 
            like to conduct this analysis

        sample_rate : float
            frequency in Hz at which input data are sampled

        **kwargs : dict
            keyward arguments to be internalized by the ResponseFytter object
        """        
        super(ResponseFytter, self).__init__()
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
        """ 
        Add a timeseries or set of timeseries to the general
        design matrix as a confound

        Parameters
        ----------
        confound : array
            Confound of (n_timepoints) or (n_timepoints, n_confounds)

        """

        confound = Confound(name, self, confound)
        self._add_regressor(confound)


    def _add_regressor(self, regressor, oversample=1):
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

        assert event_name not in self.X.columns.get_level_values(0), "The event_name %s is already in use" % event_name

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
        """
        regress a created design matrix on the input_data, creating internal
        variables betas, residuals, rank and s. 
        The beta values are then injected into the event_type objects the
        response_fitter contains. 

        Parameters
        ----------
        type : string, optional
            the type of fit to be done. Options are 'ols' for np.linalg.lstsq,
            'ridge' for CV ridge regression.

        """
        if type == 'ols':
            self.betas, self.ssquares, self.rank, self.s = \
                                np.linalg.lstsq(self.X, self.input_signal, rcond=None)
            self._send_betas_to_regressors()

            if store_residuals:
                self.residuals = self.input_signal - self.predict_from_design_matrix()

        elif type == 'ridge':   # betas and residuals are internalized by ridge_regress
            if self.input_signal.shape[1] > 1:
                raise NotImplementedError('No support for multidimensional signals yet')
            self.ridge_regress(cv=cv, alphas=alphas, store_residuals=store_residuals)


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
            self.residuals = self.input_signal - self.rcv.predict(self.X)

        self._send_betas_to_regressors()

    def _send_betas_to_regressors(self):
        self.betas = pd.DataFrame(self.betas, 
                                  index=self.X.columns,
                                  columns=self.input_signal.columns)

        for key in self.events:
            self.events[key].betas = self.betas.loc[[key]]

    def predict_from_design_matrix(self, X=None):
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

        return prediction


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
                         *args,
                         **kwargs):

        tc = self.get_timecourses(melt=True,
                                  oversample=oversample)
        tc['subj_idx'] = 'dummy'

        plot_timecourses(tc, *args, **kwargs)

    def rsq(self):
        """
        calculate the rsq of a given fit. 
        calls predict_from_design_matrix to predict the signal that has been fit
        """


        assert hasattr(self, 'betas'), \
                        'no betas found, please run regression before rsq'

        # rsq only counts where we actually try to explain data
        predicted_signal = self.predict_from_design_matrix().values


        rsq = 1.0 - np.sum((np.atleast_2d(predicted_signal).T - self.input_signal)**2, axis = 0) / \
                        np.sum(self.input_signal.squeeze()**2, axis = 0)
        return np.squeeze(rsq)

    #def get_timecourses

    def get_epochs(self, onsets, interval, remove_incomplete_epochs=True):
        """ 
        Return a matrix corresponding to specific onsets, within a given
        interval. Matrix size is (n_onsets, n_timepoints_within_interval).

        Note that any events that are in the ResponseFytter-object will
        be regressed out before calculating the epochs.
        """
        
        # If no other events are defined, no need to regress them out
        if self.X.shape[1] == 0:
            signal = self.input_signal
        else:
            self.regress()
            signal = self.residuals
            
            
        onsets = np.array(onsets)
        
        indices = np.array([signal.index.get_loc(onset, method='nearest') for onset in onsets + interval[0]])
        
        interval_duration = interval[1] - interval[0]
        interval_n_samples = int(interval_duration * self.sample_rate) + 1
        
        indices = np.tile(indices[:, np.newaxis], (1, interval_n_samples)) + np.arange(interval_n_samples)
        
        # Set elements in epochs that fall out of time series to nan
        indices[indices >= signal.shape[0]] = -1

        # Make dummy element to fill epochs with nans if they fall out of the timeseries
        signal = pd.concat((signal, pd.DataFrame([np.nan], index=[np.nan])))

        # Calculate epochs
        epochs =  pd.DataFrame(signal.values.ravel()[indices], columns=np.linspace(interval[0], interval[1], interval_n_samples))
        
        # Get rid of incomplete epochs:
        if remove_incomplete_epochs:
            epochs = epochs[~epochs.isnull().any(1)]
        return epochs

class NiftiResponseFytter(ResponseFytter):

    def __init__(self,
                 func_img,
                 sample_rate,
                 oversample_design_matrix=20,
                 add_intercept=True,
                 mask=None,
                 weight_mask=True,
                 average_over_mask=False,
                 threshold=0,
                 detrend=False,
                 standardize=False,
                 confounds_for_extraction=None,
                 memory=None,
                 **kwargs):


        self.average_over_mask = average_over_mask
        self.confounds = confounds_for_extraction

        if isinstance(mask, input_data.NiftiMasker):
            self.masker = mask
        else:

            mask = load_niimg(mask)

            if weight_mask:
                bool_mask = image.math_img('mask > {}'.format(threshold),
                                           mask=mask)
                self.masker = input_data.NiftiMasker(bool_mask,
                                                     detrend=detrend,
                                                     standardize=standardize,
                                                     memory=memory)
                self.mask_weights = self.masker.fit_transform(load_niimg(mask))
                self.mask_weights /= self.mask_weights.sum()
            else:
                self.masker = input_data.NiftiMasker(mask,
                                                     detrend=detrend,
                                                     standardize=standardize,
                                                     memory=memory)

        input_signal = self.masker.fit_transform(func_img) 
        self.n_voxels = input_signal.shape[1]

        if not weight_mask:
            self.mask_weights = np.ones((1,self.n_voxels))
            self.mask_weights /= self.n_voxels


        super(NiftiResponseFytter, self).__init__(input_signal=input_signal,
                                           sample_rate=sample_rate,
                                           oversample_design_matrix=oversample_design_matrix,
                                           add_intercept=add_intercept,
                                           **kwargs)


    
    def ridge_regress(self, *args, **kwargs):
        raise NotImplementedError('Not implemented for NiftiResponseFytter')


    def predict_from_design_matrix(self,
                                   average_over_mask=False,
                                   X=None):
        prediction = super(NiftiResponseFytter, self).predict_from_design_matrix(X)
        return self._inverse_transform(prediction)

    def get_timecourses(self, 
                        oversample=None,
                        average_over_mask=None,
                        **kwargs
                        ):

        if average_over_mask is None:
            average_over_mask = self.average_over_mask

        if len(self.events) is 0:
            raise Exception("No events were added")

        timecourses = super(NiftiResponseFytter, self).get_timecourses(oversample=oversample,
                                                                       melt=False,
                                                                       **kwargs)

        if average_over_mask:
            tc_nii = self._inverse_transform(timecourses, True)
        else:
            tc_df = []
            for (event_type, covariate), tc in timecourses.groupby(level=['event type', 'covariate']):
                tc_nii = self._inverse_transform(tc, False)
                tc = pd.DataFrame([tc_nii], index=pd.MultiIndex.from_tuples([(event_type, covariate)],
                                                                         names=['event type', 'covariate']),
                                  columns=['nii'])
                tc_df.append(tc)

            return pd.concat(tc_df)

                                       


    def _inverse_transform(self, 
                           data,
                           average_over_mask=None):

        if average_over_mask is None:
            average_over_mask = self.average_over_mask

        if average_over_mask:
            data = data.dot(self.mask_weights.T) 
            return data.sum(1)
        else:
            return self.masker.inverse_transform(data)

