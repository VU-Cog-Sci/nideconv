from .regressors import Event, Confound, Intercept
import numpy as np
import pandas as pd
from sklearn import linear_model

class ResponseFytter(object):
    """ResponseFytter takes an input signal and performs deconvolution on it. 
    To do this, it requires event times, and possible covariates.
    ResponseFytter can, for each event type, use different basis function sets,
    see Event."""
    def __init__(self, input_signal, input_sample_frequency, add_intercept=True, **kwargs):
        """ Initialize a ResponseFytter object.

        Parameters
        ----------
        input_signal : numpy array, dimensions preferably (X, n)
            input data, of X timeseries of n timepoints 
            sampled at the frequency at which we would 
            like to conduct this analysis

        input_sample_frequency : float
            frequency in Hz at which input data are sampled

        **kwargs : dict
            keyward arguments to be internalized by the ResponseFytter object
        """        
        super(ResponseFytter, self).__init__()
        self.__dict__.update(kwargs)

        self.input_sample_frequency = input_sample_frequency

        self.input_sample_duration = 1.0/self.input_sample_frequency

        self.input_signal_time_points = np.linspace(0, (input_signal.shape[0]-1) *self.input_sample_duration, input_signal.shape[0]) 

        self.input_signal = pd.DataFrame(input_signal, index=self.input_signal_time_points)

        self.X = pd.DataFrame(np.ones((self.input_signal.shape[0], 0)),
                              index=self.input_signal_time_points)
        self.X.index.rename('t', inplace=True)

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


    def _add_regressor(self, regressor):        
        regressor.create_design_matrix()
        if self.X.shape[1] == 0:
            self.X = pd.concat((regressor.X, self.X), 1)
        else:
            self.X = pd.concat((self.X, regressor.X), 1)


    def add_event(self, event_name, **kwargs):
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

        assert event_name not in self.X.columns.get_level_values(0), "The event_name %s is already in use" % event_name

        ev = Event(name=event_name, fitter=self, **kwargs)
        self._add_regressor(ev)

        self.events[event_name] = ev

    def regress(self, type='ols', cv=20, alphas=None):
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
            self.residuals = self.input_signal.values.ravel() - self.predict_from_design_matrix().values.ravel()
            self.residuals = pd.Series(self.residuals, index=self.input_signal_time_points)
        elif type == 'ridge':   # betas and residuals are internalized by ridge_regress
            self.ridge_regress(cv=cv, alphas=alphas)


    def ridge_regress(self, cv=20, alphas=None):
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
        self.residuals = self.input_signal - self.rcv.predict(self.X)

        self._send_betas_to_regressors()

    def _send_betas_to_regressors(self):
        self.betas = pd.Series(self.betas.ravel(), index=self.X.columns)
        self.betas.index.set_names(['event type','covariate', 'regressor'], inplace=True)

        for key in self.events:
            self.events[key].betas = self.betas[key]

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


        prediction = pd.Series(np.dot(self.betas, X.T), index=self.input_signal_time_points)

        return prediction


    def get_timecourses(self):
        assert hasattr(self, 'betas'), 'no betas found, please run regression before prediction'

        timecourses = pd.DataFrame()

        for event_type in self.events:
            tc = self.events[event_type].get_timecourses()
            tc['event type'] = event_type
            timecourses = pd.concat((timecourses, tc), ignore_index=True)

        timecourses.set_index(['event type', 'covariate', 't'], inplace=True)

        return timecourses

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
