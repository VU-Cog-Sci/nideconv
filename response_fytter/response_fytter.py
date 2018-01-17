from .event_type import EventType
import numpy as np
import pandas as pd

class ResponseFytter(object):
    """ResponseFytter takes an input signal and performs deconvolution on it. 
    To do this, it requires event times, and possible covariates.
    ResponseFytter can, for each event type, use different basis function sets,
    see EventType."""
    def __init__(self, input_signal, input_sample_frequency, **kwargs):
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

        self.input_signal_time_points = np.arange(0, 
                                                input_signal.shape[0] * self.input_sample_duration, 
                                                self.input_sample_duration)

        self.input_signal = pd.DataFrame(input_signal, index=self.input_signal_time_points)

        self.X = pd.DataFrame({('general', 'intercept', 0):np.ones((self.input_signal.shape[0]))},
                              index=self.input_signal_time_points)
        self.X.index.rename('t', inplace=True)


    def create_event_design_matrix(self, event_name, **kwargs):
        """
        create design matrix for a given event type.

        Parameters
        ----------
        event_name : string
            Name of the event type, used as key to lookup this event type's
            characteristics

        **kwargs : dict
            keyward arguments to be internalized by the generated and 
            internalized EventType object. Needs to consist of the 
            necessary arguments to create an EventType object, 
            see EventType constructor method.

        """
        ev = EventType(fitter=self, name=event_name, **kwargs)
        ev.create_design_matrix()

        self.X = pd.concat((self.X, ev.X), 1)

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
            self.betas, self.residuals, self.rank, self.s = \
                                np.linalg.lstsq(self.X, self.input_signal)
        elif type == 'ridge':   # betas and residuals are internalized by ridge_regress
            self.ridge_regress(cv=cv, alphas=alphas)

        self.betas = pd.Series(self.betas.ravel(), index=self.X.columns)

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
        self.rcv.fit(self.design_matrix.T, self.resampled_signal.T)

        self.betas = self.rcv.coef_.T
        self.residuals = self.resampled_signal - self.rcv.predict(self.design_matrix.T)

    def predict_from_design_matrix(self, Xt):
        """
        predict a signal given a design matrix. Requires regression to have
        been run.

        Parameters
        ----------
        Xt : np.array, (nr_regressors, timepoints)
            the design matrix for which to predict data.

        """
        # check if we have already run the regression - which is necessary
        assert hasattr(self, 'betas'), 'no betas found, please run regression before prediction'
        assert Xt.shape[0] == self.betas.shape[0], \
                    """designmatrix needs to have the same number of regressors 
                    as the betas already calculated"""

        prediction = np.dot(self.betas.T, Xt)

        return prediction


    def rsq(self):
        """
        calculate the rsq of a given fit. 
        calls predict_from_design_matrix to predict the signal that has been fit
        """


        assert hasattr(self, 'betas'), \
                        'no betas found, please run regression before rsq'

        # rsq only counts where we actually try to explain data
        explained_signal_timepoints = self.X.sum(axis = 0) != 0        
        predicted_signal = self.predict_from_design_matrix(self.X).T

        valid_prediction = explained_signal[:,explained_signal_timepoints]
        valid_signal = self.input_signal[:,explained_signal_timepoints]

        self.rsq = 1.0 - np.sum((valid_prediction - valid_signal)**2, axis = -1) / \
                                np.sum(valid_signal.squeeze()**2, axis = -1)
        return np.squeeze(self.rsq)

    #def get_timecourses
