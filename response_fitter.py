

from event_type import EventType

class ResponseFitter(object):
    """ResponseFitter takes an input signal and performs deconvolution on it. 
    To do this, it requires event times, and possible covariates.
    ResponseFitter can, for each event type, use different basis function sets."""
    def __init__(self, input_signal, input_sample_frequency, **kwargs):
        super(ResponseFitter, self).__init__()
        self.__dict__.update(kwargs)

        if len(self.input_signal.shape) == 1:
            self.input_signal = self.input_signal[np.newaxis, :]

        self.input_sample_duration = 1.0/self.input_sample_frequency
        self.input_signal_time_points = np.arange(0, 
                                                self.input_signal.shape[1], 
                                                self.input_sample_duration)

        self.X = np.ones((1,self.input_signal.shape[1]))
        self.regressor_lookup_table = {'int':[0]}
        self.event_types = {}

    def create_event_design_matrix(self, event_name, **kwargs):
        ev = EventType(**kwargs)
        ev.create_design_matrix()

        self.X = np.hstack(self.X, ev.X)
        self.regressor_lookup_table.update({
            event_name: np.arange(
                            self.X.shape[0]-ev.X.shape[0], 
                            ev.X.shape[0])
                        })
        self.event_types.update({event_name, ev})

    def regress(self, type):
        """"""
        if type == 'ols':
            self.betas, self.residuals, self.rank, self.s = np.linalg.lstsq(self.X.T, self.input_signal)

        for ev in self.event_types.iteritems():
            self.event_types[ev].betas = \
                self.betas[self.regressor_lookup_table[ev]]

    def ridge_regress(self, cv = 20, alphas = None ):
        """perform k-folds cross-validated ridge regression on the design_matrix. To be used when the design matrix contains very collinear regressors. For cross-validation and ridge fitting, we use sklearn's RidgeCV functionality. Note: intercept is not fit, and data are not prenormalized. 

            :param cv: cross-validated folds, inherits RidgeCV cv argument's functionality.
            :type cv: int, standard = 20
            :param alphas: values of penalization parameter to be traversed by the procedure, inherits RidgeCV cv argument's functionality. Standard value, when parameter is None, is np.logspace(7, 0, 20)
            :type alphas: numpy array, from >0 to 1. 
            :returns: instance variables 'betas' (nr_betas x nr_signals) and 'residuals' (nr_signals x nr_samples) are created.
        """
        if alphas is None:
            alphas = np.logspace(7, 0, 20)
        self.rcv = linear_model.RidgeCV(alphas=alphas, 
                fit_intercept=False, 
                cv=cv) 
        self.rcv.fit(self.design_matrix.T, self.resampled_signal.T)

        self.betas = self.rcv.coef_.T
        self.residuals = self.resampled_signal - self.rcv.predict(self.design_matrix.T)

        self.logger.debug('performed ridge regression on %s design_matrix and %s signal, resulting alpha value is %f' % (str(self.design_matrix.shape), str(self.resampled_signal.shape), self.rcv.alpha_))


    def predict_from_design_matrix(self, Xt):
        """predict_from_design_matrix predicts signals given a design matrix.

            :param design_matrix: design matrix from which to predict a signal.
            :type design_matrix: numpy array, (nr_samples x betas.shape)
            :returns: predicted signal(s) 
            :rtype: numpy array (nr_signals x nr_samples)
        """
        # check if we have already run the regression - which is necessary
        assert hasattr(self, 'betas'), 'no betas found, please run regression before prediction'
        assert Xt.shape[0] == self.betas.shape[0], \
                    """designmatrix needs to have the same number of regressors 
                    as the betas already calculated"""

        prediction = np.dot(self.betas.astype(np.float32).T, Xt.astype(np.float32))

        return prediction


    def rsq(self):
        """"""

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


