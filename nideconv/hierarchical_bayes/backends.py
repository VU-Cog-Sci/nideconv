try:
    import stan
except ImportError:
    pystan = None

import os
import pickle as pkl
import numpy as np
import pandas as pd
from .utils import do_ols

__dir__ = os.path.abspath(os.path.dirname(__file__))


class HierarchicalModel(object):

    def __init__(self, X, subject_ids, subjectwise_errors=False, cauchy_priors=False):

        self.X = pd.DataFrame(X)
        self.subject_ids = np.array(subject_ids).squeeze()
        self.subjectwise_errors = subjectwise_errors
        self.cauchy_priors = cauchy_priors

        if(self.subject_ids.shape[0] != self.X.shape[0]):
            raise Exception("Number of subjects indices should"
                            "correspond to number of rows in the"
                            "design matrices.")
        self._get_subj_idx()

    def sample(self, signal, chains, *args, **kwargs):
        measure = signal.squeeze()
        if(len(measure) != self.X.shape[0]):
            raise Exception("Signal should have same number of elements"
                            "as rows in the design matrix.")

    def _get_subj_idx(self):
        self.unique_subject_ids = np.sort(np.unique(self.subject_ids))
        self.n_subjects = len(self.unique_subject_ids)
        self.subj_idx = np.searchsorted(
            self.unique_subject_ids, self.subject_ids)

    def get_ols_estimates(self, signal):
        print("Estimating parameters using OLS...")

        signal = pd.DataFrame(signal, index=self.X.index)

        matrix = pd.concat((signal, self.X), 1)

        self.ols_betas = matrix.groupby(self.subject_ids).apply(do_ols)
        index = [(e,) + t for e, t in zip(self.ols_betas.index.get_level_values(0),
                                          self.ols_betas.index.get_level_values(1))]

        self.ols_betas.index = pd.MultiIndex.from_tuples(index,
                                                         names=['subject_id',
                                                                'event_type',
                                                                'covariate',
                                                                'regressor'])

        self.ols_betas_group = self.ols_betas.groupby(
            level=[1, 2, 3], sort=False).mean()
        self.ols_sd_group = self.ols_betas.groupby(
            level=[1, 2, 3], sort=False).std()

        if len(self.unique_subject_ids) == 1:
            self.ols_sd_group.iloc[:] = 1


class HierarchicalStanModel(HierarchicalModel):

    def __init__(self, X, subject_ids, subjectwise_errors=False, cauchy_priors=False, recompile=False, model_code=None):
        
        if pystan is None:
            raise ImportError("This feature requres pystan. Please install with 'pip install nideconv[stan]'")
    
        super(HierarchicalStanModel, self).__init__(
            X, subject_ids, subjectwise_errors)

        if model_code is not None:
            fn_string = model_code
        else:
            if subjectwise_errors:
                fn_string = 'subjectwise_errors'
            else:
                fn_string = 'groupwise_errors'

            if cauchy_priors:
                fn_string += '_cauchy'
            else:
                fn_string += '_normal'

        stan_model_fn_pkl = os.path.join(
            __dir__, 'stan_models', '%s.pkl' % fn_string)
        stan_model_fn_stan = os.path.join(
            __dir__, 'stan_models', '%s.stan' % fn_string)

        if not os.path.exists(stan_model_fn_pkl) or recompile:
            self.model = stan.StanModel(file=stan_model_fn_stan)

            with open(stan_model_fn_pkl, 'wb') as f:
                pkl.dump(self.model, f)

        else:
            with open(stan_model_fn_pkl, 'rb') as f:
                self.model = pkl.load(f)

    def sample(self, signal, chains=1, iter=1000, init_ols=False, *args, **kwargs):

        super(HierarchicalStanModel, self).sample(
            signal, chains, *args, **kwargs)

        data = {'measure': signal,
                'subj_idx': self.subj_idx + 1,
                'n': self.X.shape[0],
                'j': self.n_subjects,
                'm': self.X.shape[1],
                'X': self.X.values}

        if init_ols:
            init_dict = [self.get_init_dict(signal)] * chains
        else:
            init_dict = 'random'

        self.results = self.model.sampling(data=data,
                                           chains=chains,
                                           iter=iter,
                                           init=init_dict,
                                           *args,
                                           **kwargs)

    def get_subject_traces(self, melt=False):

        if not hasattr(self, 'results'):
            raise Exception('Model has not been sampled yet!')

        traces = self.results['beta_subject'].reshape((self.results['beta_subject'].shape[0],
                                                       np.prod(self.results['beta_subject'].shape[1:])))

        columns = [(c,) if type(
            c) is not tuple else c for c in self.X.columns.values]
        columns = [
            (sid,) + column for sid in self.unique_subject_ids for column in columns]
        columns = pd.MultiIndex.from_tuples(columns,
                                            names=['subject_id'] + self.X.columns.names)

        traces = pd.DataFrame(traces, columns=columns)

        if melt:
            return pd.melt(traces)
        else:
            return traces

    def get_group_traces(self, melt=False):

        if not hasattr(self, 'results'):
            raise Exception('Model has not been sampled yet!')

        traces = pd.DataFrame(
            self.results['beta_group'], columns=self.X.columns)
        if melt:
            return pd.melt(traces)
        else:
            return traces

    def get_group_parameters(self):

        if not hasattr(self, 'results'):
            raise Exception('Model has not been sampled yet!')

    def get_init_dict(self, signal):
        self.get_ols_estimates(signal)

        init_dict = {}
        init_dict['beta_subject_offset'] = self.ols_betas.unstack(
            level=0).T - self.ols_betas_group.unstack([0, 1, 2])['beta'].T
        init_dict['beta_subject_offset'] = init_dict['beta_subject_offset'][self.X.columns].values

        init_dict['beta_group'] = self.ols_betas_group.values.squeeze()
        init_dict['group_sd'] = self.ols_sd_group.values.squeeze()

        if self.subjectwise_errors:
            init_dict['eps'] = np.ones(len(self.unique_subject_ids))
        else:
            init_dict['eps'] = 1

        return init_dict
