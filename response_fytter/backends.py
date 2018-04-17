import pystan
import os
import pickle as pkl
import numpy as np
import pymc3 as pm
import theano.tensor as T

__dir__ = os.path.abspath(os.path.dirname(__file__))

stan_code = """
data {
    int<lower=0> n; // number of observations
    int<lower=0> m; // number of predictors
    int<lower=0> j; // number of groups

    real measure[n];
    matrix[n, m] X;
    int<lower=0> subj_idx[n];
}

parameters {
    real<lower=0> eps;
    row_vector[m] group_beta;
    matrix[j, m] beta_subject_offset;
    row_vector<lower=0>[m] group_sd;

}
transformed parameters {
    matrix[n, m] beta;

    for (i in 1:n)
        beta[i, :] = group_beta + group_sd .* beta_subject_offset[subj_idx[i]];

}
model {



    to_vector(group_beta) ~ normal(0, 10);
    to_vector(beta_subject_offset) ~ normal(0, 1);
    to_vector(group_sd) ~ cauchy(0, 2.5);

    eps ~ cauchy(0, 2.5);

    measure ~ normal(rows_dot_product(X, beta), eps);
}
"""

class HierarchicalModel(object):

    def __init__(self, X, subject_ids):
        
        self.X = np.array(X)
        self.subject_ids = np.array(subject_ids).squeeze()

        if(self.subject_ids.shape[0] != self.X.shape[0]):
            raise Exception("Number of subjects indices should" \
                            "correspond to number of rows in the" \
                            "design matrices.")
        self._get_subj_idx()

    def sample(self, signal):
        pass

    def _get_subj_idx(self):
        self.unique_subject_ids = np.sort(np.unique(self.subject_ids))
        self.n_subjects = len(self.unique_subject_ids)
        self.subj_idx= np.searchsorted(self.unique_subject_ids, self.subject_ids)

class HierarchicalStanModel(HierarchicalModel):

    def __init__(self, X, subject_ids, recompile=False):
        
        super(HierarchicalStanModel, self).__init__(X, subject_ids)

        stan_model_fn = os.path.join(__dir__, 'stanmodel.pkl')

        if not os.path.exists(stan_model_fn) or recompile:
            self.model = pystan.StanModel(model_code=stan_code)

            with open(stan_model_fn, 'wb') as f:
                pkl.dump(self.model, f)

        else:
            with open(stan_model_fn, 'rb') as f:
                self.model = pkl.load(f)

    def sample(self, signal, chains=1, iter=1000, *args, **kwargs):

        measure = signal.squeeze()
        if(len(measure) != self.X.shape[0]):
            raise Exception("Signal should have same number of elements" \
                            "as rows in the design matrix.")

        data = {'measure':measure, 
                'subj_idx':self.subj_idx + 1,
                'n':self.X.shape[0],
                'j':self.n_subjects,
                'm':self.X.shape[1],
                'X':self.X}

        self.results = self.model.sampling(data=data, 
                                           chains=chains,
                                           iter=iter,
                                           *args,
                                           **kwargs)

class HierarchicalPymc3Model(HierarchicalModel):

    def __init__(self, X, subject_ids, recompile=False):
        
        super(HierarchicalPymc3Model, self).__init__(X, subject_ids)

            

    def sample(self, signal, chains=1, iter=1000, *args, **kwargs):

        with pm.Model() as self.model:
            hyperpriors_mu = pm.Cauchy('hyperpriors_mu', 0, 5, shape=(self.X.shape[1], 1))
            hyperpriors_sd = pm.HalfCauchy('hyperpriors_sd', 5, shape=(self.X.shape[1], 1))
            
            subjectwise_offsets = pm.Cauchy('subjectwise_offsets', 0, 5, shape=(self.X.shape[1], self.n_subjects))
            subjectwise_est = pm.Deterministic("subjectwise_parameters", hyperpriors_mu + subjectwise_offsets * hyperpriors_sd)
            
            eps = pm.HalfCauchy('eps', 5)

            #signal_est = self.X[:, 0] * subjectwise_est[self.subj_idx, 0]
            
            #for regressor in range(1, self.X.shape[1]):
                #signal_est += self.X[:, regressor] * subjectwise_est[self.subj_idx, regressor]

            signal_est = T.dot(self.X, subjectwise_est[:, self.subj_idx])
            residuals = signal - signal_est
            likelihood = pm.Normal('like', mu=0, sd=eps, observed=residuals)

            self.results = pm.sample(draws=iter, chains=chains, *args, **kwargs)
