import pystan
import os
import pickle as pkl

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

def _get_hierarchical_stan_model(rebuild=False):

    stan_model_fn = os.path.join(__dir__, 'stanmodel.pkl')

    if not os.path.exists(stan_model_fn) or rebuild:
        sm = pystan.StanModel(model_code=stan_code)

        with open(stan_model_fn, 'wb') as f:
            pkl.dump(sm, f)

    else:
        with open(stan_model_fn, 'rb') as f:
            sm = pkl.load(f)

    return sm

