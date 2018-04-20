data {
    int<lower=0> n; // number of observations
    int<lower=0> m; // number of predictors
    int<lower=0> j; // number of groups

    real measure[n];
    matrix[n, m] X;
    int<lower=0> subj_idx[n];
}

parameters {
    vector<lower=0>[j] eps;
    row_vector[m] beta_group;
    matrix[j, m] beta_subject_offset;
    row_vector<lower=0>[m] group_sd;

}
transformed parameters {

    matrix[j, m] beta_subject;

    for (i in 1:j)
        beta_subject[i, :] = beta_group + group_sd .* beta_subject_offset[i];

}

model {
    beta_group ~ normal(0, 2.5);
    group_sd ~ normal(0, 2.5);

    to_vector(beta_subject_offset) ~ normal(0, 2.5);

    eps ~ normal(0, 2.5);

    measure ~ normal(rows_dot_product(X, beta_subject[subj_idx]), eps[subj_idx]);
}
