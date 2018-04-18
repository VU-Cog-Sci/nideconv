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
    matrix[n, m] beta;
    for (i in 1:n)
        beta[i, :] = beta_subject[subj_idx[i]];
    beta_group ~ normal(0, 10);
    to_vector(beta_subject_offset) ~ normal(0, 1);
    to_vector(group_sd) ~ cauchy(0, 2.5);
    eps ~ cauchy(0, 2.5);
    measure ~ normal(rows_dot_product(X, beta), eps);
}
