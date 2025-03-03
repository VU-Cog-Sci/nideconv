from nideconv.hierarchical_bayes.backends import HierarchicalStanModel
import numpy as np
from nose.tools import assert_greater
from numpy.testing import assert_allclose

def test_simple_model():

    n_subjects = 15
    n_cols = 5
    length_signal = 10

    beta_subject = np.random.randn(n_subjects, n_cols) + np.arange(n_cols)

    X = np.random.randn(n_subjects * length_signal,
                        n_cols)

    X[:, 0] = 1

    subj_ix = np.repeat(np.arange(n_subjects), length_signal)


    beta = beta_subject[subj_ix]
    Y = np.einsum('ij,ij->i', beta, X)
    Y += np.random.randn(*Y.shape)

    model = HierarchicalStanModel(X, subj_ix)
    model.sample(Y)

    r = np.corrcoef(model.get_subject_traces().mean(),
                    beta_subject.ravel())[0,0]
    
    assert_greater(r, 0.95)
    assert_allclose(model.get_group_traces().mean(),
                    np.arange(n_cols),
                    atol=.5)

