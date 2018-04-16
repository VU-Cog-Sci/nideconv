import numpy as np
import pandas as pd
import pymc3 as pm
    
class HierarchicalBayesianModel(object):
    
    def __init__(self):
        
        self.subj_idxs = []
        self.design_matrices = []
        self.signals = []
        self.response_fytters = []
        
    def add_run(self, fytter, subj_idx):
        
        if len(self.design_matrices) == 0:            
            self.subj_idxs.append(subj_idx)
            self.signals.append(fytter.input_signal)
            self.design_matrices.append(fytter.X)
            self.response_fytters.append(fytter)
        else:
            self.subj_idxs.append(subj_idx)
            self.signals.append(fytter.input_signal)
            
            if not ((fytter.X.columns == self.design_matrices[0].columns).all()):
                raise Exception('Different design matrices!')
            
            self.design_matrices.append(fytter.X)
            self.response_fytters.append(fytter)            
            
            
    def build_model(self):
        
        self._count_subjects()
        
        dm = pd.concat(self.design_matrices).values
        signal = np.concatenate(self.signals, 0).squeeze()
        
        subject_labels = np.concatenate([[subj_idx] * len(self.design_matrices[i]) for i, subj_idx in enumerate(self.subj_idxs)])        
        subjects_ix = np.searchsorted(self.unique_subj_idx, subject_labels)
        
        with pm.Model() as self.model:
            hyperpriors_mu = pm.Cauchy('hyperpriors_mu', 0, 5, shape=(1, dm.shape[1]))
            hyperpriors_sd = pm.HalfCauchy('hyperpriors_sd', 5, shape=(1, dm.shape[1]))
            
            subjectwise_offsets = pm.Cauchy('subjectwise_offsets', 0, 5, shape=(self.n_subjects, dm.shape[1]))
            subjectwise_est = pm.Deterministic("subjectwise_parameters", hyperpriors_mu + subjectwise_offsets * hyperpriors_sd)
            
            eps = pm.HalfCauchy('eps', 5)
            
            signal_est = dm[:, 0] * subjectwise_est[subjects_ix, 0]
            
            for regressor in range(1, dm.shape[1]):
                signal_est += dm[:, regressor] * subjectwise_est[subjects_ix, regressor]
            
            residuals = signal - signal_est
            
            likelihood = pm.Normal('like', mu=0, sd=eps, observed=residuals)
            
            
    def _count_subjects(self):
        self.unique_subj_idx = np.sort(np.unique(self.subj_idxs))
        self.n_subjects = len(self.unique_subj_idx)
        
        
    def sample(self, *args, **kwargs):
        with self.model:
            self.results = pm.sample(*args, **kwargs)
