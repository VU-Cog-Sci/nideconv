import numpy as np
import pandas as pd
import pymc3 as pm
from .backends import HierarchicalStanModel
    
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
            
            
    def build_model(self, backend='stan', subjectwise_errors=False, *args, **kwargs):
        
        self.X = pd.concat(self.design_matrices)
        self.signal = np.concatenate(self.signals, 0).squeeze()
        
        subject_labels = np.concatenate([[subj_idx] * len(self.design_matrices[i]) for i, subj_idx in enumerate(self.subj_idxs)])        

        if backend == 'stan':
            self._model = HierarchicalStanModel(self.X, subject_labels, *args, **kwargs)
        elif backend == 'pymc3':
            raise NotImplementedError()

    def sample(self, chains=1, iter=1000, *args, **kwargs):
        self._model.sample(self.signal, chains=chains, iter=iter, *args, **kwargs)
            
    def _count_subjects(self):
        self.unique_subj_idx = np.sort(np.unique(self.subj_idxs))
        self.n_subjects = len(self.unique_subj_idx)
        
