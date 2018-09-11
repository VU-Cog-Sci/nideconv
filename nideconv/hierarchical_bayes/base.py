import numpy as np
import pandas as pd
import pymc3 as pm
from .backends import HierarchicalStanModel
import warnings
import seaborn as sns
from .plotting import plot_hpd
import matplotlib.pyplot as plt

    
class HierarchicalBayesianModel(object):
    
    def __init__(self,
                 oversample_design_matrix=20):
        
        self.subj_idxs = []
        self.design_matrices = []
        self.signals = []
        self.response_fitters = []

        self.oversample_design_matrix = oversample_design_matrix
        
    def add_run(self, fitter, subj_idx):

        if len(self.design_matrices) == 0:     
            self.subj_idxs.append(subj_idx)
            self.signals.append(fitter.input_signal)
            self.design_matrices.append(fitter.X)
            self.response_fitters.append(fitter)
        else:
            self.subj_idxs.append(subj_idx)
            self.signals.append(fitter.input_signal)
            
            if not ((fitter.X.columns == self.design_matrices[0].columns).all()):
                raise Exception('Different design matrices!')
            
            self.design_matrices.append(fitter.X)
            self.response_fitters.append(fitter)            
            
            
    def build_model(self, backend='stan', subjectwise_errors=False,
                    cauchy_priors=False,
                    *args, **kwargs):
        
        self.X = pd.concat(self.design_matrices)
        self.signal = np.concatenate(self.signals, 0).squeeze()
        
        subject_labels = np.concatenate([[subj_idx] * len(self.design_matrices[i]) for i, subj_idx in enumerate(self.subj_idxs)])        

        if backend == 'stan':
            self._model = HierarchicalStanModel(self.X,
                                                subject_labels,
                                                subjectwise_errors,
                                                cauchy_priors,
                                                *args,
                                                **kwargs)
        elif backend == 'pymc3':
            raise NotImplementedError()

    def sample(self, chains=1, iter=1000, init_ols=False, *args, **kwargs):
        self._model.sample(self.signal, chains=chains, iter=iter, init_ols=init_ols, *args, **kwargs)


    def get_mean_group_timecourse(self,
                                  oversample=None,
                                  melt=False):
        
        traces = self.get_group_timecourse_traces(oversample=oversample)

        if melt:
            return traces.mean().to_frame('value')
        else:
            return traces.mean().to_frame('value').T

    def get_mean_subject_timecourses(self,
                                     oversample=None,
                                     melt=False):

        traces = self.get_subject_timecourse_traces(oversample=oversample)

        if melt:
            return traces.mean().to_frame('value')
        else:
            return traces.mean().to_frame('value').T


    def get_group_timecourse_traces(self, 
                                    oversample=None, 
                                    melt=False, 
                                    n=None):

        if oversample is None:
            oversample = self.oversample_design_matrix

        traces = self._model.get_group_traces()

        timecourses = []

        for event_key, event in self.response_fitters[0].events.items():
            
            for covariate in event.covariates.columns:
                L = event.get_basis_function(oversample=oversample)
                columns = pd.MultiIndex.from_product([[event_key], [covariate], L.index], 
                                                     names=['event_type', 'covariate', 't'])
                tmp = traces[event_key, covariate].dot(L.T)
                tmp.columns = columns
                timecourses.append(tmp)
                
        timecourses = pd.concat((timecourses), 1)

        return _process_timecourses(timecourses, melt, n)

    def get_subject_timecourse_traces(self,
                                      oversample=None,
                                      melt=False,
                                      n=None):

        if oversample is None:
            oversample = self.oversample_design_matrix

        timecourses = []
        traces = self._model.get_subject_traces()

        for subject_id in traces.columns.levels[0]:
            for event_key, event in self.response_fitters[0].events.items():
                for covariate in event.covariates.columns:
                    L = event.get_basis_function(oversample=oversample)
                    columns = pd.MultiIndex.from_product([[subject_id], [event_key], [covariate], L.index], 
                                                         names=['subject_id', 'event_type', 'covariate', 't'])
                    tmp = traces[subject_id, event_key, covariate].dot(L.T)
                    tmp.columns = columns
                    timecourses.append(tmp)

        timecourses = pd.concat((timecourses), 1)
        return _process_timecourses(timecourses, melt, n)


    def plot_group_timecourses(self, 
                               oversample=None,
                               hue='event_type', 
                               col='covariate', 
                               alpha=0.05,
                               transparency=0.1,
                               row=None, 
                               covariates=None, 
                               event_types=None,
                               hline=True,
                               vline=True,
                               legend=True):
        
        
        tc = self.get_group_timecourse_traces(oversample=oversample,
                                              melt=True)

        if covariates is not None:
            tc = tc[np.in1d(tc.covariate, covariates)]

        if event_types is not None:
            tc = tc[np.in1d(tc['event_type'], event_types)]        
            
        fac = sns.FacetGrid(tc, hue=hue, col=col, row=row, aspect=1.5)
        fac.map_dataframe(plot_hpd, alpha=alpha, transparency=transparency)
        
        if hline:
            fac.map(plt.axhline, c='k', ls='--') 
            
        if vline:
            fac.map(plt.axvline, c='k', ls='--') 
        
        if legend:
            fac.add_legend()
            for patch in fac._legend.get_patches():
                patch.set_alpha(.8)
            
        return fac 

    def plot_subject_timecourses(self, 
                                 oversample=None,
                                 hue='event_type', 
                                 col=None,
                                 row=None,
                                 subject_ids=None,
                                 alpha=0.05,
                                 transparency=0.1,
                                 covariates=None, 
                                 event_types=None,
                                 hline=True,
                                 vline=True,
                                 col_wrap=4,
                                 sharex=True,
                                 sharey=True,
                                 legend=True):


        tc = self.get_subject_timecourse_traces(oversample=oversample,
                                                melt=True)

        if covariates is not None:
            tc = tc[np.in1d(tc.covariate, covariates)]

        if event_types is not None:
            tc = tc[np.in1d(tc['event_type'], event_types)] 
            
        if subject_ids is not None:
            tc = tc[np.in1d(tc['subject_id'], subject_ids)]          
        
        if col is None:
            if len(tc.covariate.unique()) == 1:
                col = 'subject_id'
                col_wrap = col_wrap
            else:
                col = 'covariate'
                row = 'subject_id'
                col_wrap = None

        fac = sns.FacetGrid(tc, hue=hue, col=col, row=row, col_wrap=col_wrap, aspect=1.5)
        fac.map_dataframe(plot_hpd, alpha=alpha, transparency=transparency)

        if hline:
            fac.map(plt.axhline, c='k', ls='--') 

        if vline:
            fac.map(plt.axvline, c='k', ls='--') 

        if legend:
            fac.add_legend()
            for patch in fac._legend.get_patches():
                patch.set_alpha(.8)



        return fac 

    @classmethod
    def from_groupresponsefitter(cls, group_response_fitter):
        hbm = cls(oversample_design_matrix=group_response_fitter.oversample_design_matrix)

        subject_col = group_response_fitter.response_fitters.index.names.index('subject')

        for ix, rf in group_response_fitter.response_fitters.items():
            hbm.add_run(rf, ix[subject_col])

        return hbm

    
def _process_timecourses(timecourses, melt, n):

    if n is not None:
        if n > len(timecourses):
            warnings.warn('You asked for %d samples, but there are only %d' % (n, len(timecourses)))

        stepsize = np.max((np.floor(len(timecourses) / n), 1)).astype(int)
    
        timecourses = timecourses.iloc[::stepsize]
        timecourses = timecourses.iloc[:n]

    if melt:
        timecourses['sample'] = timecourses.index
        return pd.melt(timecourses, id_vars=['sample'])
    else:
        return timecourses
