import numpy as np
import pandas as pd
import pymc3 as pm
from .backends import HierarchicalStanModel
import warnings
import seaborn as sns
from .plotting import plot_hpd
    
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
            

    def get_group_timecourse_traces(self, melt=False, n=None):
        traces = self._model.get_group_traces()

        timecourses = []

        for event_key, event in self.response_fytters[0].events.items():
            
            for covariate in event.covariates.columns:
                columns = pd.MultiIndex.from_product([[event_key], [covariate], event.timepoints], 
                                                     names=['event type', 'covariate', 't'])
                tmp = pd.DataFrame(traces[event_key, covariate].dot(event.L))
                tmp.columns = columns
                timecourses.append(tmp)
                
        timecourses = pd.concat((timecourses), 1)

        return _process_timecourses(timecourses, melt, n)

    def get_subject_timecourse_traces(self, melt=False, n=None):
        timecourses = []

        
        traces = self._model.get_subject_traces()

        for subject_id in traces.columns.levels[0]:
            for event_key, event in self.response_fytters[0].events.items():
                for covariate in event.covariates.columns:
                    columns = pd.MultiIndex.from_product([[subject_id], [event_key], [covariate], event.timepoints], 
                                                         names=['subject_id', 'event type', 'covariate', 't'])
                    tmp = pd.DataFrame(traces[subject_id, event_key, covariate].dot(event.L))
                    tmp.columns = columns
                    timecourses.append(tmp)

        timecourses = pd.concat((timecourses), 1)
        return _process_timecourses(timecourses, melt, n)


    def plot_group_timecourses(self, 
                              hue='event type', 
                              col='covariate', 
                              alpha=0.05,
                              transparency=0.1,
                              row=None, 
                              covariates=None, 
                              event_types=None,
                              hline=True,
                              vline=True,
                              legend=True):
        
        if type(covariates) is str:
            covariates = [covariates]
            
        if type(event_types) is str:
            covariates = [event_types]        
        
        tc = self.get_group_timecourse_traces(True)

        if covariates is not None:
            tc = tc[np.in1d(tc.covariate, covariates)]

        if event_types is not None:
            tc = tc[np.in1d(tc['event type'], event_types)]        
            
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
