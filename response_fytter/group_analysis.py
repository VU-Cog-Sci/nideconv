from .response_fytter import ResponseFytter
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from .plotting import plot_timecourses
import scipy as sp

class GroupResponseFytter(object):

    def __init__(self,
                 timeseries,
                 behavior,
                 input_sample_rate,
                 oversample_design_matrix=20,
                 confounds=None,
                 *args,
                 **kwargs):

        timeseries = pd.DataFrame(timeseries)

        self.timeseries = timeseries
        self.onsets = behavior
        self.confounds = confounds

        self.oversample_design_matrix = oversample_design_matrix

        if 'trial_type' not in self.onsets:
            self.onsets['trial_type'] = 'intercept'

        self.index_columns = []

        for c in ['subj_idx', 'run']:
            if c in self.timeseries.columns:
                self.index_columns.append(c)

        self.timeseries['t'] = self.timeseries.groupby(self.index_columns).apply(_make_time_column, 
                                                                                 input_sample_rate)


        self.response_fitters = []

        if self.index_columns is []:
            raise Exception('GroupResponseFytter is only to be used for datasets with multiple subjects'
                             'or runs')
        else:
            self.timeseries = self.timeseries.set_index(self.index_columns + ['t'])
            self.onsets = self.onsets.set_index(self.index_columns + ['trial_type'])

            if self.confounds is not None:
                self.confounds['t'] = self.confounds.groupby(self.index_columns).apply(_make_time_column, 
                                                                                       input_sample_rate)
                self.confounds = self.confounds.set_index(self.index_columns + ['t'])

            for idx, ts in self.timeseries.groupby(level=self.index_columns):
                rf = ResponseFytter(ts,
                                    input_sample_rate,
                                    self.oversample_design_matrix,
                                    *args,
                                    **kwargs)
                self.response_fitters.append(rf)
                if self.confounds is not None:
                    self.response_fitters[-1].add_confounds('confounds', self.confounds.loc[idx])


    def add_event(self,
                 event=None,
                 basis_set='fir', 
                 interval=[0,10], 
                 n_regressors=None, 
                 covariates=None,
                 add_intercept=True,
                 **kwargs):

        if event is None:
            event = self.onsets.index.get_level_values('trial_type').unique()
        if event is str:
            event = [event]

        if type(covariates) is str:
            covariates = [covariates]

        for i, (col, ts) in self._groupby_ts():
            for e in event:
                
                if col + (e,) not in self.onsets.index:
                    warnings.warn('Event %s is not available for run %s. Event is ignored for this '
                                  'run' % (e, col))
                else:
                    if type(col) is not tuple:
                        col = (col,)

                    if covariates is None:
                        covariate_matrix = None
                    else:
                        covariate_matrix = self.onsets.loc[col + (e,), covariates]

                        if add_intercept:
                            intercept_matrix = pd.DataFrame(np.ones((len(covariate_matrix), 1)),
                                                            columns=['intercept'],
                                                            index=covariate_matrix.index)
                            covariate_matrix = pd.concat((intercept_matrix, covariate_matrix), 1)

                    self.response_fitters[i].add_event(e,
                                                       onset_times=self.onsets.loc[col + (e,), 'onset'],
                                                       basis_set=basis_set,
                                                       interval=interval,
                                                       n_regressors=n_regressors,
                                                       covariates=covariate_matrix)


    def fit(self):
        for response_fitter in self.response_fitters:
            response_fitter.regress()

    def get_timecourses(self, oversample=None):

        if oversample is None:
            oversample = self.oversample_design_matrix

        df = []
        for i, (col, ts) in self._groupby_ts():

            if type(col) is not tuple:
                col = (col,)

            tc = self.response_fitters[i].get_timecourses(oversample)

            for ic, value in zip(self.index_columns, col):
                tc[ic] = value
            

            df.append(tc)

        return pd.concat(df).reset_index().set_index(self.index_columns + tc.index.names)

    def _groupby_ts(self): 
        return enumerate(self.timeseries.groupby(level=self.index_columns))


    def get_subjectwise_timecourses(self, 
                                    oversample=None, 
                                    melt=False):

        tc = self.get_timecourses(oversample=oversample)
        tc = tc.reset_index().groupby(['subj_idx', 'event type','covariate', 'time', ]).mean()

        for c in self.index_columns:
            if c in tc.columns:
                tc.drop(columns=c, inplace=True)

        if melt:
            return tc.reset_index().melt(id_vars=tc.index.names,
                                         var_name='roi')
        else:
            return tc

    def get_conditionwise_timecourses(self,
                                      oversample=None,
                                      kind='mean'):

        subj_tc = self.get_subjectwise_timecourses(oversample)

        if kind == 'mean':
            return subj_tc.groupby(level=['event type', 'covariate', 'time']).mean()

        else:
            t = (self.get_timecourses(oversample)
                     .groupby(level=['event type', 'covariate', 'time'])
                     .apply(lambda d: pd.Series(sp.stats.ttest_1samp(d, 0, 0)[0], index=d.columns)
                     .T)
                 )

            if kind =='t':
                return t

            elif kind == 'z':
                t_dist = sp.stats.t(len(self.timeseries.index.get_level_values('subj_idx')
                                                       .unique()))
                norm_dist = sp.stats.norm()

                return pd.DataFrame(norm_dist.ppf(t_dist.cdf(t)),
                                    columns=t.columns,
                                    index=t.index)
            else:
                raise NotImplementedError("kind should be 'mean', 't', or 'z'")

    def plot_groupwise_timecourses(self,
                                   plots='roi',
                                   col='covariate',
                                   row=None,
                                   col_wrap=None,
                                   hue='event type',
                                   max_n_plots=40,
                                   oversample=None,
                                   extra_axes=True,
                                   sharex=True,
                                   sharey=False,
                                   aspect=1.5,
                                   *args,
                                   **kwargs):

        tc = self.get_subjectwise_timecourses(oversample=oversample,
                                              melt=True)

        return plot_timecourses(tc,
                                plots=plots,
                                col=col,
                                row=row,
                                col_wrap=col_wrap,
                                hue=hue,
                                max_n_plots=max_n_plots,
                                oversample=oversample,
                                extra_axes=extra_axes,
                                sharex=sharex,
                                sharey=sharey,
                                aspect=aspect,
                                *args,
                                **kwargs)

        


def _make_time_column(d, sample_rate):
    return pd.DataFrame(np.arange(0, len(d) * 1./sample_rate, 1./sample_rate), index=d.index)

