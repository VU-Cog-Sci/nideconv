from .response_fitter import ResponseFitter, ConcatenatedResponseFitter
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from .plotting import plot_timecourses
import scipy as sp
import logging


class GroupResponseFitter(object):

    """Can fit a group of individual subjects and/or
    runs using a high-level interface.
    """

    def __init__(self,
                 timeseries,
                 onsets,
                 input_sample_rate,
                 oversample_design_matrix=20,
                 confounds=None,
                 concatenate_runs=True,
                 *args,
                 **kwargs):

        timeseries = pd.DataFrame(timeseries.copy())

        self.timeseries = timeseries
        self.onsets = onsets.copy().reset_index()

        self.confounds = confounds

        self.concatenate_runs = concatenate_runs

        self.oversample_design_matrix = oversample_design_matrix

        self.index_columns = []

        idx_fields = ['subject', 'session', 'task', 'run']

        for field in idx_fields:
            if field in self.onsets.index.names:
                self.onsets.reset_index(field, inplace=True)

            if field in self.timeseries.index.names:
                self.timeseries.reset_index(field, inplace=True)

            if confounds is not None:
                if field in self.confounds.index.names:
                    self.confounds.reset_index(field, inplace=True)

        if 'event' in self.onsets.index.names:
            self.onsets.reset_index('event', inplace=True)

        if ('subject' not in self.timeseries.columns) and ('subject' not in self.onsets.columns):
            self.timeseries['subject'] = 1
            self.onsets['subject'] = 1

            if self.confounds is not None:
                self.confounds['subject'] = 1

        for c in idx_fields:
            if c in self.timeseries.columns:
                self.index_columns.append(c)

                if self.timeseries[c].dtype != self.onsets[c].dtype:
                    logging.warning('The dtype of column "{}" is different for '
                                    'onsets ({}) and timeseries ({}).\n'
                                    'Converting dtype of onsets to {}...'.format(c,
                                                                                 self.onsets[c].dtype,
                                                                                 self.timeseries[c].dtype,
                                                                                 self.timeseries[c].dtype))
                    self.onsets[c] = self.onsets[c].astype(
                        self.timeseries[c].dtype)

        if 'event_type' not in self.onsets:
            if 'condition' in self.onsets:
                self.onsets['event_type'] = self.onsets['condition']
            else:
                self.onsets['event_type'] = 'intercept'

        index = pd.MultiIndex(names=self.index_columns,
                              levels=[[]]*len(self.index_columns),
                              codes=[[]]*len(self.index_columns),)
        self.response_fitters = pd.Series(index=index)

        if self.index_columns is []:
            raise Exception('GroupDeconvolution is only to be used for datasets with multiple subjects'
                            'or runs')
        else:
            self.timeseries = self.timeseries.set_index(self.index_columns)
            self.timeseries['t'] = _make_time_column(self.timeseries,
                                                     self.index_columns,
                                                     input_sample_rate)
            self.timeseries.set_index('t', inplace=True, append=True)

            self.onsets = self.onsets.set_index(
                self.index_columns + ['event_type'])

            if self.confounds is not None:
                self.confounds = self.confounds.set_index(self.index_columns)
                self.confounds['t'] = _make_time_column(self.confounds,
                                                        self.index_columns,
                                                        input_sample_rate)

                self.confounds = self.confounds.set_index('t', append=True)

            for idx, ts in self.timeseries.groupby(level=self.index_columns):

                rf = ResponseFitter(ts,
                                    input_sample_rate,
                                    self.oversample_design_matrix,
                                    *args,
                                    **kwargs)
                self.response_fitters.loc[idx] = rf
                if self.confounds is not None:
                    if type(idx) is int:
                        idx = idx,
                    self.response_fitters.loc[idx].add_confounds(
                        'confounds', self.confounds.loc[idx])

    def add_event(self,
                  event=None,
                  basis_set='fir',
                  interval=[0, 10],
                  n_regressors=None,
                  covariates=None,
                  add_intercept=True,
                  show_warnings=True
                  **kwargs):

        if not hasattr(self, 'events'):
            self.events = []

        if event is None:
            event = self.onsets.index.get_level_values('event_type').unique()
            if show_warnings:
                logging.warning(
                'No event type was given, automatically entering the following event types: %s' % event.tolist())

        if type(event) is str:
            event = [event]

        if type(covariates) is str:
            covariates = [covariates]

        for e in event:
            self.events += e

        for i, (col, ts) in self._groupby_ts_runs():
            for e in event:

                if type(col) is not tuple:
                    col = (col,)

                if col + (e,) not in self.onsets.index:
                    if show_warnings:
                        warnings.warn('Event %s is not available for run %s. Event is ignored for this '
                                  'run' % (e, col))
                else:

                    if covariates is None:
                        covariate_matrix = None
                    else:
                        covariate_matrix = self.onsets.loc[[
                            col + (e,)], covariates]

                        if add_intercept:
                            intercept_matrix = pd.DataFrame(np.ones((len(covariate_matrix), 1)),
                                                            columns=[
                                                                'intercept'],
                                                            index=covariate_matrix.index)
                            covariate_matrix = pd.concat(
                                (intercept_matrix, covariate_matrix), 1)

                    if 'duration' in self.onsets and np.isfinite(self.onsets.loc[[col + (e,)], 'duration']).all():
                        durations = self.onsets.loc[[col + (e,)], 'duration']
                    else:
                        durations = None

                    # print(e, self.onsets.loc[[col + (e,)], 'onset'].shape, durations.sha[p)

                    self.response_fitters[col].add_event(e,
                                                         onsets=self.onsets.loc[[
                                                             col + (e,)], 'onset'],
                                                         basis_set=basis_set,
                                                         interval=interval,
                                                         n_regressors=n_regressors,
                                                         durations=durations,
                                                         covariates=covariate_matrix)


    def fit(self,
            concatenate_runs=None,
            type='ols',
            cv=20,
            alphas=None,
            store_residuals=False,
            progressbar=False):

        if progressbar:
            from tqdm import tqdm
            bar = tqdm
        else:
            bar = lambda x: x

        if concatenate_runs is None:
            concatenate_runs = self.concatenate_runs

        if concatenate_runs:
            self.concat_response_fitters = \
                self.response_fitters.groupby('subject') \
                                     .apply(ConcatenatedResponseFitter)

            for key, concat_rf in bar(self.concat_response_fitters.iteritems()):
                try:
                    concat_rf.fit(type,
                                  cv,
                                  alphas,
                                  store_residuals)

                except Exception as e:
                    print(f'Problem with responsefitter {key}: {e}')

        else:
            for key, rf in bar(self.response_fitters.iteritems()):
                try:
                    rf.fit(type,
                           cv,
                           alphas,
                           store_residuals)
                except Exception as e:
                    ix = dict(zip(self.index_columns, key))
                    print(f'Problem with responsefitter {ix}: {e}')

    def get_timecourses(self, oversample=None,
                        melt=False,
                        concatenate_runs=None):

        if oversample is None:
            oversample = self.oversample_design_matrix

        rfs = self._get_response_fitters(concatenate_runs)

        tc_ = rfs.apply(lambda rf: rf.get_timecourses(oversample))

        tc = pd.concat(tc_.to_dict())
        index_names = tc_.index.names
        tc.index.set_names(index_names, level=range(len(index_names)),
                           inplace=True)

        if melt:
            return tc.reset_index().melt(id_vars=tc.index.names,
                                         var_name='roi')
        else:
            return tc

        return tc

    def get_t_value_timecourses(self,
                                oversample=None,
                                concatenate_runs=None):

        if oversample is None:
            oversample = self.oversample_design_matrix

        rfs = self._get_response_fitters(concatenate_runs)

        t_ = rfs.apply(lambda rf: rf.get_t_value_timecourses(oversample))

        t = pd.concat(t_.to_dict())
        index_names = t_.index.names
        t.index.set_names(index_names, level=range(len(index_names)),
                          inplace=True)

        return t

    def _get_response_fitters(self,
                              concatenate_runs=None):

        if concatenate_runs is None:
            concatenate_runs = self.concatenate_runs

        if concatenate_runs:
            if not hasattr(self, 'concat_response_fitters'):
                raise Exception('GroupResponseFitter not yet fitted')
            rfs = self.concat_response_fitters
        else:
            rfs = self.response_fitters

        return rfs

    def _groupby_ts_runs(self):
        return enumerate(self.timeseries.groupby(level=self.index_columns))

    def get_subjectwise_timecourses(self,
                                    oversample=None,
                                    melt=False):

        tc = self.get_timecourses(oversample=oversample)
        tc = tc.reset_index().groupby(
            ['subject', 'event type', 'covariate', 'time', ]).mean()

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

            if kind == 't':
                return t

            elif kind == 'z':
                t_dist = sp.stats.t(len(self.timeseries.index.get_level_values('subject')
                                                       .unique()))
                norm_dist = sp.stats.norm()

                return pd.DataFrame(norm_dist.ppf(t_dist.cdf(t)),
                                    columns=t.columns,
                                    index=t.index)
            else:
                raise NotImplementedError("kind should be 'mean', 't', or 'z'")

    def plot_groupwise_timecourses(self,
                                   event_types=None,
                                   covariates=None,
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

        if event_types is not None:

            if type(event_types) is str:
                event_types = [event_types]

            tc = tc[np.in1d(tc['event type'], event_types)]

        if covariates is not None:

            if type(covariates) is str:
                covariates = [covariates]

            tc = tc[np.in1d(tc['covariate'], covariates)]

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

    def plot_subject_timecourses(self,
                                 event_types=None,
                                 covariates=None,
                                 plots='roi',
                                 col='subject',
                                 row=None,
                                 col_wrap=4,
                                 hue='event type',
                                 max_n_plots=40,
                                 oversample=None,
                                 extra_axes=True,
                                 sharex=True,
                                 sharey=False,
                                 aspect=1.5,
                                 unit=None,
                                 *args,
                                 **kwargs):

        tc = self.get_timecourses(oversample=oversample,
                                  melt=True)

        if event_types is not None:

            if type(event_types) is str:
                event_types = [event_types]

            tc = tc[np.in1d(tc['event type'], event_types)]

        if covariates is not None:

            if type(covariates) is str:
                covariates = [covariates]

            tc = tc[np.in1d(tc['covariate'], covariates)]

        if unit is None:
            if 'run' in self.index_columns:
                unit = 'run'
            else:
                unit = 'subject'

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
                                unit=unit,
                                *args,
                                **kwargs)

    def get_rsq(self):
        if self.concatenate_runs:
            rfs = self.concat_response_fitters
        else:
            rfs = self.response_fitters

        rsq = pd.concat(rfs.apply(lambda d: d.get_rsq()).tolist())

        rsq.index = rfs.index

        return rsq

    def get_sse(self):
        if self.concatenate_runs:
            rfs = self.concat_response_fitters
        else:
            rfs = self.response_fitters

        sse = pd.DataFrame([rf.sse for rf in rfs], index=rfs.index, columns=self.timeseries.columns)

        return sse

    def get_time_to_peak(self,
                         oversample=1.0,
                         cutoff=1.0,
                         negative_peak=False,
                         include_prominence=False):
        ttp = pd.concat(self.response_fitters.apply(lambda d: d.get_time_to_peak(oversample=oversample,
                                                                                 cutoff=cutoff,
                                                                                 negative_peak=negative_peak,
                                                                                 include_prominence=include_prominence)).tolist())
        ttp.index = self.response_fitters.index

        return ttp


    def get_epochs(self, event, interval):

        if hasattr(self, 'events') and (event in self.events):
            warnings.warn(f'{event} is already in the events list, note that it will be regressed ' \
                          'out before extracting the epoch!!')

        epochs = []
        for key, (col, ts) in self._groupby_ts_runs():
            onsets = self.onsets.loc[(col) + (event,)]
            e = self.response_fitters[col].get_epochs(onsets['onset'], interval)

            for key, name in zip(col, self.index_columns):
                e[name] = key
                e.set_index(name, append=True, inplace=True)

            epochs.append(e.reorder_levels(self.index_columns + ['onset'], axis=0))

        return pd.concat(epochs)

def _make_time_column(df, index_columns, sample_rate):
    t = pd.Series(np.zeros(len(df)), index=df.index)

    TR = 1./sample_rate

    for ix, d in df.groupby(index_columns):
        t.loc[ix] = np.linspace(0, len(df.loc[ix]) * TR, len(d), endpoint=False)

    return t
