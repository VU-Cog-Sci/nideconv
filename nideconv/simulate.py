import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from .utils import convolve_with_function


def simulate_fmri_experiment(conditions=None,
                             TR=1.,
                             n_subjects=1,
                             n_runs=1,
                             n_trials=40,
                             run_duration=300,
                             oversample=20,
                             noise_level=1.0,
                             n_rois=1,
                             kernel='double_gamma',
                             kernel_pars={}):
    """
    Simulates an fMRI experiment and returns a pandas
    DataFrame with the resulting time series in an analysis-ready format.

    By default a single run of a single subject is simulated, but a
    larger number of subjects, runs, and ROIs can also be simulated.

    Parameters
    ----------
    conditions : list of dictionaries or *None*
        Can be used to customize different conditions.
        Every conditions is represented as a dictionary in
        this list and has the following form:

        ::

            [{'name':'Condition A',
              'mu_group':1,
              'std_group':0.1},
              {'name':'Condition B',
              'mu_group':1,
              'std_group':0.1}]


        *mu_group* indicates the mean amplitude of the response
        to this condition across subjects.
        *std_group* indicates the standard deviation of this amplitude
        across subjects.

        Potentially, customized onsets can also be used as follows:

        ::

            {'name':'Condition A',
             'mu_group':1,
             'std_group':0.1
             'onsets':[10, 20, 30]}

    TR : float
        Indicates the time between volume acquistisions in seconds (Inverse
        of the sample rate).

    n_subjects : int
        Number of subjects.

    n_runs : int
        Number of runs *per subject*.

    n_trials : int
        Number of trials *per condition per run*. Only used when
        no custom onsets are provided (see *conditions*).

    run_duration : float
        Duration of a single run in seconds.

    noise_level : float
        Standard deviation of Gaussian noise added to time
        series.

    n_rois : int
        Number of regions-of-interest. Determines the number
        of columns of *data*.


    Other Parameters
    ----------------
    oversample : int
        Determines how many times the kernel is oversampled before
        convolution. Should usually not be changed.
    kernel : str
        Sets which kernel to use for response function. Currently
        only '`double_hrf`' can be used.



    Returns
    -------
    data : DataFrame
        Contains simulated time series with subj_idx, run and time (s)
        as index. Columns correspond to different ROIs
    onsets : DataFrame
        Contains used event onsets with subj_idx, run and trial type
        as index.
    parameters : DataFrame
        Contains parameters (amplitude) of the different event type.


    Examples
    --------

    By default, `simulate_fmri_experiment` simulates
    a 5 minute run with 40 trials for one subject

    >>> data, onsets, params = simulate_fmri_experiment()
    >>> print(data.head())
                        area 1
    subj_idx run t
    1        1   0.0 -1.280023
                 1.0  0.908086
                 2.0  0.850847
                 3.0 -1.010475
                 4.0 -0.299650
    >>> print(data.onsets)
                                  onset
    subj_idx run event_type
    1        1   A            94.317361
                 A           106.547084
                 A           198.175115
                 A            34.941112
                 A            31.323272
    >>> print(params)
                            amplitude
    subj_idx event_type
    1        A                 1.0
             B                 2.0


    With n_subjects we can increase the number of subjects

    >>> data, onsets, params = simulate_fmri_experiment(n_subjects=20)
    >>> data.index.get_level_values('subj_idx').unique()
    Int64Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20],
               dtype='int64', name='subj_idx')

    """

    if kernel not in ['double_gamma', 'gamma']:
        raise NotImplementedError()

    data = []

    if conditions is None:
        conditions = [{'name': 'Condition A',
                       'mu_group': 1,
                       'std_group': 0, },
                      {'name': 'Condition B',
                       'mu_group': 2,
                       'std_group': 0}]

    conditions = pd.DataFrame(conditions).set_index('name')

    sample_rate = 1./TR

    frametimes = np.arange(0, run_duration, TR)
    all_onsets = []

    parameters = []
    for subject in np.arange(1, n_subjects+1):

        for i, condition in conditions.iterrows():
            amplitude = sp.stats.norm(
                loc=condition['mu_group'], scale=condition['std_group']).rvs()
            condition['amplitude'] = amplitude
            condition['subject'] = subject
            condition['event_type'] = condition.name
            parameters.append(condition.drop(
                ['mu_group', 'std_group'], axis=0))

    parameters = pd.DataFrame(parameters).set_index(['subject', 'event_type'])

    if 'kernel' not in parameters.columns:
        parameters['kernel'] = kernel
    else:
        parameters['kernel'].fillna(kernel, inplace=True)

    if 'kernel_pars' not in parameters.columns:
        parameters['kernel_pars'] = np.nan

    if type(n_trials) is int:
        n_trials = [n_trials] * len(conditions)

    for subject in np.arange(1, n_subjects+1):

        for run in range(1, n_runs+1):

            signals = np.zeros((len(conditions), len(frametimes)))

            for i, (_, condition) in enumerate(conditions.iterrows()):
                if 'onsets' in condition:
                    onsets = np.array(condition.onsets)
                else:
                    onsets = np.ones(0)

                    while len(onsets) < n_trials[i]:
                        isis = np.random.gamma(
                            run_duration / n_trials[i], 1, size=n_trials[i] * 10)
                        onsets = np.cumsum(isis)
                        onsets = onsets[onsets < run_duration]

                    onsets = np.random.choice(onsets,
                                              n_trials[i],
                                              replace=False)

                signals[i, (onsets / TR).astype(int)
                        ] = parameters.loc[(subject, condition.name), 'amplitude']

                all_onsets.append(pd.DataFrame({'onset': onsets}))
                all_onsets[-1]['subject'] = subject
                all_onsets[-1]['run'] = run
                all_onsets[-1]['event_type'] = condition.name

                if np.isnan(parameters.loc[(subject, condition.name), 'kernel_pars']):
                    kernel_pars_ = kernel_pars
                else:
                    kernel_pars_ = parameters.loc[(
                        subject, condition.name), 'kernel_pars']

                signals[i] = convolve_with_function(
                    signals[i],
                    parameters.loc[(subject, condition.name), 'kernel'],
                    sample_rate,
                    **kernel_pars_
                )

            signal = signals.sum(0)
            signal = np.repeat(signal[:, np.newaxis], n_rois, 1)
            signal += np.random.randn(*signal.shape) * noise_level

            if n_rois == 1:
                columns = ['signal']
            else:
                columns = [f'area {d}' for d in range(1, n_rois + 1)]

            tmp = pd.DataFrame(signal,
                               columns=columns)

            tmp['t'] = frametimes
            tmp['subject'], tmp['run'] = subject, run

            data.append(tmp)

    data = pd.concat(data).set_index(['subject', 'run', 't'])
    onsets = pd.concat(all_onsets).set_index(['subject', 'run', 'event_type'])

    if n_subjects == 1:
        data.index = data.index.droplevel('subject')
        onsets.index = onsets.index.droplevel('subject')

    if n_runs == 1:
        data.index = data.index.droplevel('run')
        onsets.index = onsets.index.droplevel('run')

    return data, onsets, parameters
