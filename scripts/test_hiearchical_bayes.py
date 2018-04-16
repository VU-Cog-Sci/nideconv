import response_fytter
from response_fytter.simulate import simulate_fmri_experiment
from response_fytter.hiearchical_bayes import HierarchicalBayesianModel
import pandas as pd
import numpy as np

sample_rate = 5

data, onsets, parameters = simulate_fmri_experiment([{'name':'Condition A', 'mu_group':2, 'mu_std':1},
                                                   {'name':'Condition B', 'mu_group':3, 'mu_std':1}],
                                                 sample_rate=sample_rate, n_subjects=20, run_duration=30, n_trials=5)

df = []

for (subj_idx, run), d in data.groupby(['subj_idx', 'run']):
    
    fytter = response_fytter.ResponseFytter(d.signal.values, sample_rate)
    
    fytter.add_event('Condition A',
                     onset_times=onsets.loc[subj_idx, run, 'Condition A'].onset,
                     basis_set='fourier',
                     n_regressors=8,
                     interval=[0, 20])
    
    fytter.add_event('Condition B',
                     basis_set='fourier',
                     n_regressors=8,
                     onset_times=onsets.loc[subj_idx, run, 'Condition B'].onset,
                     interval=[0, 20])    
    
    fytter.regress()
    
    df.append(fytter.get_timecourses())
    df[-1]['subj_idx'] = subj_idx
    df[-1]['run'] = run
    
df = pd.concat(df)

hfit = HierarchicalBayesianModel()

df = []

for (subj_idx, run), d in data.reset_index().groupby(['subj_idx', 'run']):
    fytter = response_fytter.ResponseFytter(d.signal.values.astype(np.float32), 1)
    
    fytter.add_event('Condition A',
                     onset_times=onsets.loc[subj_idx, run, 'Condition A'].onset,
                     interval=[-1, 20])
    
    fytter.add_event('Condition B',
                     onset_times=onsets.loc[subj_idx, run, 'Condition B'].onset,
                     interval=[-1, 20])    
    
    fytter.regress()
    
    hfit.add_run(fytter, subj_idx)

hfit.build_model()
results = hfit.sample(5000, chains=4, njobs=4)#, init='advi')

import pickle as pkl

with open('model.pkl', 'wb') as file:
    data = {}
    data['model'] = hfit.model
    data['results'] = hfit.results
    pkl.dump(data, file)
