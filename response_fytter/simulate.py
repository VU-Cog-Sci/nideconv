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
                             kernel='double_hrf'):
    """
    This function simulates an fMRI experiment. 
    
    The conditions-variable is a list of dictionaries, each including a mu_group and mu_std-field
    to indicate the mean impulse height, as well as the standard deviation across subjects.
    It also includes a n_trials-field to simulate the number of trials for that condition and
    potentially a 'name'-field to label the condition.
    The sd of the noise in the signal is always unity.
    """
    
    data = []

    if conditions is None:
        conditions = [{'name':'A',
                       'mu_group':1,
                       'std_group':0},
                       {'name':'B',
                       'mu_group':2,
                       'std_group':0}]
    
    sample_rate = 1./TR
    
    frametimes = np.arange(0, run_duration, TR)
    all_onsets = []
    
    parameters = []
    for subj_idx in np.arange(1, n_subjects+1):    
        
        for condition in conditions:
            amplitude = sp.stats.norm(loc=condition['mu_group'], scale=condition['std_group']).rvs()
            parameters.append({'subj_idx':subj_idx,
                               'trial_type':condition['name'],
                               'amplitude':amplitude})    
            
    parameters = pd.DataFrame(parameters).set_index(['subj_idx', 'trial_type'])
    
    for subj_idx in np.arange(1, n_subjects+1):    
        
        for run in range(1, n_runs+1):
            
            signals = np.zeros((len(conditions), len(frametimes)))

            for i, condition in enumerate(conditions):
                if 'name' in condition:
                    name = condition['name']
                else:
                    name = 'Condition %d' % (i+1)
                    


                isis = np.random.gamma(run_duration / n_trials, 1, size=n_trials * 10)
                onsets = np.cumsum(isis)
                onsets = np.random.choice(onsets[onsets < run_duration], 
                                          n_trials,
                                          replace=False)

                signals[i, (onsets / TR).astype(int)] = parameters.loc[subj_idx, name]
                
                
                all_onsets.append(pd.DataFrame({'onset':onsets}))
                all_onsets[-1]['subj_idx'] = subj_idx
                all_onsets[-1]['run'] = run
                all_onsets[-1]['trial_type'] = name
                
                
            signal = signals.sum(0)
            signal = convolve_with_function(signal, kernel, sample_rate)
            signal += np.random.randn(len(signal))
            
            tmp = pd.DataFrame({'signal':signal})
            tmp['t'] = frametimes
            tmp['subj_idx'], tmp['run'] = subj_idx, run
            
            
                
            data.append(tmp)
            
    data = pd.concat(data).set_index(['subj_idx', 'run', 't'])
    
    onsets = pd.concat(all_onsets).set_index(['subj_idx', 'run', 'trial_type'])
    
    return data, onsets, parameters

