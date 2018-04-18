import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal


def simulate_fmri_experiment(conditions=None,
                             TR=1.,
                             n_subjects=1, 
                             n_runs=1, 
                             n_trials=40, 
                             run_duration=300,
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
    base_n_trials = n_trials
    
    frametimes = np.arange(0, run_duration, TR)
    all_onsets = []
    
    parameters = []
    for subj_idx in np.arange(1, n_subjects+1):    
        
        for condition in conditions:
            amplitude = sp.stats.norm(loc=condition['mu_group'], scale=condition['std_group']).rvs()
            parameters.append({'subj_idx':subj_idx,
                               'condition':condition['name'],
                               'amplitude':amplitude})    
            
    parameters = pd.DataFrame(parameters).set_index(['subj_idx', 'condition'])
    
    for subj_idx in np.arange(1, n_subjects+1):    
        
        for run in range(1, n_runs+1):
            
            signals = np.zeros((len(conditions), len(frametimes)))

            for i, condition in enumerate(conditions):
                if 'name' in condition:
                    name = condition['name']
                else:
                    name = 'Condition %d' % (i+1)


                if 'n_trials' in condition:
                    n_trials = condition['n_trials']
                else:
                    n_trials = base_n_trials
                    
                
                if type(n_trials) in [tuple, list]:
                    n_trials_ = np.random.randint(*n_trials)
                else:
                    n_trials_ = n_trials


                if n_trials_ > 0:
                    isis = np.random.gamma(run_duration / n_trials_, 1, size=n_trials_ * 10)
                    onsets = np.cumsum(isis)

                    while(np.sum(onsets < run_duration) < n_trials_):
                        isis = np.random.gamma(run_duration / n_trials_, 1, size=n_trials_ * 10)
                        onsets = np.cumsum(isis)

                    onsets = np.random.choice(onsets[onsets < run_duration], n_trials_)

                    signals[i, (onsets / TR).astype(int)] = parameters.loc[subj_idx, name]
                    
                    
                    all_onsets.append(pd.DataFrame({'onset':onsets}))
                    all_onsets[-1]['subj_idx'] = subj_idx
                    all_onsets[-1]['run'] = run
                    all_onsets[-1]['condition'] = name
                

            signal = signals.sum(0)
            signal = convolve_with_function(signal, kernel, sample_rate)
            signal += np.random.randn(len(signal))
            
            tmp = pd.DataFrame({'signal':signal})
            tmp['t'] = frametimes
            tmp['subj_idx'], tmp['run'] = subj_idx, run
                
            data.append(tmp)
            
    data = pd.concat(data).set_index(['subj_idx', 'run', 't'])
    
    onsets = pd.concat(all_onsets).set_index(['subj_idx', 'run', 'condition'])
    
    return data, onsets, parameters

def convolve_with_function(input, function, signal_samplerate, interval=(0, 20), *args, **kwargs):
    
    if function == 'double_hrf':
        function = double_gamma_with_d
    
    new_sample_rate = 20*signal_samplerate
    
    duration = interval[1] - interval[0]
    
    t = np.linspace(*interval, new_sample_rate*duration)
    
    f = function(t, *args, **kwargs)
    
    output_signal = signal.decimate(signal.convolve(np.repeat(input, 20) / 20, f, 'full')[:input.shape[0]*20], 20)
    
    return output_signal

    
def double_gamma_with_d(x, a1=6, a2=12, b1=0.9, b2=0.9, c=0.35, d1=5.4, d2=10.8):    
    y = (x/(d1))**a1 * np.exp(-(x-d1)/b1) - c*(x/(d2))**a2 * np.exp(-(x-d2)/b2)
    y /= y.max()
    return y


