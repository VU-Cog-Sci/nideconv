import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal


def simulate_fmri_experiment(event_types=None,
                             TR=1.,
                             n_subjects=1, 
                             n_runs=1, 
                             n_trials=40, 
                             run_duration=300,
                             kernel='double_hrf'):
    """
    This function simulates an fMRI experiment. 
    
    The event_types-variable is a list of dictionaries, each including a mu_group and mu_std-field
    to indicate the mean impulse height, as well as the standard deviation across subjects.
    It also includes a n_trials-field to simulate the number of trials for that event_type and
    potentially a 'name'-field to label the event_type.
    The sd of the noise in the signal is always unity.
    """
    
    data = []

    if event_types is None:
        event_types = [{'name':'A',
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
    for subject_id in np.arange(1, n_subjects+1):    
        
        for event_type in event_types:
            amplitude = sp.stats.norm(loc=event_type['mu_group'], scale=event_type['std_group']).rvs()
            parameters.append({'subject_id':subject_id,
                               'event_type':event_type['name'],
                               'amplitude':amplitude})    
            
    parameters = pd.DataFrame(parameters).set_index(['subject_id', 'event_type'])
    
    for subject_id in np.arange(1, n_subjects+1):    
        
        for run in range(1, n_runs+1):
            
            signals = np.zeros((len(event_types), len(frametimes)))

            for i, event_type in enumerate(event_types):
                if 'name' in event_type:
                    name = event_type['name']
                else:
                    name = 'Condition %d' % (i+1)


                if 'n_trials' in event_type:
                    n_trials = event_type['n_trials']
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

                    signals[i, (onsets / TR).astype(int)] = parameters.loc[subject_id, name]
                    
                    
                    all_onsets.append(pd.DataFrame({'onset':onsets}))
                    all_onsets[-1]['subject_id'] = subject_id
                    all_onsets[-1]['run'] = run
                    all_onsets[-1]['event_type'] = name
                

            signal = signals.sum(0)
            signal = convolve_with_function(signal, kernel, sample_rate)
            signal += np.random.randn(len(signal))
            
            tmp = pd.DataFrame({'signal':signal})
            tmp['t'] = frametimes
            tmp['subject_id'], tmp['run'] = subject_id, run
                
            data.append(tmp)
            
    data = pd.concat(data).set_index(['subject_id', 'run', 't'])
    
    onsets = pd.concat(all_onsets).set_index(['subject_id', 'run', 'event_type'])
    
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


