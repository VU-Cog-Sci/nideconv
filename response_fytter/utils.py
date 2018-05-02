import numpy as np
from scipy import signal

def get_proper_interval(interval, sample_duration):
    interval = np.array(interval)
    old_duration = interval[1] - interval[0]
    new_interval_duration = np.floor(old_duration / sample_duration)  * sample_duration
    return np.array([interval[0], interval[0] + new_interval_duration])


def convolve_with_function(input,
                           function,
                           signal_samplerate,
                           interval=(0, 20),
                           oversample=20,
                           *args, **kwargs):
    
    if function == 'double_hrf':
        function = double_gamma_with_d
    
    new_sample_rate = oversample * signal_samplerate
    
    duration = interval[1] - interval[0]
    
    t = np.linspace(*interval, 
                    new_sample_rate*duration)
    
    f = function(t, *args, **kwargs)
    
    output_signal = signal.decimate(signal.convolve(np.repeat(input, oversample) / oversample, f, 'full')[:input.shape[0]*oversample], oversample)
    
    return output_signal

    
def double_gamma_with_d(x, a1=6, a2=12, b1=0.9, b2=0.9, c=0.35, d1=5.4, d2=10.8):    
    y = (x/(d1))**a1 * np.exp(-(x-d1)/b1) - c*(x/(d2))**a2 * np.exp(-(x-d2)/b2)
    y[x < 0] = 0
    y /= y.max()
    return y
