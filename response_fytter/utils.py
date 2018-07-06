import numpy as np
from scipy import signal

def get_proper_interval(interval, sample_duration):
    interval = np.array(interval)
    old_duration = interval[1] - interval[0]
    new_interval_duration = np.floor(old_duration / sample_duration)  * sample_duration
    return np.array([interval[0], interval[0] + new_interval_duration])

def resample_and_zscore(s, old_samplerate=1000, target_samplerate=20):
    
    if old_samplerate / target_samplerate < 13:
        downsample_factor = int(old_samplerate / target_sample_rate)
        s = sp.signal.decimate(s.values.ravel(), downsample_factor)
        new_samplerate = old_samplerate / downsample_factor
    else:
        downsample_factor = int(np.sqrt(old_samplerate / target_samplerate))
        s = sp.signal.decimate(s.values.ravel(), downsample_factor)
        s = sp.signal.decimate(s, downsample_factor)
        
        new_samplerate = old_samplerate / downsample_factor**2
        
    s -= s.mean()
    s /= s.std()
    return s, new_samplerate

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
                    num=new_sample_rate*duration)
    
    f = function(t, *args, **kwargs)
    
    output_signal = signal.decimate(signal.convolve(np.repeat(input, oversample) / oversample, f, 'full')[:input.shape[0]*oversample], oversample)
    
    return output_signal

    
def double_gamma_with_d(x, a1=6, a2=12, b1=0.9, b2=0.9, c=0.35, d1=5.4, d2=10.8):    
    y = (x/(d1))**a1 * np.exp(-(x-d1)/b1) - c*(x/(d2))**a2 * np.exp(-(x-d2)/b2)
    y[x < 0] = 0
    y /= y.max()
    return y
