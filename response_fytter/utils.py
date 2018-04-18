import numpy as np
import scipy as sp

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
