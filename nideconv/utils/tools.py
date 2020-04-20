import numpy as np
from scipy import signal
import pandas as pd

def get_proper_interval(interval, sample_duration):
    interval = np.array(interval)
    old_duration = interval[1] - interval[0]
    new_interval_duration = np.floor(old_duration / sample_duration)  * sample_duration
    return np.array([interval[0], interval[0] + new_interval_duration])

def resample_and_zscore(s, old_samplerate=1000, target_samplerate=20):
    
    if old_samplerate / target_samplerate < 13:
        downsample_factor = int(old_samplerate / target_samplerate)
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
                           kernel,
                           signal_samplerate,
                           interval=(0, 20),
                           oversample=20,
                           *args, **kwargs):
    
    if (kernel == 'double_hrf') or (kernel == 'double_gamma'):
        kernel = double_gamma_with_d
    elif kernel == 'gamma':
        kernel = gamma
    
    new_sample_rate = oversample * signal_samplerate
    
    duration = interval[1] - interval[0]
    
    t = np.linspace(*interval, 
                    num=int(new_sample_rate*duration),
                    endpoint=False)
    
    f = kernel(t, *args, **kwargs)
    
    output_signal = signal.decimate(signal.convolve(np.repeat(input, oversample) / oversample, f, 'full')[:input.shape[0]*oversample], oversample)
    
    return output_signal

    
def double_gamma_with_d(x, a1=6, a2=12, b1=0.9, b2=0.9, c=0.35, d1=5.4, d2=10.8):    
    y = (x/(d1))**a1 * np.exp(-(x-d1)/b1) - c*(x/(d2))**a2 * np.exp(-(x-d2)/b2)
    y[x < 0] = 0
    y /= y.max()
    return y

def gamma(x, a1=6, b1=0.9, d1=5.4):
    y = (x/(d1))**a1 * np.exp(-(x-d1)/b1)
    y[x < 0] = 0
    y /= y.max()
    return y

def double_gamma_with_d_time_derivative(x,
                                    a1=6,
                                    a2=12,
                                    b1=0.9,
                                    b2=0.9,
                                    c=0.35,
                                    d1=5.4,
                                    d2=10.8,
                                    dt=0.1):

    dhrf = 1. / dt * (double_gamma_with_d(x + dt, a1, a2, b1, b2, c, d1, d2) -
                      double_gamma_with_d(x, a1, a2, b1, b2, c, d1, d2))
    return dhrf


def _get_peaks(col, cutoff=1.0):
    
    peaks, _ = signal.find_peaks(col)
    prominence, _, _ = signal.peak_prominences(col, peaks)
    
    r = pd.DataFrame({'prominence':prominence,
                      'time peak':col.index.get_level_values(level='time')[peaks]},
                       index=[col.name]*len(peaks))
    
    r = r[r.prominence >= cutoff * r.prominence.max()].sort_values('prominence', ascending=False)
    if len(r) == 0:
        return pd.DataFrame({'prominence':[np.nan],
                             'time peak':[np.nan]},
                            index=[col.name])

    return r


def get_time_to_peak_from_timecourse(tc, cutoff=1., negative_peak=False):

    if tc.ndim == 1:
        return _get_peaks(tc).T

    else:
        peaks = tc.T.apply(_get_peaks, axis=1)
        return pd.concat(peaks.tolist()).T


def get_ss(timeseries):
    return ((timeseries - timeseries.mean(0))**2).sum(0)
