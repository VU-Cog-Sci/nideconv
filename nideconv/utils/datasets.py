import os
import pandas as pd

_package_directory = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(_package_directory, "data")

def get_timeseries_stroop(atlas='pauli'):
    
    if atlas == 'pauli':
        fn = os.path.join(data_path, 'pauli.pkl')
    
    elif atlas == 'harvard_oxford':
        fn = os.path.join(data_path, 'harvard_oxford.pkl')
    else:
        raise ValueError('{} is not a valid atlas'.format(atlas)+\
                         ' for this datase')
    return pd.read_pickle(fn)
