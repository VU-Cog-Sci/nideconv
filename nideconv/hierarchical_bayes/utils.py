import pymc3 as pm
import pandas as pd
import numpy as np

def get_hpd(timecourse_traces, melted=None, alpha=0.05, ):
    
    if melted is None:
        melted = timecourse_traces.columns.names[-1] != 't'
    
    if melted:
        extra_columns = [c for c in timecourse_traces.columns if c not in ['sample', 't', 'value']]
        assert((timecourse_traces.groupby(extra_columns + ['sample', 't', 'value']).size() == 1).all())
        timecourse_traces = timecourse_traces.pivot_table(index='sample', 
                                                        columns=extra_columns + ['t'], 
                                                        values='value')

    hpd = pm.stats.hpd(timecourse_traces.values, alpha=alpha)
    hpd = pd.DataFrame(hpd, columns=[alpha, 1-alpha], 
                       index=timecourse_traces.columns)
    return hpd

def do_ols(matrix):
    
    betas, ssquares, rank, _ = \
                                np.linalg.lstsq(matrix.iloc[:, 1:], 
                                                matrix.iloc[:, :1], 
                                                rcond=None)
    
    return pd.DataFrame(betas, index=matrix.columns[1:], columns=['beta'])
