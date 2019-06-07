import pandas as pd
import numpy as np

def make_indices(dimensions):
    # Generates complete set of indices for given dimensions
    level = len(dimensions)
    if level == 1:
        return list(range(dimensions[0]))
    indices = [[]]
    while level:
        _indices = []
        for j in range(dimensions[level - 1]):
            _indices += [[j] + i for i in indices]
        indices = _indices
        level -= 1
    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width
    Assumes that x is sorted numpy array.
    """
    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max

def get_hpd_(x, alpha=0.05, transform=lambda x: x):
    """Calculate highest posterior density (HPD) of array for given alpha. The HPD is the
    minimum width Bayesian credible interval (BCI).
    This function assumes the posterior distribution is unimodal:
    it always returns one interval per variable.
    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      alpha : float
          Desired probability of type I error (defaults to 0.05)
      transform : callable
          Function to transform data (defaults to identity)
    """
    # Make a copy of trace
    x = transform(x.copy())

    # For multivariate node
    if x.ndim > 1:

        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
        dims = np.shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1] + (2,))

        for index in make_indices(dims[:-1]):

            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])

            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)

        # Transpose back before returning
        return np.array(intervals)

    else:
        # Sort univariate node
        sx = np.sort(x)

        return np.array(calc_min_interval(sx, alpha))

def get_hpd(timecourse_traces, melted=None, alpha=0.05, ):
    
    if melted is None:
        melted = timecourse_traces.columns.names[-1] != 't'
    
    if melted:
        extra_columns = [c for c in timecourse_traces.columns if c not in ['sample', 't', 'value']]
        assert((timecourse_traces.groupby(extra_columns + ['sample', 't', 'value']).size() == 1).all())
        timecourse_traces = timecourse_traces.pivot_table(index='sample', 
                                                        columns=extra_columns + ['t'], 
                                                        values='value')

    hpd = get_hpd_(timecourse_traces.values, alpha=alpha)
    hpd = pd.DataFrame(hpd, columns=[alpha, 1-alpha],
                       index=timecourse_traces.columns)
    return hpd

def do_ols(matrix):
    
    betas, ssquares, rank, _ = \
                                np.linalg.lstsq(matrix.iloc[:, 1:], 
                                                matrix.iloc[:, :1], 
                                                rcond=None)
    
    return pd.DataFrame(betas, index=matrix.columns[1:], columns=['beta'])
