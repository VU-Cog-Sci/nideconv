"""
Bayesian hierachical deconvolution of neural signals
====================================================
So far we have used _frequentist_ methods to estimate the GLMs
and deconvolve neural signals.

An alternative statistical paradigm is _Bayesian_ estimation.
For a solid, readable introduction into Bayesian statistics, please
see the `puppy book <http://www.indiana.edu/~kruschke/DoingBayesianDataAnalysis/>`_
by John K. Kruschke or
the `Bayesian Methods for Hackers<https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers>`_
-book.


With *nideconv*, you can estimate _Hierachical_ GLMs using the Bayesian 
framework. This is possible by using Markov Chain Monte Calro sampling using 
the NUTS (no U-turn sampler) implemented in `STAN <https://mc-stan.org/>`.

First we again simulate som data.
To make it interesting, we have a 'Correct' and 'Error'-condition.
Importantly, every run has only 1-6 Error trials
"""
from nideconv import simulate
conditions = [{'name':'Correct', 'mu_group':.25, 'std_group':.1, 'n_trials':16},
              {'name':'Error', 'mu_group':.5, 'std_group':.1, 'n_trials':(1,6)}]

data, onsets, pars = simulate.simulate_fmri_experiment(conditions,
                                                       n_subjects=9,
                                                       n_runs=1,
                                                       TR=1.5)

##############################################################################
# First, we fit this data using the traditional frequentist GLM with Fourier
# basis functions:
from nideconv import GroupResponseFitter
gmodel = GroupResponseFitter(data, onsets, input_sample_rate=1/1.5, concatenate_runs=False)
gmodel.add_event('Correct', basis_set='fourier', n_regressors=9, interval=[0, 21])
gmodel.add_event('Error', basis_set='fourier', n_regressors=9, interval=[0, 21])

##############################################################################
# We fit the model using ridge regression
gmodel.fit(type='ridge', alphas=[1.0])

##############################################################################
# The response we estimate for the group looks pretty good.
gmodel.plot_groupwise_timecourses()

##############################################################################
# However, the response estimates for different indivdiuals (solid lines)
# are quite off from the ground truth (dotted lines)


# Now we plot for every subject the estimated HRF
fac = gmodel.plot_subject_timecourses(ci=95, col_wrap=3, size=10, legend=False, n_boot=100)

# ...and the underlying ground truth
import seaborn as sns
from nideconv.utils import convolve_with_function
import numpy as np

t = np.linspace(0, 21, 21*20)
hrf = np.zeros_like(t)
hrf[0] = 1
hrf = convolve_with_function(hrf, 'double_hrf', 20)

for subject, ax in enumerate(fac[0].axes.ravel()):
    subject += 1
    ax.plot(t, hrf * pars.loc[subject, 'Correct'].amplitude, ls='--', lw=1.5, c=sns.color_palette()[0])
    ax.plot(t, hrf * pars.loc[subject, 'Error'].amplitude, ls='--', lw=1.5, c=sns.color_palette()[1])


##############################################################################
# We convert the `GroupResponseModel` to a `HierarchicalBayesianModel`
# and estimate posterior distributions by MCMC sampling:

from nideconv import HierarchicalBayesianModel
model = HierarchicalBayesianModel.from_groupresponsefitter(gmodel)
model.build_model()
model.sample()


##############################################################################
# Plot the individual subject time courses and their Bayesian credible interval
# (CI)
fac = model.plot_subject_timecourses(col_wrap=3, legend=False)

# plot ground truth
for subject, ax in enumerate(fac.axes.ravel()):
    subject += 1 
    ax.plot(t, hrf*pars.loc[subject, 'Correct'].amplitude, ls='--', lw=1.5, color=sns.color_palette()[0])
    ax.plot(t, hrf*pars.loc[subject, 'Error'].amplitude, ls='--', lw=1.5, color=sns.color_palette()[1])
