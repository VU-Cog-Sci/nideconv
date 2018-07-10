"""
What are basis functions and how do I use them?
===============================================
In the previous tutorial we have seen that overlapping even-related time courses
can be recovered using the general linear model, as long as we assume they add up
linearly and are time-invariant.

Another important assumption pertains to what we believe the response we are 
interested in looks like. In the most extreme case, we for example assume that a
task-related BOLD fMRI response exactly follows the canonical HRF.
"""
from nideconv.utils import double_gamma_with_d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
sns.set_context('notebook')

plt.figure(figsize=(6,3))
t = np.linspace(0, 20)
plt.plot(t, double_gamma_with_d(t))
plt.title('Canonical HRF')
sns.despine()


##############################################################################
# Well-specified model
# --------------------
# Let’s simulate some data with very little noise and the standard HRF, 
# fit a model, and see how well our model fits the data 

from nideconv import simulate
from nideconv import ResponseFitter

conditions = [{'name':'A',
              'mu_group':5,
              'std_group':1,
               'onsets':[0, 20]}]

# Simulate data with very short run time and TR for illustrative purposes
data, onsets, pars = simulate.simulate_fmri_experiment(conditions,
                                                       TR=0.2,
                                                       run_duration=40,
                                                       noise_level=.5,
                                                       n_rois=1)

# Make ResponseFitter-object to fit these data
rf = ResponseFitter(input_signal=data,
                    sample_rate=5) # Sample rate is inverse of TR (1/TR)
rf.add_event('A',
             onsets.loc['A'].onset,
             interval=[0, 20],
             basis_set='canonical_hrf')

rf.fit()

rf.plot_model_fit()

##############################################################################
# As you can see, the model fits the data well, the model is well-specified. 
# Now let’s simulate some data with different HRF from what the model assumes.

# For this HRFm the first peak is much earlier (3.5 seconds versus 5.8)
# and the second "dip" is not there (c=0).
kernel_pars = {'a1':3.5,
               'c':0}

plt.plot(t, double_gamma_with_d(t, **kernel_pars))
plt.title('Alternative HRF')
sns.despine()
plt.gcf().set_size_inches(10, 4)

##############################################################################
# Simulate data again
data, onsets, pars = simulate.simulate_fmri_experiment(conditions,
                                                       TR=0.2,
                                                       run_duration=40,
                                                       noise_level=.5,
                                                       n_rois=1,
                                                       kernel_pars=kernel_pars)

rf = ResponseFitter(input_signal=data,
                    sample_rate=5)
rf.add_event('A',
             onsets.loc['A'].onset,
             interval=[0, 20],
             basis_set='canonical_hrf')

##############################################################################
# Plot the model fit
rf.fit()
rf.plot_model_fit()


##############################################################################
# And the estimated time course
rf.plot_timecourses()
plt.suptitle('Estimated event-related response')

##############################################################################
# We can use `get_time_to_peak` to get an estimate of where the largest
# peak of this event-related time course comes to lie
print(rf.get_time_to_peak())

##############################################################################
# Clearly, the model now does not fit very well. This is because the design 
# matrix :math:`X` does not allow for properly modelling the event-related 
# time course. 
# No matter what linear combination :math:`\beta` you take of the intercept 
# and canonical HRF (that has been convolved with the event onsets), 
# the data can never be properly explained.

def plot_design_matrix(rf, legend=True):
    sns.set_style('whitegrid')
    ax = plt.subplot(121)
    plt.plot(rf.X, rf.X.index)
    plt.gca().invert_yaxis()
    plt.ylabel('time (s)')
    plt.xlabel('Height regressor')

    if legend:
        plt.legend(rf.X.columns.get_level_values('regressor'), loc='lower center')

    plt.title('Design matrix')

    plt.subplot(122, sharey=ax)
    plt.plot(rf.input_signal, rf.input_signal.index, c='k')
    plt.title('Measured signal')
    plt.xlabel('Measured signal')

plot_design_matrix(rf)

##############################################################################
# The derivative of the cHRF with respect to time
# -----------------------------------------------
# The solution to this mis-specification is to increase the model complexity 
# by adding extra regressors that increase the flexibility of the model. A 
# very standard approach in BOLD fMRI is to include the derivative of the 
# HRF with respect to time for dt=0.1. Then the design matrix looks like this
rf = ResponseFitter(input_signal=data,
                    sample_rate=5)
rf.add_event('A',
             onsets.loc['A'].onset,
             interval=[0, 20],
             basis_set='canonical_hrf_with_time_derivative') # note the more complex
                                                             # basis function set

plot_design_matrix(rf)

##############################################################################
# The GLM can now “use the new, red regressor” to somewhat shift the 
# original HRF earlier in time.
sns.set_style('white')
rf.fit()
rf.plot_model_fit()

##############################################################################
# 
rf.plot_timecourses()
plt.suptitle('Estimated event-related time course')
print(rf.get_time_to_peak())



##############################################################################
# Even more complex basis functions
# ---------------------------------
# Note that the model still does not fit perfectly: the Measured signal is 
# still peaking earlier in time than the model. And the model still assumes 
# as post-peak dip that is not there in the data.
#
# One solution is to use yet more complex basis functions, such as the 
# Finite Impulse Response functions we used in the previous tutorial. This 
# basis functions consists of one regressor per time-bin (as in time 
# offset since event).

rf = ResponseFitter(input_signal=data,
                    sample_rate=5)
rf.add_event('A',
             onsets.loc['A'].onset,
             interval=[0, 20],
             basis_set='fir',
             n_regressors=20) # One regressor per second

plot_design_matrix(rf, legend=False)


##############################################################################
# Clearly, this model is much more flexible, and, hence, it fits better:
rf.fit()
rf.plot_model_fit()

##############################################################################
# Clearly, this model is much more flexible, and, hence, it fits better:
rf.plot_timecourses()
print(rf.get_time_to_peak())
