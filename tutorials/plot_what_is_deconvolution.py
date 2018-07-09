"""
What is Deconvolution?
======================
Neuroscientists (amongst others) are often interested in time series that are derived
from neural activity, such as fMRI BOLD and pupil dilation. However, for some classes
of data, neural activity gets temporally delayed and 
dispersed. This means that if the time series is related to some behavioral events that 
are close together in time, these event-related responses will contaminate each other.

"""

# Import libraries and set up plotting
import nideconv
from nideconv import simulate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook')
palette = sns.color_palette('Set1')

##############################################################################
# Simulate data
# -------------
# We simulate fMRI data with a "cue - stimulus" design.
# There are four cues and corresponding stimulus presentations.
# The cue is always followed by a stimulus, seperated in time by 
# 1, 2, 3, or 4 seconds.
# The cue leads to a small de-activation (0.5 % signal change), the stimulus to an
# activation (1.0 % signal change).

cue_onsets = [5, 15, 25, 35]
stim_onsets = [6, 17, 28, 39]

cue_pars = {'name':'cue',
            'mu_group':-.5, # Slight negative response for cue
            'std_group':0,
            'onsets':cue_onsets}

stim_pars = {'name':'stim',
             'mu_group':1, # Positive response for stimulus presentation
             'std_group':0,
             'onsets':stim_onsets}

conditions = [cue_pars,
              stim_pars]

data, onsets, parameters = simulate.simulate_fmri_experiment(conditions,
                                                             run_duration=60,
                                                             noise_level=0.05)

##############################################################################
# Underlying data-generating model
# --------------------------------
# Because we simulated the data, we know that the event-related responses should
# exactly follow the *canonical Hemodynamic Response Function* [1]_are
from nideconv.utils import double_gamma_with_d
import numpy as np

plt.figure(figsize=(8, 2.5))

t = np.linspace(0, 20, 100)
ax1 = plt.subplot(121)
plt.title('Ground truth cue-related response')
plt.plot(t, double_gamma_with_d(t) * -.5,
         color=palette[0])
plt.xlabel('Time since event (s)')
plt.ylabel('Percent signal change')
plt.axhline(0, c='k', ls='--')

plt.subplot(122, sharey=ax1)
plt.title('Ground truth stimulus-related response')
plt.plot(t, double_gamma_with_d(t),
         color=palette[1])
plt.axhline(0, c='k', ls='--')
plt.xlabel('Time since event (s)')
plt.ylabel('Percent signal change')

sns.despine()

##############################################################################
# Plot simulated data
# -------------------
data.loc[1, 1].plot(c='k')
sns.despine()

for onset in cue_onsets:
    l1 =plt.axvline(onset, c=palette[0], ymin=.25, ymax=.75)

for onset in stim_onsets:
    l2 =plt.axvline(onset, c=palette[1], ymin=.25, ymax=.75)

plt.legend([l1, l2], ['Cue', 'Stimulus'])
plt.gcf().set_size_inches(10, 4)

##############################################################################
# Naive approach: epoched averaging
# ---------------------------------
# A simple approach that is appropriate for fast electrphysiological signals
# like EEG and MEG, but not necessarily fMRI, would be to select little chunks of the 
# time series, corresponding to the onset of the events-of-interest and the subsequent
# 20 seconds of signal ("epoching").

##############################################################################
# We can do such a epoch-analysis using nideconv, by making a ResponseFitter
# object and using the `get_epochs()`-function:

rf =nideconv.ResponseFitter(input_signal=data,
                            sample_rate=1)

# Get all the epochs corresponding to cue-onsets for subject 1,
# run 1.
cue_epochs = rf.get_epochs(onsets=onsets.loc[1, 1, 'cue'].onset, 
                           interval=[0, 20])

#############################################################################
# Now we have a 4 x 21 DataFrame of epochs, that we can all plot 
# in the same figure:
print(cue_epochs)
cue_epochs['area 1'].T.plot(c=palette[0], alpha=.5, ls='--', legend=False)
cue_epochs['area 1'].mean().plot(c=palette[0], lw=2, alpha=1.0)
sns.despine()
plt.xlabel('Time (s)')
plt.title('Epoched average of cue-locked response')
plt.gcf().set_size_inches(8, 3)

#############################################################################
# We can do the same for the stimulus-locked responses:
stim_epochs = rf.get_epochs(onsets=onsets.loc[1, 1, 'stim'].onset, 
                           interval=[0, 20])
stim_epochs['area 1'].T.plot(c=palette[1], alpha=.5, ls='--', legend=False)
stim_epochs['area 1'].mean().plot(c=palette[1], lw=2, alpha=1.0)
sns.despine()
plt.xlabel('Time (s)')
plt.title('Epoched average of stimulus-locked response')
plt.gcf().set_size_inches(8, 3)


#############################################################################
# Contamination
# ~~~~~~~~~~~~~
# As you can see, when we use epoched averaging, both the cue- and 
# stimulus-related response are *contaminated*  by adjacent responses (from 
# both response types). The result is some sinewave-like pattern that
# has little to do with the data-generating response functions of
# both cue- and stimulus-related activity

# This is because the event-related responses are overlapping in time: 
from nideconv.utils import double_gamma_with_d

t = np.linspace(0, 30)

cue1_response = double_gamma_with_d(t) * -.5
stim1_response = double_gamma_with_d(t-1)

cue2_response = double_gamma_with_d(t-10) * -.5
stim2_response = double_gamma_with_d(t-12)

palette2 = sns.color_palette('Set2')

plt.fill_between([0, 20],  -.5, -.6, color=palette2[0])
plt.plot([[0, 20], [0, 20]], [-.5, 0], color=palette2[0])
plt.fill_between([1, 21],  -.6, -.7, color=palette2[1])
plt.plot([[1, 21], [1, 21]], [-.6, 0], color=palette2[1])
plt.fill_between([10, 30],  -.7, -.8, color=palette2[2])
plt.plot([[10, 30], [10, 30]], [-.7, 0], color=palette2[2])
plt.fill_between([12, 32],  -.8, -.9, color=palette2[3])
plt.plot([[12, 32], [12, 32]], [-.8, 0], color=palette2[3])
            

plt.plot(t, cue1_response, c=palette2[0], ls='--', label='Cue 1-related activity')
plt.plot(t, stim1_response, c=palette2[1], ls='--', label='Stimulus 1-related activity')

plt.plot(t, cue2_response, c=palette2[2], ls='--', label='Cue 2-related activity')
plt.plot(t, stim2_response, c=palette2[3], ls='--', label='Stimulus 2-related activity')

plt.plot(t, cue1_response + \
            stim1_response + \
            cue2_response + \
            stim2_response, 
         c='k', label='Combined activity')
plt.legend()
sns.despine()
plt.gcf().set_size_inches(8, 4)
plt.xlim(-1, 32.5)
plt.title('Illustration of overlap problem')


#############################################################################
# Solution: the Genera Linear Model
# ---------------------------------
# An often-used solution to the "overlap problem" is to assume a 
# `linear time-invariant system 
# <https://en.wikipedia.org/wiki/Linear_time-invariant_theory>`_. 
# This means that you assume that overlapping responses influencing time point
# :math:`t` add up linearly.
# Assuming this linearity, the deconvolution boils down to solving a linear
# sytem: every timepoint :math:`y_t` from signal :math:`Y` is a linear 
# combination of the overlapping responses. These responses are modeled
# by a set of basis functions in matrix :math:`X`.
# We just need to find the 'weights' of the responses :math:`\beta`.:
# 
# .. math:: Y = X\beta
#
# We can do this using a General Linear Model (GLM) and its closed-form solution
# ordinary least-squares (OLS).
#
# .. math:: \hat{\beta} = (X^TX)^{-1} X^TY
#
# This solution is part of the main functionality of nideconv. 
# and can be applied by creating a `ResponseFitter`-object:
rf =nideconv.ResponseFitter(input_signal=data,
                            sample_rate=1)

#############################################################################
# To which the events-of-interest can be added as follows:

rf.add_event(event_name='cue', 
             onsets=onsets.loc[1, 1, 'cue'].onset,
             interval=[0,20])
rf.add_event(event_name='stimulus', 
             onsets=onsets.loc[1, 1, 'stim'].onset,
             interval=[0,20])

#############################################################################
# Nideconv aumatically creates a design matrix.
# By default, it does so using 'Finite Impulse
# Response'-regressors (FIR). Each one of these regressors corresponds
# to a different event and temporal offset. Such a  design matrix looks like this:

sns.heatmap(rf.X)
print(rf.X)

#############################################################################
# (Note the hierarchical columns (event type / covariate / regressor) 
# on the regressors!)

#############################################################################
# Now we can solve this linear system using ordinary least squares:
rf.fit()
print(rf.betas)

#############################################################################
# Importantly, with nideconv it is also very easy to 'convert` these 
# beta-estimates to the found event-related time courses, at a higher temporal
# resolution:
tc =rf.get_timecourses()
print(tc)

#############################################################################
#  as well as plot these responses...
sns.set_palette(palette)
rf.plot_timecourses()
plt.suptitle('Linear deconvolution using GLM and FIR')
plt.title('')
plt.legend()

#############################################################################
# As you can see, these estimated responses are much closer to the 
# original data-generating functions we were looking for.
#
# Cleary, the linear deconvolution approach allows us to very quickly
# and effectively 'decontaminate' overlapping responses.
# Have a look at the next section () for more theory and plots
# on the selection of appropriate `basis functions`.


#############################################################################
# References
# -----------
# .. [1] Glover, G. H. (1999). Deconvolution of impulse response in 
# event-related BOLD fMRI. NeuroImage, 9(4), 416–429.
