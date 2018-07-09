"""
What is Deconvolution?
======================
Neuroscientists (amongst others) are often interested in time series that are derived
from neural activity, such as fMRI BOLD and pupil dilation. However, for some classes
of data (notably, pupil dilation and fMRI BOLD), neural activity gets temporally delayed and 
dispersed. This means that if the time series is related to some behavioral events that 
are close together in time, these event-related responses will contaminate each other.

"""

from response_fytter import simulate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook')

##############################################################################
# Simulate data
# -------------
# Here we simulate fMRI data with a "cue - stimulus" design.
# There are four cues and stimulus pairs.
# The cue is always followed by a stimulus in 1, 2, 3, or 4 seconds.
# The cue leads to a small de-activation (0.5 % signal change), the stimulus a
# slight activation (1.0 % signal change)

cue_onsets = [5, 15, 25, 35]
stim_onsets = [6, 17, 28, 39]

cue_pars = {'name':'cue',
            'mu_group':-.5,
            'std_group':0,
            'onsets':cue_onsets}

stim_pars = {'name':'stim',
             'mu_group':1,
             'std_group':0,
             'onsets':stim_onsets}

conditions = [cue_pars,
              stim_pars]

data, onsets, parameters = simulate.simulate_fmri_experiment(conditions,
                                                             run_duration=60,
                                                             noise_level=0.1)

##############################################################################
# Plot simulated data
# -------------------
data.plot()
sns.despine()

for onset in cue_onsets:
    l1 =plt.axvline(onset, c='r')

for onset in stim_onsets:
    l2 =plt.axvline(onset, c='g')

plt.legend([l1, l2], ['Cue', 'Stimulus'])
plt.gcf().set_size_inches(10, 4)

##############################################################################
# Underlying data-generating model
# --------------------------------
# Because we simulated the data, we know that the event-related responses should
# exactly follow the *canonical Hemodynamic Response Function* [1]_are
from response_fytter.utils import double_gamma_with_d
import numpy as np

plt.figure(figsize=(12, 4))

t = np.linspace(0, 20, 100)
ax1 = plt.subplot(121)
plt.title('Ground truth cue-related response')
plt.plot(t, double_gamma_with_d(t) * -.5,
         c=sns.color_palette()[0])
plt.xlabel('Time since event (s)')
plt.ylabel('Percent signal change')
plt.axhline(0, c='k', ls='--')

plt.subplot(122, sharey=ax1)
plt.title('Ground truth stimulus-related response')
plt.plot(t, double_gamma_with_d(t),
         c=sns.color_palette()[1])
plt.axhline(0, c='k', ls='--')
plt.xlabel('Time since event (s)')
plt.ylabel('Percent signal change')

sns.despine()

##############################################################################
# Naive approach: epoched averaging
# ---------------------------------
# A simple approach that is more appropriate for fast electrphysiological signals
# like EEG and MEG would be to select little chunks of the time series,
# corresponding to the onset of our events-of-interest and the first 20 seconds
# ("epoching").
#
# nideconv

#############################################################################
# References
# -----------
# .. [1] Glover et al., 1999
 
