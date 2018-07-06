"""
What is Deconvolution?
======================
Neuroscientists (amongst others) are often interested in time series that are derived
from neural activity, such as fMRI BOLD and pupil dilation. However, for some classes
of data (notably, pupil dilation and fMRI BOLD), neural activity gets temporally delayed and 
dispersed. This means that if the time series is related to some behavioral events that 
are close together in time, these event-related responses will contaminate each other.

Let's start with an example. With response_fytter we can make a simulated fMRI dataset
with an event at both 5 and 8 seconds.
"""

##############################################################################
# This code block is executed, although it produces no output. Lines starting
# with a simple hash are code comment and get treated as part of the code
# block. To include this new comment string we started the new block with a
# long line of hashes.
from response_fytter import simulate
import matplotlib.pyplot as plt
data, onsets, parameters = simulate.simulate_fmri_experiment()

print(data)

plt.plot([0,1], [0,1])
##############################################################################
