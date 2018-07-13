
"""
Extract ROI-timeseries from fmriprep data
=========================================
The first step in most deconvolution analyes is the extraction of the signal
from different regions-of-interest.

Nideconv contains an easy-to-use module for this that is leverarging
the functionality of `nilearn <http://nilearn.github.io/>`_.

Extracting a time series from a single functional run
-----------------------------------------------------
Let's try to extract some signals from a single functional run 
from a import *Stroop* dataset we got from a `open data repository
<https://openneuro.org/datasets/ds000164/versions/00001>`_ on
Openneuro.
The data has been preprocessed using the 
`fmriprep <http://fmriprep.readthedocs.io/en/latest/>`_ package.
"""

# Libraries
from nilearn import datasets
from nideconv.utils import roi
import pandas as pd
from nilearn import plotting

# Locate the data
func = '/data/openfmri/ds000164/derivatives/fmriprep/sub-001/func/sub-001_task-stroop_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
# ... and confounds extracted by fmriprep
confounds_fn = '/data/openfmri/ds000164/derivatives/fmriprep/sub-001/func/sub-001_task-stroop_bold_confounds.tsv'
# We need to load the confounds and fill nas
confounds = pd.read_table(confounds_fn).fillna(method='bfill')

# We only want to include a subset of confounds
confounds_to_include = ['FramewiseDisplacement', 'aCompCor00',
                        'aCompCor01', 'aCompCor02', 'aCompCor03',
                        'aCompCor04', 'aCompCor05', 'X', 'Y', 'Z',
                        'RotX', 'RotY', 'RotZ']
confounds = confounds[confounds_to_include]


# Use the cortical Harvard-Oxford atlas
atlas = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
plotting.plot_prob_atlas(atlas.maps)

ts = roi.extract_timecourse_from_nii(atlas,
                                     func,
                                     confounds=confounds.values, 
                                     t_r=1.5,
                                     high_pass=1./128,
                                     )

ts.head()

##############################################################################
#
# .. rst-class:: sphx-glr-script-out
# 
#  Out:
# 
#  .. code-block:: none
# 
#    Some output from Python
