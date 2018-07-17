import pandas as pd
from bids.grabbids import BIDSLayout

from .roi import extract_timecourse_from_nii

def get_fmriprep_timeseries(fmriprep_folder,
                            sourcedata_folder,
                            atlas,
                            atlas_type=None,
                            low_pass=None,
                            high_pass=1./128,
                            confounds_to_include=None,
                            *args,
                            **kwargs):

    """
    Extract time series for each subject, task and run in a preprocessed
    dataset in BIDS format, given all the ROIs in `atlas`.

    Currently only `fmriprep` outputs are supported. The `sourcedata_folder`
    is necessary to look up the TRs of the functional runs.

    Parameters
    ----------

    fmriprep_folder: string  
        Path to the folder that contains fmriprep'ed functional MRI data.

    sourcedata_folder: string
        Path to BIDS folder that has been used as input for fmriprep

    atlas: sklearn.datasets.base.Bunch  
        This Bunch should contain at least a `maps`-attribute
        containing a label (3D) or probabilistic atlas (4D),
        as well as an `label` attribute, with one label for
        every ROI in the atlas.
        The function automatically detects which of the two is
        provided. It extracts a (weighted) time course per ROI.
        In the case of the probabilistic atlas, the voxels are
        weighted by their probability (see also the Mappers in
        nilearn).

    atlas_type: str, optional
        Can be 'labels' or 'probabilistic'. A label atlas
        should be 3D and contains one unique number per ROI.
        A Probabilistic atlas contains as many volume as 
        ROIs.
        Usually, `atlas_type` can be detected automatically.

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    confounds_to_include: list of strings
        List of confounds that should be regressed out.
        By default a limited list of confounds is regressed out:
        Namely, FramewiseDisplacement, aCompCor00, aCompCor01, aCompCor02,
        aCompCor03, aCompCor04, aCompCor05, X, Y, Z, RotX, RotY, and RotZ 
    
    Examples
    --------
    >>> source_data = '/data/ds001/sourcedata'
    >>> fmriprep_data = '/data/ds001/derivatives/fmriprep'
    >>> from nilearn import datasets 
    >>> atlas = datasets.fetch_atlas_pauli_2017()
    >>> from nideconv.utils.roi import get_fmriprep_timeseries
    >>> ts = get_fmriprep_timeseries(fmriprep_data,
                                     source_data,
                                     atlas)
    >>> ts.head()
    roi                        Pu        Ca
    subject task   time                    
    001     stroop 0.0  -0.023651 -0.000767
                   1.5  -0.362429 -0.012455
                   3.0   0.087955 -0.062127
                   4.5  -0.099711  0.146744
                   6.0  -0.443499  0.093190
    

    """

    if confounds_to_include is None:
        confounds_to_include = ['FramewiseDisplacement', 'aCompCor00',
                                'aCompCor01', 'aCompCor02', 'aCompCor03',
                                'aCompCor04', 'aCompCor05', 'X', 'Y', 'Z',
                                'RotX', 'RotY', 'RotZ']

    index_keys = []
    timecourses = []

    for func, confounds, meta in _get_func_and_confounds(fmriprep_folder,
                                                         sourcedata_folder):

        print("Extracting signal from {}...".format(func.filename))
        confounds = pd.read_table(confounds.filename).fillna(method='bfill')

        tc = extract_timecourse_from_nii(atlas,
                                         func.filename,
                                         t_r=meta['RepetitionTime'],
                                         atlas_type=atlas_type,
                                         low_pass=low_pass,
                                         high_pass=high_pass,
                                         confounds=confounds[confounds_to_include].values)

        for key in ['subject', 'task', 'run', 'session']:
            if hasattr(func, key):
                tc[key] = getattr(func, key)

                if key not in index_keys:
                    index_keys.append(key)

        timecourses.append(tc)

    timecourses = pd.concat(timecourses)
    timecourses = timecourses.set_index(index_keys, append=True)
    timecourses = timecourses.reorder_levels(index_keys + ['time'])

    return timecourses

def get_bids_onsets(bids_folder):
    """
    Get event onsets from a BIDS folder in a nideconv-ready
    format.

    Parameters
    ----------
    bids_folder: str
        Folder containing fMRI dataset according to BIDS-standard.


    Returns
    -------
    onsets: DataFrame
        Dataframe containing onsets, with subject and potentially
        session, task and run as indices.

    """

    layout = BIDSLayout(bids_folder)
    
    events = layout.get(type='events', extensions='tsv')


    onsets =[]
    index_keys = []

    for event in events:

        onsets_ = pd.read_table(event.filename)

        for key in ['subject', 'run', 'task', 'session']:

            if hasattr(event, key):
                onsets_[key] = getattr(event, key)
                if key not in index_keys:
                    index_keys.append(key)


        onsets.append(onsets_)

    onsets = pd.concat(onsets).set_index(index_keys)
    
    return onsets


def _get_func_and_confounds(fmriprep_folder,
                            sourcedata_folder):

    fmriprep_layout = BIDSLayout(fmriprep_folder)
    sourcedata_layout = BIDSLayout(sourcedata_folder)

    files = fmriprep_layout.get(extensions=['.nii', 'nii.gz'],
                                modality='func', type='preproc')

    confounds = []
    metadata = []

    for f in files:
        kwargs = {}

        for key in ['subject', 'run', 'task', 'session']:
            if hasattr(f, key):
                kwargs[key] = getattr(f, key)

        c = fmriprep_layout.get(type='confounds', **kwargs)
        c = c[0]
        confounds.append(c)

        sourcedata_file = sourcedata_layout.get(
            modality='func', extensions='nii.gz', **kwargs)

        assert(len(sourcedata_file) == 1)
        md = sourcedata_layout.get_metadata(sourcedata_file[0].filename)
        metadata.append(md)

    return list(zip(files, confounds, metadata))
