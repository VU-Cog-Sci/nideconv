import pandas as pd
from nilearn import input_data
import nibabel as nb
from nilearn._utils import check_niimg
from nilearn import image
import numpy as np

def extract_timecourse_from_nii(atlas,
                                nii,
                                mask=None,
                                confounds=None,
                                atlas_type=None,
                                t_r=None,
                                low_pass=None,
                                high_pass=1./128,
                                *args,
                                **kwargs):
    """
    Extract time courses from a 4D `nii`, one for each label 
    or map in `atlas`,

    This method extracts a set of time series from a 4D nifti file
    (usually BOLD fMRI), corresponding to the ROIs in `atlas`.
    It also performs some minimal preprocessing using 
    `nilearn.signal.clean`.
    It is especially convenient when using atlases from the
    `nilearn.datasets`-module.

    Parameters
    ----------

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

    nii: 4D niimg-like object
        This NiftiImage contains the time series that need to
        be extracted using `atlas`

    mask: 3D niimg-like object
        Before time series are extracted, this mask is applied,
        can be useful if you want to exclude non-gray matter.

    confounds: CSV file or array-like, optional
        This parameter is passed to nilearn.signal.clean. Please 
        see the related documentation for details.
        shape: (number of scans, number of confounds)

    atlas_type: str, optional
        Can be 'labels' or 'probabilistic'. A label atlas
        should be 3D and contains one unique number per ROI.
        A Probabilistic atlas contains as many volume as 
        ROIs.
        Usually, `atlas_type` can be detected automatically.

    t_r, float, optional
        Repetition time of `nii`. Can be important for
        temporal filtering.

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    Examples
    --------

    >>> from nilearn import datasets
    >>> data = '/data/ds001/derivatives/fmriprep/sub-01/func/sub-01_task-checkerboard_bold.nii.gz'
    >>> atlas = datasets.fetch_atlas_pauli_2017()
    >>> ts = extract_timecourse_from_nii(atlas,
                                         data,
                                         t_r=1.5)
    >>> ts.head()

    """

    standardize = kwargs.pop('standardize', False)
    detrend = kwargs.pop('detrend', False)

    if atlas_type is None:
        maps = check_niimg(atlas.maps)

        if len(maps.shape) == 3:
            atlas_type = 'labels'
        else:
            atlas_type = 'prob'

    if atlas_type == 'labels':
        masker = input_data.NiftiLabelsMasker(atlas.maps,
                                              mask_img=mask,
                                              standardize=standardize,
                                              detrend=detrend,
                                              t_r=t_r,
                                              low_pass=low_pass,
                                              high_pass=high_pass,
                                              *args, **kwargs)
    else:
        masker = input_data.NiftiMapsMasker(atlas.maps,
                                            mask_img=mask,
                                            standardize=standardize,
                                            detrend=detrend,
                                            t_r=t_r,
                                            low_pass=low_pass,
                                            high_pass=high_pass,
                                            *args, **kwargs)

    data = _make_psc(nii)

    results = masker.fit_transform(data,
                                   confounds=confounds)

    # For weird atlases that have a label for the background
    if len(atlas.labels) == results.shape[1] + 1:
        atlas.labels = atlas.labels[1:]

    if t_r is None:
        t_r = 1

    index = pd.Index(np.arange(0,
                               t_r*data.shape[-1],
                               t_r),
                     name='time')

    columns = pd.Index(atlas.labels,
                       name='roi')

    return pd.DataFrame(results,
                        index=index,
                        columns=columns)


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

def _make_psc(data):
    mean_img = image.mean_img(data)

    # Replace 0s for numerical reasons
    mean_data = mean_img.get_data()
    mean_data[mean_data == 0] = 1
    denom = image.new_img_like(mean_img, mean_data)

    return image.math_img('data / denom[..., np.newaxis] * 100 - 100',
                          data=data, denom=denom)


def _get_func_and_confounds(fmriprep_folder,
                            sourcedata_folder):

    from bids import BIDSLayout
    fmriprep_layout = BIDSLayout(fmriprep_folder)
    sourcedata_layout = BIDSLayout(sourcedata_folder)

    files = fmriprep_layout.get(extensions=['.nii', 'nii.gz'],
                                modality='func', suffix='preproc')

    confounds = []
    metadata = []

    for f in files:
        kwargs = {}

        for key in ['subject', 'run', 'task', 'session']:
            if hasattr(f, key):
                kwargs[key] = getattr(f, key)

        c = fmriprep_layout.get(suffix='confounds', **kwargs)
        c = c[0]
        confounds.append(c)

        sourcedata_file = sourcedata_layout.get(
            modality='func', extensions='nii.gz', **kwargs)

        assert(len(sourcedata_file) == 1)
        md = sourcedata_layout.get_metadata(sourcedata_file[0].filename)
        metadata.append(md)

    return list(zip(files, confounds, metadata))
