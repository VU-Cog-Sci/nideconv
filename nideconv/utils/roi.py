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



def _make_psc(data):
    mean_img = image.mean_img(data)

    # Replace 0s for numerical reasons
    mean_data = mean_img.get_data()
    mean_data[mean_data == 0] = 1
    denom = image.new_img_like(mean_img, mean_data)

    return image.math_img('data / denom[..., np.newaxis] * 100 - 100',
                          data=data, denom=denom)


