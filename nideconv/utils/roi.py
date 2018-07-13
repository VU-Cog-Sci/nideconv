import pandas as pd
from nilearn import input_data
import nibabel as nb
from nilearn._utils import check_niimg
from nilearn import image

def _extract_timecourse_from_nii(atlas, 
                                nii, 
                                mask=None,
                                confounds=None,
                                atlas_type=None,
                                standardize='psc',
                                *args,
                                **kwargs):

    if atlas_type is None:        
        maps = check_niimg(atlas.maps)
        
        if len(maps.shape) == 3:
            atlas_type = 'labels'
        else:
            atlas_type = 'prob'
            
    if atlas_type == 'labels':        
        masker = input_data.NiftiLabelsMasker(atlas.maps, mask_img=mask, *args, **kwargs)
    else:
        masker = input_data.NiftiMapsMasker(atlas.maps, mask_img=mask, *args, **kwargs)
    

    results = masker.fit_transform(nii, standardize=True, confounds=confounds)

    if standardize == 'psc':
        mean_image = image.mean_image(nii)
        std_image = image.math_img('nii.std(-1)', nii=nii)

        mean_st_rois = masker.transform(mean_image)

        results = (results * std_image) / mean_image * 100


    # For weird atlases that have a label for the background
    if len(atlas.labels) == results.shape[1] + 1:
        atlas.labels = atlas.labels[1:]

    return pd.DataFrame(results, columns=atlas.labels)



    
