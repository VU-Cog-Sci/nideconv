from .response_fytter import ResponseFytter
from nilearn._utils import load_niimg
from nilearn import input_data, image
import pandas as pd
import numpy as np
import logging


class NiftiResponseFytter(ResponseFytter):

    def __init__(self,
                 func_img,
                 sample_rate,
                 mask=None,
                 oversample_design_matrix=20,
                 add_intercept=True,
                 threshold=0,
                 detrend=False,
                 standardize=False,
                 confounds_for_extraction=None,
                 memory=None,
                 **kwargs):


        self.confounds = confounds_for_extraction

        if isinstance(mask, input_data.NiftiMasker):
            self.masker = mask
        else:

            if mask is None:
                logging.warn('No mask has been given. Nilearn will automatically try to'\
                              'make one')
            else:
                mask = load_niimg(mask)
            self.masker = input_data.NiftiMasker(mask,
                                                 detrend=detrend,
                                                 standardize=standardize,
                                                 memory=memory)

        input_signal = self.masker.fit_transform(func_img) 
        self.n_voxels = input_signal.shape[1]

        super(NiftiResponseFytter, self).__init__(input_signal=input_signal,
                                           sample_rate=sample_rate,
                                           oversample_design_matrix=oversample_design_matrix,
                                           add_intercept=add_intercept,
                                           **kwargs)


    
    def ridge_regress(self, *args, **kwargs):
        raise NotImplementedError('Not implemented for NiftiResponseFytter')


    def predict_from_design_matrix(self, X=None):
        prediction = super(NiftiResponseFytter, self).predict_from_design_matrix(X)
        return self._inverse_transform(prediction)

    def get_timecourses(self, 
                        oversample=None,
                        average_over_mask=False,
                        **kwargs
                        ):

        if len(self.events) is 0:
            raise Exception("No events were added")

        timecourses = super(NiftiResponseFytter, self).get_timecourses(oversample=oversample,
                                                                       melt=False,
                                                                       **kwargs)

        if average_over_mask:
            
            average_over_mask = load_niimg(average_over_mask)

            weights = image.math_img('mask / mask.sum()', 
                                     mask=average_over_mask)

            weights = self.masker.fit_transform(weights)

            timecourses = timecourses.dot(weights.T) 
            return timecourses.sum(1)

        else:
            tc_df = []
            for (event_type, covariate), tc in timecourses.groupby(level=['event type', 'covariate']):
                tc_nii = self._inverse_transform(tc)
                tc = pd.DataFrame([tc_nii], index=pd.MultiIndex.from_tuples([(event_type, covariate)],
                                                                         names=['event type', 'covariate']),
                                  columns=['nii'])
                tc_df.append(tc)

            return pd.concat(tc_df)

                                       


    def _inverse_transform(self, 
                           data):

        return self.masker.inverse_transform(data)

