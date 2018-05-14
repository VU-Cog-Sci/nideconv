from .response_fytter import ResponseFytter
from nilearn._utils import load_niimg
from nilearn import input_data, image
import pandas as pd
import numpy as np


class NiftiResponseFytter(ResponseFytter):

    def __init__(self,
                 func_img,
                 sample_rate,
                 oversample_design_matrix=20,
                 add_intercept=True,
                 mask=None,
                 weight_mask=True,
                 average_over_mask=False,
                 threshold=0,
                 detrend=False,
                 standardize=False,
                 confounds_for_extraction=None,
                 memory=None,
                 **kwargs):


        self.average_over_mask = average_over_mask
        self.confounds = confounds_for_extraction

        if isinstance(mask, input_data.NiftiMasker):
            self.masker = mask
        else:

            mask = load_niimg(mask)

            if weight_mask:
                bool_mask = image.math_img('mask > {}'.format(threshold),
                                           mask=mask)
                self.masker = input_data.NiftiMasker(bool_mask,
                                                     detrend=detrend,
                                                     standardize=standardize,
                                                     memory=memory)
                self.mask_weights = self.masker.fit_transform(load_niimg(mask))
                self.mask_weights /= self.mask_weights.sum()
            else:
                self.masker = input_data.NiftiMasker(mask,
                                                     detrend=detrend,
                                                     standardize=standardize,
                                                     memory=memory)

        input_signal = self.masker.fit_transform(func_img) 
        self.n_voxels = input_signal.shape[1]

        if not weight_mask:
            self.mask_weights = np.ones((1,self.n_voxels))
            self.mask_weights /= self.n_voxels


        super(NiftiResponseFytter, self).__init__(input_signal=input_signal,
                                           sample_rate=sample_rate,
                                           oversample_design_matrix=oversample_design_matrix,
                                           add_intercept=add_intercept,
                                           **kwargs)


    
    def ridge_regress(self, *args, **kwargs):
        raise NotImplementedError('Not implemented for NiftiResponseFytter')


    def predict_from_design_matrix(self,
                                   average_over_mask=False,
                                   X=None):
        prediction = super(NiftiResponseFytter, self).predict_from_design_matrix(X)
        return self._inverse_transform(prediction)

    def get_timecourses(self, 
                        oversample=None,
                        average_over_mask=None,
                        **kwargs
                        ):

        if average_over_mask is None:
            average_over_mask = self.average_over_mask

        if len(self.events) is 0:
            raise Exception("No events were added")

        timecourses = super(NiftiResponseFytter, self).get_timecourses(oversample=oversample,
                                                                       melt=False,
                                                                       **kwargs)

        if average_over_mask:
            tc_nii = self._inverse_transform(timecourses, True)
        else:
            tc_df = []
            for (event_type, covariate), tc in timecourses.groupby(level=['event type', 'covariate']):
                tc_nii = self._inverse_transform(tc, False)
                tc = pd.DataFrame([tc_nii], index=pd.MultiIndex.from_tuples([(event_type, covariate)],
                                                                         names=['event type', 'covariate']),
                                  columns=['nii'])
                tc_df.append(tc)

            return pd.concat(tc_df)

                                       


    def _inverse_transform(self, 
                           data,
                           average_over_mask=None):

        if average_over_mask is None:
            average_over_mask = self.average_over_mask

        if average_over_mask:
            data = data.dot(self.mask_weights.T) 
            return data.sum(1)
        else:
            return self.masker.inverse_transform(data)

