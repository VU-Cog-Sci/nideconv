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
            for (event_type, covariate, time), tc in timecourses.groupby(level=['event type', 'covariate', 'time']):
                #timepoints = tc.index.get_level_values('time')
                tc_nii = self._inverse_transform(tc)
                tc = pd.DataFrame([tc_nii], index=pd.MultiIndex.from_tuples([(event_type, covariate, time)],
                                                                         names=['event type', 'covariate', 'time']),
                                  columns=['nii'])
                tc_df.append(tc)

            return pd.concat(tc_df)

                                       


    def _inverse_transform(self, 
                           data):

        return self.masker.inverse_transform(data)


class GroupNiftiResponseFytter(object):


    def __init__(self,
                 oversample_design_matrix=20,
                 detrend=False,
                 standardize=False,
                 memory=None,
                 add_intercept=True,
                 **kwargs):
        
        self.oversample_design_matrix = oversample_design_matrix
        self.add_intercept = add_intercept

        self.detrend = detrend
        self.standardize = standardize
        self.memory = memory

        self.events = []
        self.images = []



    
    def add_event(self,
                 event=None,
                 basis_set='fir', 
                 interval=[0,10], 
                 n_regressors=None, 
                 covariates=None,
                 add_intercept=True,
                 **kwargs):


        event_kws = {'event':event,
                     'basis_set':basis_set,
                     'interval':interval,
                     'n_regressors':n_regressors,
                     'covariates':covariates,
                     'add_intercept':add_intercept
                     }
        event_kws.update(kwargs)

        self.events.append(event_kws)

    
    def add_image(self,
                  func_img,
                  behavior,
                  sample_rate,
                  subj_idx,
                  run=None,
                  mask=None,
                  confounds=None):
       
        image_kws = {'func_img':func_img,
                     'behavior':behavior,
                     'subj_idx':subj_idx,
                     'run':run,
                     'mask':mask,
                     'confounds':confounds,
                     'sample_rate':sample_rate}

        self.images.append(image_kws)


    def fit(self, keep_timeseries=False):

        self.timecourses = pd.DataFrame()

        self.fitters = pd.DataFrame([],
                                    index=pd.MultiIndex.from_tuples([], names=['subj_idx', 'run']),
                                    columns=['fitter'])

        for image in self.images:

            logging.info('Fitting subject {subj_idx}, run {run}'.format(**image))
            
            fitter = NiftiResponseFytter(image['func_img'],
                                         image['sample_rate'],
                                         image['mask'],
                                         self.oversample_design_matrix,
                                         self.add_intercept,
                                         self.detrend,
                                         self.standardize,
                                         memory=self.memory)


            if image['confounds'] is not None:
                fitter.add_confounds('confounds', image['confounds'].copy())

            for event in self.events:

                behavior = image['behavior']
                behavior = behavior[behavior.trial_type == event['event']] 

                if event['covariates'] is None:
                    covariate_matrix = None
                else:
                    covariate_matrix = behavior[covariates]

                    if add_intercept:
                        intercept_matrix = pd.DataFrame(np.ones((len(covariate_matrix), 1)),
                                                        columns=['intercept'],
                                                        index=covariate_matrix.index)
                        covariate_matrix = pd.concat((intercept_matrix, covariate_matrix), 1)
                
                if 'duration' in behavior and np.isfinite(behavior['duration']).all():
                    durations = behavior['duration']
                else:
                    durations = None

                fitter.add_event(event['event'],
                                 behavior.onset,
                                 event['basis_set'],
                                 event['interval'],
                                 event['n_regressors'],
                                 durations,
                                 covariate_matrix)
        
            fitter.regress()

            tc = fitter.get_timecourses()

            if not keep_timeseries:
                tc.input_signal = None

            self.fitters.loc[(image['subj_idx'], image['run']), 'fitter'] = fitter

            tc['subj_idx'] = image['subj_idx']
            tc['run'] = image['run']

            tc.set_index(['subj_idx', 'run'])

            self.timecourses = pd.concat((self.timecourses, tc), 0)
            
        self.timecourses = self.timecourses.reset_index().set_index(['subj_idx', 'run', 'event type', 'covariate', 'time'])


    def get_timecourses(self, mask=None, names=None):

        if mask is None:
            return self.timecourses

        if type(mask) is not list:
            mask = [mask]


        if names:
            names = pd.Index(names, name='roi')

        masker = input_data.NiftiMapsMasker(mask)

        masker.fit()

        timecourses = self.timecourses.nii.apply(lambda d: pd.Series(masker.transform(d).squeeze(), 
                                                                index=names))

        return timecourses

