from .response_fitter import ResponseFitter
from .utils import roi
from nilearn._utils import load_niimg
from nilearn import maskers, image
import pandas as pd
import numpy as np
import logging
from .utils import get_time_to_peak_from_timecourse


class NiftiResponseFitter(ResponseFitter):

    def __init__(
        self,
        func_img,
        sample_rate,
        mask=None,
        atlas=None,
        oversample_design_matrix=20,
        add_intercept=True,
        detrend=False,
        standardize=False,
        confounds_for_extraction=None,
        memory=None,
        roi_kws={},
        **kwargs
        ):


        if isinstance(confounds_for_extraction, pd.DataFrame):
            confounds_for_extraction = confounds_for_extraction.values

        self.confounds = confounds_for_extraction

        if atlas is not None:

            input_signal = roi.extract_timecourse_from_nii(
                atlas,
                func_img,
                t_r=1/sample_rate,
                confounds=self.confounds,
                **roi_kws
            )
        else:
            if isinstance(mask, maskers.NiftiMasker):
                self.masker = mask
            else:

                if mask is None:
                    logging.warn('No mask has been given. Nilearn will automatically try to make one')
                else:
                    mask = load_niimg(mask)

                self.masker = maskers.NiftiMasker(
                    mask_img=mask,
                    detrend=detrend,
                    standardize=standardize,
                    memory=memory
                )

            input_signal = self.masker.fit_transform(func_img, confounds=confounds_for_extraction)

        self.n_voxels = input_signal.shape[1]

        super().__init__(
            input_signal=input_signal,
            sample_rate=sample_rate,
            oversample_design_matrix=oversample_design_matrix,
            add_intercept=add_intercept,
            **kwargs
        )

    
    def ridge_regress(self, *args, **kwargs):
        raise NotImplementedError('Not implemented for NiftiResponseFitter')


    def predict_from_design_matrix(self, X=None):
        prediction = super(NiftiResponseFitter, self).predict_from_design_matrix(X)
        return self._inverse_transform(prediction)

    def get_timecourses(
        self, 
        oversample=None,
        average_over_mask=False,
        transform_to_niftis=True,
        **kwargs
        ):

        if len(self.events) is 0:
            raise Exception("No events were added")

        timecourses = super().get_timecourses(
            oversample=oversample,
            melt=False,
            **kwargs
        )

        if transform_to_niftis and hasattr(self, "masker"):
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
                    tc = pd.DataFrame(
                        [tc_nii],
                        index=pd.MultiIndex.from_tuples(
                            [(event_type, covariate, time)],
                            names=['event type', 'covariate', 'time']
                        ),
                        columns=['nii']
                    )
                    tc_df.append(tc)

                return pd.concat(tc_df)
        else:
            return timecourses

    def _inverse_transform(self, data):
        return self.masker.inverse_transform(data)

    def get_residuals(self):
        residuals = image.math_img(
            'data - prediction',
            data=self._inverse_transform(self.input_signal),
            prediction=self.predict_from_design_matrix()
        )

        return residuals

    def get_rsq(self):
        """
        calculate the rsq of a given fit. 
        calls predict_from_design_matrix to predict the signal that has been fit
        """

        return self._inverse_transform(super().get_rsq())

    def get_time_to_peak(
        self,
        oversample=None,
        negative_peak=False,
        include_prominence=False):
        
        if oversample is None:
            oversample = self.oversample_design_matrix

        if include_prominence:
            ix = ['time peak', 'prominence']
        else:
            ix = ['time peak']

        peaks = self.get_timecourses(
            oversample=oversample,
            transform_to_niftis=False
            ).groupby(['event type', 'covariate']).apply(
                get_time_to_peak_from_timecourse,
                negative_peak=negative_peak
            )
        
        return peaks.apply(
            lambda d: self._inverse_transform(d), 1
            ).to_frame('nii').loc[(slice(None), slice(None), ix), :]


class GroupNiftiResponseFitter(object):


    def __init__(
        self,
        oversample_design_matrix=20,
        detrend=False,
        standardize=False,
        memory=None,
        add_intercept=True,
        **kwargs
        ):
        
        self.oversample_design_matrix = oversample_design_matrix
        self.add_intercept = add_intercept

        self.detrend = detrend
        self.standardize = standardize
        self.memory = memory

        self.events = []
        self.images = []
    
    def add_event(
        self,
        event=None,
        basis_set='fir', 
        interval=[0,10], 
        n_regressors=None, 
        covariates=None,
        add_intercept=True,
        **kwargs):


        event_kws = {
            'event':event,
            'basis_set':basis_set,
            'interval':interval,
            'n_regressors':n_regressors,
            'covariates':covariates,
            'add_intercept':add_intercept
        }

        event_kws.update(kwargs)

        self.events.append(event_kws)

    
    def add_image(
        self,
        func_img,
        behavior,
        sample_rate,
        subject,
        run=None,
        mask=None,
        atlas=None,
        confounds=None):
       
        image_kws = {
            'func_img':func_img,
            'behavior':behavior,
            'subject':subject,
            'run':run,
            'mask':mask,
            'atlas': atlas,
            'confounds':confounds,
            'sample_rate':sample_rate
        }

        self.images.append(image_kws)


    def fit(
        self,
        store_residuals=False,
        type='ols',
        cv=20,
        alphas=None,
        keep_timeseries=False
        ):

        self.timecourses = pd.DataFrame()

        self.fitters = pd.DataFrame(
            [],
            index=pd.MultiIndex.from_tuples([], names=['subject', 'run']),
            columns=['fitter']
        )

        for image in self.images:

            logging.info('Fitting subject {subject}, run {run}'.format(**image))
            
            fitter = NiftiResponseFitter(
                image['func_img'],
                sample_rate=image['sample_rate'],
                mask=image['mask'],
                atlas=image['atlas'],
                oversample_design_matrix=self.oversample_design_matrix,
                add_intercept=self.add_intercept,
                detrend=self.detrend,
                standardize=self.standardize,
                memory=self.memory
            )


            if image['confounds'] is not None:
                fitter.add_confounds('confounds', image['confounds'].copy())

            for event in self.events:

                behavior = image['behavior']
                behavior = behavior[behavior.event_type == event['event']] 

                if event['covariates'] is None:
                    covariate_matrix = None
                else:
                    covariate_matrix = behavior[event['covariates']]

                    if event['add_intercept']:
                        intercept_matrix = pd.DataFrame(
                            np.ones((len(covariate_matrix), 1)),
                            columns=['intercept'],
                            index=covariate_matrix.index
                        )
                        covariate_matrix = pd.concat((intercept_matrix, covariate_matrix), 1)
                
                if 'duration' in behavior and np.isfinite(behavior['duration']).all():
                    durations = behavior['duration']
                else:
                    durations = None

                fitter.add_event(
                    event['event'],
                    behavior.onset,
                    event['basis_set'],
                    event['interval'],
                    event['n_regressors'],
                    durations,
                    covariate_matrix
                )
        
            fitter.fit(
                type=type,
                cv=cv,
                alphas=alphas,
                store_residuals=store_residuals
            )

            tc = fitter.get_timecourses()

            if not keep_timeseries:
                tc.input_signal = None

            self.fitters.loc[(image['subject'], image['run']), 'fitter'] = fitter

            tc['subject'] = image['subject']
            tc['run'] = image['run']

            tc.set_index(['subject', 'run'])

            self.timecourses = pd.concat((self.timecourses, tc), axis=0)
            
        self.timecourses = self.timecourses.reset_index().set_index(['subject', 'run', 'event type', 'covariate', 'time'])


    def get_timecourses(
        self,
        mask=None,
        names=None,
        event_types=None,
        covariates=None,
        resampling_target=None
        ):

        if mask is None:
            return self.timecourses

        if type(mask) is not list:
            mask = [mask]

        
        if event_types is None:
            event_types = slice(None)
        
        if covariates is None:
            covariates = slice(None)

        if names:
            names = pd.Index(names, name='roi')

        masker = maskers.NiftiMasker(mask_img=mask, resampling_target=resampling_target)

        masker.fit()

        ix = pd.IndexSlice
        timecourses = self.timecourses.loc[ix[:, :,event_types, covariates, :], :]

        logging.info('Concatenating...')
        concat_niis = image.concat_imgs(timecourses.nii.tolist())

        logging.info('Masking...')
        timecourses = pd.DataFrame(masker.transform(concat_niis),
                                   index=timecourses.index,
                                   columns=names)

        return timecourses

    def get_time_to_peak(self,
                         oversample=None, 
                         cutoff=1.0, 
                         negative_peak=False,
                         include_prominence=False):

        if oversample is None:
            oversample = self.oversample_design_matrix

        if include_prominence:
            ix = ['time peak', 'prominence']
        else:
            ix = ['time peak']


        peaks = self.get_timecourses(oversample=oversample,
                                    transform_to_niftis=False)\
                   .groupby(['event type', 'covariate'])\
                   .apply(get_time_to_peak_from_timecourse,
                          negative_peak=negative_peak,
                          cutoff=cutoff)\
            .loc[(slice(None), slice(None), ix), :]
        
        return peaks.groupby(['event type', 'covariate']) \
            .apply(lambda d: self._inverse_transform(d), 1)\
            .to_frame('nii')
