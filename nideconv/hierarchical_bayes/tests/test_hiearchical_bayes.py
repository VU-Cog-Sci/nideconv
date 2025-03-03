import nose
import nideconv
from nideconv.simulate import simulate_fmri_experiment
from nideconv.hierarchical_bayes import HierarchicalBayesianModel
import numpy as np
from scipy import signal
import logging
import pandas as pd
import unittest

class HierarchicalBayestest(unittest.TestCase):
    """Tests for ResponseFitter"""

    def setUp(self,
              n_subjects=25,
              TR=1.5):

        self.TR = TR
        self.log = logging.getLogger('hb_test')


        self.data, self.onsets, self.parameters = (
            simulate_fmri_experiment([{'name': 'Condition A', 'mu_group': 5, 'std_group': 3},
                                      {'name': 'Condition B', 'mu_group': 10, 'std_group': 3}],
                                     n_subjects=n_subjects,
                                     run_duration=60,
                                     n_trials=10,
                                     TR=TR)
        )

        self.hfit = HierarchicalBayesianModel()

        df = []

        for subject, d in self.data.reset_index().groupby(['subject']):
            fitter = nideconv.ResponseFitter(
                d.signal.values, 1. / self.TR)

            fitter.add_event('Condition A',
                             self.onsets.loc[subject, 'Condition A'].onset,
                             interval=[0, 20])

            fitter.add_event('Condition B',
                             self.onsets.loc[subject, 'Condition B'].onset,
                             interval=[0, 20])

            self.hfit.add_run(fitter, subject)

    def _test_model(self,
                   cauchy_priors=False,
                   subjectwise_errors=False,
                   recompile=False):

        self.hfit.build_model(cauchy_priors=cauchy_priors,
                              subjectwise_errors=subjectwise_errors,
                              recompile=recompile)
        self.hfit.sample()
        self.assert_correlation_subject_estimates()

    def test_model_cauchy1(self):
        self.log.info('Testing a model with cauchy priors and groupwise '
                      'error terms.')
        return self._test_model(cauchy_priors=True)

    def test_model_cauchy2(self):
        self.log.info('Testing a model with cauchy priors and subjectwise '
                      'error terms.')
        return self._test_model(cauchy_priors=True,
                               subjectwise_errors=True)

    def test_model_normal1(self):
        self.log.info('Testing a model with normal priors and groupwise '
                      'error terms.')
        return self._test_model(subjectwise_errors=True)

    def test_model_normal2(self):
        self.log.info('Testing a model with normal priors and subjectwise '
                      'error terms.')
        return self._test_model(subjectwise_errors=True)

    def assert_correlation_subject_estimates(self):
        # Detect peaks on a per-subject, per-condition basis
        hrf_peaks = self.hfit.get_mean_subject_timecourses().T.groupby(level=[0, 1]).max()
        tmp = pd.concat((hrf_peaks, self.parameters), 1)

        for condition, d in tmp.groupby(level='event_type'):
            r = np.corrcoef(d['amplitude'], d['value'])[0, 1]
            self.assertGreater(r, 0.9)
            self.log.info('Correlation between subject parameters and their estimates is %.2f ' 
                          'for condition %s' % (r, condition))
