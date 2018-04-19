import unittest
import response_fytter
from response_fytter.simulate import simulate_fmri_experiment
from response_fytter.hierarchical_bayes import HierarchicalBayesianModel
import numpy as np
from scipy import signal

class HierarchicalBayestest(unittest.TestCase):
    """Tests for ResponseFytter"""
    def create_signals(self,
                       TR=1.5):
        self.TR = TR

        return simulate_fmri_experiment([{'name':'Condition A', 'mu_group':5, 'std_group':1},
                                         {'name':'Condition B', 'mu_group':10, 'std_group':1}], 
                                        n_subjects=5, run_duration=60, n_trials=[1, 5], TR=TR)

    def test_simple_model(self):
        data, onsets, parameters = self.create_signals()

        self.hfit = HierarchicalBayesianModel()

        df = []

        for (subj_idx, run), d in data.reset_index().groupby(['subj_idx', 'run']):
            fytter = response_fytter.ResponseFytter(d.signal.values, 1./self.TR)
            
            fytter.add_event('Condition A',
                             onset_times=onsets.loc[subj_idx, run, 'Condition A'].onset,
                             interval=[0, 20])
            
            fytter.add_event('Condition B',            
                             onset_times=onsets.loc[subj_idx, run, 'Condition B'].onset,
                             interval=[0, 20])    
            
            self.hfit.add_run(fytter, subj_idx)
        self.hfit.build_model()
        self.hfit.sample()
        self.hfit.plot_subject_timecourses(covariates='intercept')


if __name__ == '__main__':
    unittest.main()
