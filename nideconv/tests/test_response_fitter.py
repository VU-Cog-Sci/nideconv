import unittest
import nideconv
import numpy as np
from scipy import signal


def double_gamma_with_d(x, a1=6, a2=12, b1=0.9, b2=0.9, c=0.35, d1=5.4, d2=10.8):
    return (x/(d1))**a1 * np.exp(-(x-d1)/b1) - c*(x/(d2))**a2 * np.exp(-(x-d2)/b2)


class ResponseFytterTest(unittest.TestCase):
    """Tests for ResponseFytter"""

    def create_signals(self,
                       signal_sample_frequency=4,
                       event_1_gain=1,
                       event_2_gain=1,
                       event_1_sd=0,
                       event_2_sd=0,
                       noise_gain=1.5,
                       deconv_sample_frequency=4,
                       deconvolution_interval=[-5, 25]):
        """creates signals to be used for the deconvolution of
        2 specific impulse response shapes, with covariates.
        It's supposed to create a signal that's long enough to
        result in testable outcomes even with moderate
        amounts of noise.
        """

        self.signal_sample_frequency = signal_sample_frequency

        # deconvolution parameters
        self.deconvolution_interval = deconvolution_interval

        # create some exponentially distributed random ISI events (Dale, 1999)
        # of which we will create and deconvolve responses.
        period_durs = np.random.gamma(4.0, 8, size=1000)
        events = period_durs.cumsum()
        self.events_1, self.events_2 = events[0::2], events[1::2]

        self.durations_1, self.durations_2 = np.ones(self.events_1.shape[0])/signal_sample_frequency, \
            np.ones(self.events_2.shape[0])/signal_sample_frequency

        #self.durations_1 -= 1e-5
        #self.durations_2 -= 1e-5
        #self.durations_1, self.durations_2 = None, None
        # these events are scaled with their own underlying covariate.
        # for instance, you could have a model-based variable that scales the signal on a per-trial basis.
        self.events_gains_1 = event_1_gain * np.ones(len(self.events_1)) + \
            np.random.randn(len(self.events_1)) * event_1_sd
        self.events_gains_2 = event_2_gain * np.ones(len(self.events_2)) + \
            np.random.randn(len(self.events_2)) * event_2_sd

        times = np.arange(0, events.max()+45.0, 1.0 /
                          self.signal_sample_frequency)

        event_1_in_times = np.array([((times > te) * (times < te+d)) * eg
                                     for te, d, eg in zip(self.events_1, self.durations_1, self.events_gains_1)]).sum(axis=0)
        event_2_in_times = np.array([((times > te) * (times < te+d)) * eg
                                     for te, d, eg in zip(self.events_2, self.durations_2, self.events_gains_2)]).sum(axis=0)

        # create hrfs
        time_points_hrf = np.arange(0, 20, 1.0/self.signal_sample_frequency)
        self.hrf_1 = double_gamma_with_d(
            time_points_hrf, a1=4.5, a2=10, d1=7.0, d2=10.0)
        self.hrf_2 = double_gamma_with_d(
            time_points_hrf, a1=1.5, a2=10, d1=5.0, d2=10.0)

        self.hrf_1 /= self.hrf_1.max()
        self.hrf_2 /= self.hrf_2.max()

        signal_1 = signal.convolve(event_1_in_times, self.hrf_1, 'full')[
            :times.shape[0]]
        signal_2 = signal.convolve(event_2_in_times, self.hrf_2, 'full')[
            :times.shape[0]]

        # combine the two signals with one another, z-score and add noise
        self.input_data = signal_1 + signal_2
        # input_data = (input_data - np.mean(input_data)) / input_data.std()
        self.input_data += np.random.randn(
            self.input_data.shape[0]) * noise_gain

    def test_vanilla_deconvolve(self,
                                event_1_gain=1,
                                event_2_gain=1,
                                noise_gain=1.5,
                                signal_sample_frequency=4,
                                **kwargs):
        """The simplest of possible tests, two impulse response functions
        with different shapes, both with gain = 1
        """
        self.create_signals(signal_sample_frequency=signal_sample_frequency,
                            event_1_gain=event_1_gain,
                            event_2_gain=event_2_gain,
                            event_1_sd=0,
                            event_2_sd=0,
                            noise_gain=noise_gain)

        self.rfy = nideconv.ResponseFitter(
            input_signal=self.input_data,
            sample_rate=self.signal_sample_frequency)

        # first event type, no covariate
        self.rfy.add_event(
            event_name='1',
            onset_times=self.events_1,
            durations=self.durations_1,
            # durations=None,
            interval=self.deconvolution_interval,
            **kwargs
        )
        # second
        self.rfy.add_event(
            event_name='2',
            onset_times=self.events_2,
            durations=self.durations_2,
            # durations=None,
            interval=self.deconvolution_interval,
            **kwargs
        )

        self.rfy.regress()

        #self.assertAlmostEqual(rfy.event_types['1'].timecourses['int'], event_1_gain)
        #self.assertAlmostEqual(rfy.event_types['2'].timecourses['int'], event_2_gain)


if __name__ == '__main__':
    unittest.main()
