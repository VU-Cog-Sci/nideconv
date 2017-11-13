import unittest

class ResponseFytterTest(unittest.TestCase):
    """Tests for ResponseFytter"""
    def setUp(self, 
        signal_sample_frequency=4, 
        event_1_gain=1, 
        event_2_gain=1,
        event_1_sd=0,
        event_2_sd=0):
    
        # signal parameters
        signal_sample_frequency = 4
        event_1_gain, event_2_gain = 1, 1# 2.3, 0.85
        noise_gain = 1.5

        # deconvolution parameters
        deconvolution_interval = [-5, 25]    

        # create some exponentially distributed random ISI events (Dale, 1999) of which we will create and deconvolve responses. 
        period_durs = np.random.gamma(4.0,1.5,size = 1000)
        events = period_durs.cumsum()
        events_1, events_2 = events[0::2], events[1::2]

        durations_1, durations_2 = np.ones(events_1.shape[0])/deconv_sample_frequency, \
                                    np.ones(events_2.shape[0])/deconv_sample_frequency

        # these events are scaled with their own underlying covariate. 
        # for instance, you could have a model-based variable that scales the signal on a per-trial basis. 
        events_gains_1 = event_1_gain * np.ones(len(events_1)) + np.random.randn(len(events_1)) * event_1_sd
        events_gains_2 = event_2_gain * np.ones(len(events_2)) + np.random.randn(len(events_2)) * event_2_sd

        times = np.arange(0,events.max()+45.0,1.0/signal_sample_frequency)

        event_1_in_times = np.array([((times>te) * (times<te+d)) * eg 
                            for te, d, eg in zip(events_1, durations_1, events_gains_1)]).sum(axis = 0)
        event_2_in_times = np.array([((times>te) * (times<te+d)) * eg 
                            for te, d in zip(events_2, durations_2, events_gains_2)]).sum(axis = 0)

        # create hrfs
        def double_gamma_with_d(x, a1 = 6, a2 = 12, b1 = 0.9, b2 = 0.9, c = 0.35,d1=5.4,d2=10.8):
        return np.array([(t/(d1))**a1 * np.exp(-(t-d1)/b1) - c*(t/(d2))**a2 * np.exp(-(t-d2)/b2) for t in x])

        hrf_1 = double_gamma_with_d(time_points_hrf, a1 = 4.5, a2 = 10, d1 = 5.0, d2 = 10.0)
        hrf_2 = double_gamma_with_d(time_points_hrf, a1 = 1.5, a2 = 10, d1 = 3.0, d2 = 10.0)

        hrf_1 /= hrf_1.max()
        hrf_2 /= hrf_2.max()

        signal_1 = signal.fftconvolve(event_1_in_times, hrf_1, 'full')[:times.shape[0]]
        signal_2 = signal.fftconvolve(event_2_in_times, hrf_2, 'full')[:times.shape[0]]

        # combine the two signals with one another, z-score and add noise
        self.input_data = signal_1 + signal_2
        # input_data = (input_data - np.mean(input_data)) / input_data.std()
        self.input_data += np.random.randn(input_data.shape[0]) * noise_gain



    def tearDown(self):




















if __name__ == '__main__':
    unittest.main()