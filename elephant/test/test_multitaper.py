# -*- coding: utf-8 -*-
"""
Unit tests for the multitaper module.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import sys
import unittest

import neo
import numpy as np
import quantities as pq
from numpy.testing import assert_array_almost_equal

from elephant.multitaper import multitaper_from_analog_signals

python_version_major = sys.version_info.major


@unittest.skipUnless(python_version_major == 3,
                     "spectral_connectivity requires python 3")
class MultitaperTest(unittest.TestCase):
    
    def setUp(self):
        self.n_trials = 3
        self.n_samples = 5
        self.n_signals = 2
        trials = np.arange(
            self.n_trials * self.n_samples * self.n_signals, dtype=np.float32)
        trials = trials.reshape(
            (self.n_trials, self.n_samples, self.n_signals))
        self.analog_signals = []
        for trial in trials:
            trial = neo.AnalogSignal(trial, units=pq.mV, copy=False,
                                     sampling_rate=1 * pq.Hz)
            self.analog_signals.append(trial)
        
        # no special magic, just guessed a pattern from the output
        target = np.arange(trials.size, dtype=np.float32).reshape(
            trials.shape)
        self.target = np.transpose(target, axes=(1, 0, 2))

    def test_numpy(self):
        multitaper = multitaper_from_analog_signals(self.analog_signals)
        assert_array_almost_equal(multitaper.time_series, self.target)

    def test_list(self):
        analog_signals_lists = []
        for trial in self.analog_signals:
            trial_list = [neo.AnalogSignal(
                sig, units=trial.units, sampling_rate=trial.sampling_rate) for
                sig in trial]
            analog_signals_lists.append(trial_list)
        multitaper = multitaper_from_analog_signals(analog_signals_lists)
        assert_array_almost_equal(multitaper.time_series, self.target)

    def test_attributes(self):
        multitaper = multitaper_from_analog_signals(self.analog_signals)
        self.assertEqual(multitaper.n_trials, self.n_trials)
        self.assertEqual(multitaper.n_signals, self.n_signals)
        self.assertEqual(multitaper.n_fft_samples, self.n_samples)
        self.assertEqual(multitaper.sampling_frequency, 1.0)
        assert_array_almost_equal(multitaper.time, [0.])
        assert_array_almost_equal(multitaper.frequencies,
                                  [0., 0.2, 0.4, -0.4, -0.2])
        assert_array_almost_equal(multitaper.frequency_resolution, 0.6)


if __name__ == '__main__':
    unittest.main()
