# -*- coding: utf-8 -*-
"""
Unit tests for the synchrofact detection app
"""

import unittest

import neo
import numpy as np
import quantities as pq

from elephant import utils
from numpy.testing import assert_array_equal


class TestUtils(unittest.TestCase):

    def test_check_neo_consistency(self):
        self.assertRaises(TypeError,
                          utils.check_neo_consistency,
                          [], object_type=neo.SpikeTrain)
        self.assertRaises(TypeError,
                          utils.check_neo_consistency,
                          [neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s),
                           np.arange(2)], object_type=neo.SpikeTrain)
        self.assertRaises(ValueError,
                          utils.check_neo_consistency,
                          [neo.SpikeTrain([1]*pq.s,
                                          t_start=1*pq.s,
                                          t_stop=2*pq.s),
                           neo.SpikeTrain([1]*pq.s,
                                          t_start=0*pq.s,
                                          t_stop=2*pq.s)],
                          object_type=neo.SpikeTrain)
        self.assertRaises(ValueError,
                          utils.check_neo_consistency,
                          [neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s),
                           neo.SpikeTrain([1]*pq.s, t_stop=3*pq.s)],
                          object_type=neo.SpikeTrain)
        self.assertRaises(ValueError,
                          utils.check_neo_consistency,
                          [neo.SpikeTrain([1]*pq.ms, t_stop=2000*pq.ms),
                           neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s)],
                          object_type=neo.SpikeTrain)

    def test_discretise_spiketimes(self):
        times = (np.arange(10) + np.random.uniform(size=10)) * pq.ms
        spiketrains = [neo.SpikeTrain(times, t_stop=10*pq.ms)] * 5
        discretised_spiketrains = utils.discretise_spiketimes(spiketrains,
                                                              1/pq.ms)
        for idx in range(len(spiketrains)):
            np.testing.assert_array_equal(discretised_spiketrains[idx].times,
                                          np.arange(10) * pq.ms)

        # test for single spiketrain
        discretised_spiketrain = utils.discretise_spiketimes(spiketrains[0],
                                                             1 / pq.ms)
        np.testing.assert_array_equal(discretised_spiketrain.times,
                                      np.arange(10) * pq.ms)

        # test that no spike will be before t_start
        spiketrain = neo.SpikeTrain([0.7, 5.1]*pq.ms,
                                    t_start=0.5*pq.ms, t_stop=10*pq.ms)
        discretised_spiketrain = utils.discretise_spiketimes(spiketrain,
                                                             1 / pq.ms)
        np.testing.assert_array_equal(discretised_spiketrain.times,
                                      [0.5, 5] * pq.ms)

    def test_round_binning_errors(self):
        with self.assertWarns(UserWarning):
            n_bins = utils.round_binning_errors(0.999999, tolerance=1e-6)
            self.assertEqual(n_bins, 1)
        self.assertEqual(utils.round_binning_errors(0.999999, tolerance=None),
                         0)
        array = np.array([0, 0.7, 1 - 1e-8, 1 - 1e-9])
        with self.assertWarns(UserWarning):
            corrected = utils.round_binning_errors(array.copy())
            assert_array_equal(corrected, [0, 0, 1, 1])
        assert_array_equal(
            utils.round_binning_errors(array.copy(), tolerance=None),
            [0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
