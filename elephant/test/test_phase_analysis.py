# -*- coding: utf-8 -*-
"""
Unit tests for the phase analysis module.

:copyright: Copyright 2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function

import unittest

from neo import SpikeTrain, AnalogSignal
import numpy as np
import quantities as pq
import os.path
import scipy.io
from scipy.io import savemat

import elephant.phase_analysis

from numpy.ma.testutils import assert_allclose


class SpikeTriggeredPhaseTestCase(unittest.TestCase):

    def setUp(self):
        tlen0 = 100 * pq.s
        f0 = 20. * pq.Hz
        fs0 = 1 * pq.ms
        t0 = np.arange(
            0, tlen0.rescale(pq.s).magnitude,
            fs0.rescale(pq.s).magnitude) * pq.s
        self.anasig0 = AnalogSignal(
            np.sin(2 * np.pi * (f0 * t0).simplified.magnitude),
            units=pq.mV, t_start=0 * pq.ms, sampling_period=fs0)
        self.st0 = SpikeTrain(
            np.arange(50, tlen0.rescale(pq.ms).magnitude - 50, 50) * pq.ms,
            t_start=0 * pq.ms, t_stop=tlen0)
        self.st1 = SpikeTrain(
            [100., 100.1, 100.2, 100.3, 100.9, 101.] * pq.ms,
            t_start=0 * pq.ms, t_stop=tlen0)

    def test_perfect_locking_one_spiketrain_one_signal(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            self.st0,
            interpolate=True)

        assert_allclose(phases[0], - np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_many_spiketrains_many_signals(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0, self.st0],
            interpolate=True)

        assert_allclose(phases[0], -np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_one_spiketrains_many_signals(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0],
            interpolate=True)

        assert_allclose(phases[0], -np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_many_spiketrains_one_signal(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            [self.st0, self.st0],
            interpolate=True)

        assert_allclose(phases[0], -np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_interpolate(self):
        phases_int, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            self.st1,
            interpolate=True)

        self.assertLess(phases_int[0][0], phases_int[0][1])
        self.assertLess(phases_int[0][1], phases_int[0][2])
        self.assertLess(phases_int[0][2], phases_int[0][3])
        self.assertLess(phases_int[0][3], phases_int[0][4])
        self.assertLess(phases_int[0][4], phases_int[0][5])

        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            self.st1,
            interpolate=False)

        self.assertEqual(phases_noint[0][0], phases_noint[0][1])
        self.assertEqual(phases_noint[0][1], phases_noint[0][2])
        self.assertEqual(phases_noint[0][2], phases_noint[0][3])
        self.assertEqual(phases_noint[0][3], phases_noint[0][4])
        self.assertNotEqual(phases_noint[0][4], phases_noint[0][5])

        # Verify that when using interpolation and the spike sits on the sample
        # of the Hilbert transform, this is the same result as when not using
        # interpolation with a spike slightly to the right
        self.assertEqual(phases_noint[0][2], phases_int[0][0])
        self.assertEqual(phases_noint[0][4], phases_int[0][0])

    def test_inconsistent_numbers_spiketrains_hilbert(self):
        self.assertRaises(
            ValueError, elephant.phase_analysis.spike_triggered_phase,
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0, self.st0, self.st0], False)

        self.assertRaises(
            ValueError, elephant.phase_analysis.spike_triggered_phase,
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0, self.st0, self.st0], False)

    def test_spike_earlier_than_hilbert(self):
        # This is a spike clearly outside the bounds
        st = SpikeTrain(
            [-50, 50],
            units='s', t_start=-100 * pq.s, t_stop=100 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 1)

        # This is a spike right on the border (start of the signal is at 0s,
        # spike sits at t=0s). By definition of intervals in
        # Elephant (left borders inclusive, right borders exclusive), this
        # spike is to be considered.
        st = SpikeTrain(
            [0, 50],
            units='s', t_start=-100 * pq.s, t_stop=100 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 2)

    def test_spike_later_than_hilbert(self):
        # This is a spike clearly outside the bounds
        st = SpikeTrain(
            [1, 250],
            units='s', t_start=-1 * pq.s, t_stop=300 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 1)

        # This is a spike right on the border (length of the signal is 100s,
        # spike sits at t=100s). However, by definition of intervals in
        # Elephant (left borders inclusive, right borders exclusive), this
        # spike is not to be considered.
        st = SpikeTrain(
            [1, 100],
            units='s', t_start=-1 * pq.s, t_stop=200 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 1)

    # This test handles the correct dealing with input signals that have
    # different time units, including a CompoundUnit
    def test_regression_269(self):
        # This is a spike train on a 30KHz sampling, one spike at 1s, one just
        # before the end of the signal
        cu = pq.CompoundUnit("1/30000.*s")
        st = SpikeTrain(
            [30000., (self.anasig0.t_stop - 1 * pq.s).rescale(cu).magnitude],
            units=pq.CompoundUnit("1/30000.*s"),
            t_start=-1 * pq.s, t_stop=300 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 2)


class MeanVectorTestCase(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-15
        self.n_samples = 200
        # create a sample with all values equal to a random phase-lock phi
        self.lock_value_phi = np.random.uniform(-np.pi, np.pi, 1)
        self.dataset1 = np.ones(self.n_samples) * self.lock_value_phi
        # create a evenly spaced / uniform distribution
        self.dataset2 = np.arange(0, 2 * np.pi, (2 * np.pi) / self.n_samples)
        # create a random distribution
        self.dataset3 = np.random.uniform(-np.pi, np.pi, self.n_samples)

    def testMeanVector_direction_is_phi_and_length_is_1(self):
        """
        Test if the mean vector length is 1 and if the mean direction is phi
        for a sample with all phases equal to phi on the unit circle.

        """
        theta_bar_1, r_1 = elephant.phase_analysis.mean_phase_vector(
            self.dataset1)
        # mean direction must be phi
        self.assertAlmostEqual(theta_bar_1, self.lock_value_phi,
                               delta=self.tolerance)
        # mean vector length must be almost equal 1
        self.assertAlmostEqual(r_1, 1, delta=self.tolerance)

    def testMeanVector_length_is_0(self):
        """
        Test if the mean vector length  is 0 for a evenly spaced distribution
        on the unit circle.
        """
        theta_bar_2, r_2 = elephant.phase_analysis.mean_phase_vector(
            self.dataset2)
        # mean vector length must be almost equal 0
        self.assertAlmostEqual(r_2, 0, delta=self.tolerance)

    def testMeanVector_ranges_of_direction_and_length(self):
        """
        Test if the range of the mean vector direction follows numpy standard
        and is within (-pi, pi].
        Test if the range of the mean vector length is within [0, 1].
        """
        theta_bar_3, r_3 = elephant.phase_analysis.mean_phase_vector(
            self.dataset3)
        # mean vector direction
        self.assertTrue(-np.pi < theta_bar_3 <= np.pi)
        # mean vector length
        self.assertTrue(0 <= r_3 <= 1)


class PhaseDifferenceTestCase(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-15
        self.n_samples = 200

    def testPhaseDifference_in_range_minus_pi_to_pi(self):
        """
        Test if the range of the phase difference is within [-pi, pi] for
        random pairs of alpha and beta.
        """
        alpha = np.random.uniform(-np.pi, np.pi, self.n_samples)
        beta = np.random.uniform(-np.pi, np.pi, self.n_samples)

        phase_diff = elephant.phase_analysis.phase_difference(alpha, beta)
        self.assertTrue((-np.pi <= phase_diff).all()
                        and (phase_diff <= np.pi).all())

    def testPhaseDifference_is_delta(self):
        """
        Test if the phase difference is random delta for random pairs of
        alpha and beta, where beta is a copy of alpha shifted by delta.
        """
        delta = np.random.uniform(-np.pi, np.pi, self.n_samples)
        alpha = np.random.uniform(-np.pi, np.pi, self.n_samples)
        _beta = alpha - delta
        beta = np.arctan2(np.sin(_beta), np.cos(_beta))

        phase_diff = elephant.phase_analysis.phase_difference(alpha, beta)
        np.testing.assert_allclose(phase_diff, delta, atol=1e-10)


class PhaseLockingValueTestCase(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-15
        self.phase_shift = np.pi / 4
        self.num_time_points = 1000
        self.num_trials = 100

        # create two random uniform distributions (all trials are identical)
        self.signal_x = \
            np.full([self.num_trials, self.num_time_points],
                    np.random.uniform(-np.pi, np.pi, self.num_time_points))
        self.signal_y = \
            np.full([self.num_trials, self.num_time_points],
                    np.random.uniform(-np.pi, np.pi, self.num_time_points))

        # create two random uniform distributions, where all trails are random
        self.random_x = np.random.uniform(
            -np.pi, np.pi, (1000, self.num_time_points))
        self.random_y = np.random.uniform(
            -np.pi, np.pi, (1000, self.num_time_points))

        # simple samples of different shapes to assert ErrorRaising
        self.simple_x = np.array([[0, -np.pi, np.pi], [0, -np.pi, np.pi]])
        self.simple_y = np.array([0, -np.pi, np.pi])
        self.simple_z = np.array([0, np.pi, np.pi / 2, -np.pi])

    def testPhaseLockingValue_identical_signals_both_identical_trials(self):
        """
        Test if the PLV's are 1, when 2 identical signals with identical
        trials are passed. PLV's needed to be 1, due to the constant phase
        difference of 0 across trials at each time-point.
        """
        list1_plv_t = \
            elephant.phase_analysis.phase_locking_value(self.signal_x,
                                                        self.signal_x)
        target_plv_r_is_one = np.ones_like(list1_plv_t)
        np.testing.assert_allclose(list1_plv_t, target_plv_r_is_one,
                                   self.tolerance)

    def testPhaseLockingValue_different_signals_both_identical_trials(self):
        """
        Test if the PLV's are 1, when 2 different signals are passed, where
        within each signal the trials are identical. PLV's needed to be 1,
        due to a constant phase difference across trials, which may vary for
        different time-points.
        """
        list2_plv_t = elephant.phase_analysis.phase_locking_value(
            self.signal_x, self.signal_y)
        target_plv_r_is_one = np.ones_like(list2_plv_t)
        np.testing.assert_allclose(list2_plv_t, target_plv_r_is_one,
                                   atol=3e-15)

    def testPhaseLockingValue_different_signals_both_different_trials(self):
        """
        Test if the PLV's are close to 0, when 2 different signals are passed,
        where both have different trials, which are all randomly distributed.
        The PLV's needed to be close to 0, do to a random
        phase difference across trials for each time-point.
        """
        list3_plv_t = elephant.phase_analysis.phase_locking_value(
            self.random_x, self.random_y)
        target_plv_is_zero = np.zeros_like(list3_plv_t)
        # use default value from np.allclose() for atol=1e-8 to prevent failure
        np.testing.assert_allclose(list3_plv_t, target_plv_is_zero,
                                   rtol=1e-2, atol=1.1e-1)

    def testPhaseLockingValue_raise_Error_if_trial_number_is_different(self):
        """
        Test if a ValueError is raised, when the signals have different
        number of trails.
        """
        # different numbers of trails
        np.testing.assert_raises(
            ValueError, elephant.phase_analysis.phase_locking_value,
            self.simple_x, self.simple_y)

    def testPhaseLockingValue_raise_Error_if_trial_lengths_are_different(self):
        """
        Test if a ValueError is raised, when within a trail-pair of the signals
        the trial-lengths are different.
        """
        # different lengths in a trail pair
        np.testing.assert_raises(
            ValueError, elephant.phase_analysis.phase_locking_value,
            self.simple_y, self.simple_z)


class WeightedPhaseLagIndexTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(73)
        cls._generate_datasets_for_ground_truth_testing()

    @staticmethod
    def _generate_datasets_for_ground_truth_testing(ntrial=40, tlength=2.5,
                                                    srate=250):
        """
        Generates simple sinusoidal, artificial LFP-datasets.

        This function simulates the recording of two LFP-signals over several
        trials, for a certain duration of time and with a specified sampling
        rate. These datasets will be used to calculate one WPLI-ground-truth
        with the MATlAB package FieldTrip and its function
        ft_connectivity_wpli().
        Moreover this function generates a secound pair of LFP-signals, which
        will be used to calculate two other WPLI-ground-truths. One from the
        FieldTrip wrapper ft_connectivityanalysis() and the other from MNEs
        spectral_connectivity(), which both use multitaper for FFT, therefore
        just certain frequencies will be compared.

        Parameters
        ----------
        ntrial: int
            Number of trials in the datasets.
        tlength: float
            Time length of the datasets in seconds.
        srate: float
            Sampling rate used to 'record' the signals in Hz.

        Returns
        -------
        None

        Notes
        -----
        Used versions of MATLAB & FieldTrip:
        MATLAB          Version 9.9         (2020b)
        FieldTrip       fieldtrip-20210128

        Instead of using the ft_connectivityanalysis() wrapper function, which
        expects preprocessed data, the ft_connectivity_wpli() is called
        directly. That's because the preprocessing functions of FieldTrip have
        no 'conventional' Fourier-Transformation (FFT) method, but among
        others a multitaper-FFT. Because this python-version of WPLI doesn't
        use multitaper, but conventional FFT, the utilized MATLAB script also
        uses the conventional FFT to preprocess the data and calculate the
        cross-spectrum, which will then be passed to the ft_connectivity_wpli.

        """
        times = np.arange(0, tlength, 1 / srate)

        # # lfps_1 & 2 will be used for
        # # 1) calculating ground-truth with FieldTrips' ft_connectivity_wpli
        # # 2) comparison to mutlitaper-approaches like
        # # FieldTrips' ft_connectivityanalysis()
        # # and MNEs' spectral_connectivity at certain frequencies
        lfps_1 = [np.cos(2 * 16 * np.pi * times + np.pi/2) +
                  np.cos(2 * 36 * np.pi * times + np.pi / 2) +
                  np.cos(2 * 52 * np.pi * times) +
                  np.cos(2 * 100 * np.pi * times) +
                  np.cos(2 * 70 * np.pi * times + np.pi/2 + np.pi*(i % 2)) +
                  np.random.normal(loc=0.0, scale=1, size=len(times))
                  for i in range(ntrial)]
        lfps_1 = np.stack(lfps_1, axis=0)

        lfps_2 = [np.cos(2 * 16 * np.pi * times) +
                  np.cos(2 * 36 * np.pi * times) +
                  np.cos(2 * 52 * np.pi * times + np.pi/2) +
                  np.cos(2 * 100 * np.pi * times + np.pi/2) +
                  np.cos(2 * 70 * np.pi * times) +
                  np.random.normal(loc=0.0, scale=1, size=len(times))
                  for _ in range(ntrial)]
        lfps_2 = np.stack(lfps_2, axis=0)

        # save artificial LFP-dataset to .mat files
        mdic_1 = {"lfp_matrix": lfps_1, "time": times, "sf": srate}
        mdic_2 = {"lfp_matrix": lfps_2, "time": times, "sf": srate}
        filename1_a = os.path.sep.join(['artificial_LFPs_1.mat'])
        filename2_a = os.path.sep.join(['artificial_LFPs_2.mat'])

        savemat(filename1_a, mdic_1)
        savemat(filename2_a, mdic_2)

    def setUp(self):
        self.tolerance = 1e-15

        # # simple samples of different shapes to assert ErrorRaising
        self.simple_x = np.array([[0, -np.pi, np.pi], [0, -np.pi, np.pi]])
        self.simple_y = np.array([0, -np.pi, np.pi])
        self.simple_z = np.array([0, np.pi, np.pi / 2, -np.pi])

        # # check for ground truth consistency with REAL/ARTIFICIAL LFP-dataset
        # REAL LFP-DATASET
        # Load first & second data file
        filename1_r = os.path.sep.join(['lfp_dataset1.mat'])
        dataset1_r = scipy.io.loadmat(filename1_r, squeeze_me=True)
        filename2_r = os.path.sep.join(['lfp_dataset2.mat'])
        dataset2_r = scipy.io.loadmat(filename2_r, squeeze_me=True)
        # get the relevant values
        self.lfps1_r = dataset1_r['lfp_matrix'] * pq.uV
        self.sf1_r = dataset1_r['sf'] * pq.Hz
        self.lfps2_r = dataset2_r['lfp_matrix'] * pq.uV
        # ARRTIFICIAL LFP-DATASET
        filename1_a = os.path.sep.join(['artificial_LFPs_1.mat'])
        dataset1_a = scipy.io.loadmat(filename1_a, squeeze_me=True)
        filename2_a = os.path.sep.join(['artificial_LFPs_2.mat'])
        dataset2_a = scipy.io.loadmat(filename2_a, squeeze_me=True)
        # get the relevant values
        self.lfps1_a = dataset1_a['lfp_matrix'] * pq.uV
        self.sf1_a = dataset1_a['sf'] * pq.Hz
        self.lfps2_a = dataset2_a['lfp_matrix'] * pq.uV

        # # load ground-truth calculated by:
        # # 1) FieldTrip: ft_connectivity_wpli()
        self.wpli_ground_truth_FieldTrip_REAL = np.loadtxt(
            'ground_truth_WPLI_FieldTrip_real_LFPs.csv', delimiter=',',
            dtype=np.float64)
        self.wpli_ground_truth_FieldTrip_ARTIFICIAL = np.loadtxt(
            'ground_truth_WPLI_FieldTrip_artificial_LFPs_more_complex.csv',
            delimiter=',', dtype=np.float64)
        # # 2) FieldTrip: ft_connectivity(), uses multitaper for FFT
        self.wpli_ground_truth_FieldTrip_ARTIFICIAL_multitaper = np.loadtxt(
            'ground_truth_WPLI_FieldTrip_artificial_LFPs_multitaper.csv',
            delimiter=',', dtype=np.float64)
        # # 3) MNE: spectral_connectivity(), uses multitaper for FFT
        self.wpli_ground_truth_MNE_ARTIFICIAL_multitaper = np.loadtxt(
            'ground_truth_WPLI_MNE_spectral_connectivity_multitaper.csv',
            delimiter=',', dtype=np.float64)

    def test_WPLI_ground_truth_consistency_REAL_LFP_dataset(self):
        """
        Test if the WPLI is consistent with the ground truth generated from
        the LFP-datasets from the ICN lecture 10: 'The Local Field Potential'.
        """
        # REAL DATA
        freq_from_lfp_dataset_r, wpli_from_lfp_dataset_r = \
            elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_r, self.lfps2_r, self.sf1_r)
        mask = ~(np.isnan(wpli_from_lfp_dataset_r) |
                 np.isnan(self.wpli_ground_truth_FieldTrip_REAL))
        np.testing.assert_allclose(
            wpli_from_lfp_dataset_r[mask],
            self.wpli_ground_truth_FieldTrip_REAL[mask], atol=1e-14,
            rtol=1e-12)

    def test_WPLI_ground_truth_consistency_ARTIFICIAL_LFP_dataset(self):
        """
        Test if the WPLI is consistent with the ground truth generated from
        artificial LFP-datasets.
        """
        # ARTIFICIAL DATA
        freq_from_lfp_dataset_a, wpli_from_lfp_dataset_a = \
            elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_a, self.lfps2_a, self.sf1_a, absolute_value=False)
        mask = ~(np.isnan(wpli_from_lfp_dataset_a) |
                 np.isnan(self.wpli_ground_truth_FieldTrip_ARTIFICIAL))
        np.testing.assert_allclose(
            wpli_from_lfp_dataset_a[mask],
            self.wpli_ground_truth_FieldTrip_ARTIFICIAL[mask], atol=1e-14,
            rtol=1e-12)

    def test_WPLI_comparison_to_multitaper_approaches(self):
        """
        Test if WPLI values are equal to those calculated from
        FieldTrips' ft_connectivity() and MNEs' spectral_connectivity() at
        frequencies [16, 36, 52, 70, 100]Hz.
        """
        freq, wpli, = elephant.phase_analysis.weighted_phase_lag_index(
            self.lfps1_a, self.lfps2_a, self.sf1_a, absolute_value=False)
        # mask for supposed wpli=1 frequencies,
        # freq=70Hz with supposed wpli=0 is treated separately,
        # because of noise existence in the artificial dataset
        mask = ((freq == 16) | (freq == 36) | (freq == 52) | (freq == 100))
        # comparing to FieldTrips' ft_conectivity()
        np.testing.assert_allclose(wpli[mask],
            self.wpli_ground_truth_FieldTrip_ARTIFICIAL_multitaper[mask],
            atol=self.tolerance, rtol=self.tolerance)
        np.testing.assert_allclose(wpli[freq == 70],
            self.wpli_ground_truth_FieldTrip_ARTIFICIAL_multitaper[freq == 70],
                                   atol=0.0002, rtol=self.tolerance)
        # comparing to MNEs' spectral_connectivity()
        np.testing.assert_allclose(abs(wpli[mask]),
            self.wpli_ground_truth_MNE_ARTIFICIAL_multitaper[mask],
            atol=self.tolerance, rtol=self.tolerance)
        np.testing.assert_allclose(abs(wpli[freq == 70]),
            self.wpli_ground_truth_MNE_ARTIFICIAL_multitaper[freq == 70],
            atol=0.002, rtol=self.tolerance)

    def test_WPLI_is_zero(self):  # for: f = 70Hz
        """
        Test if WPLI is zero at frequency f=70Hz for the multi-sine
        artificial LFP dataset.
        """
        freq, wpli, = elephant.phase_analysis.weighted_phase_lag_index(
            self.lfps1_a, self.lfps2_a, self.sf1_a, absolute_value=False)
        np.testing.assert_allclose(wpli[freq == 70], 0, atol=0.002,
                                   rtol=self.tolerance)

    def test_WPLI_is_one(self):  # for: f = 16Hz and 36Hz
        """
        Test if WPLI is one at frequency f=16Hz and 36Hz for the multi-sine
        artificial LFP dataset.
        """
        freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
            self.lfps1_a, self.lfps2_a, self.sf1_a, absolute_value=False)
        mask = ((freq == 16) | (freq == 36))
        np.testing.assert_allclose(wpli[mask], 1, atol=self.tolerance,
                                   rtol=self.tolerance)

    def test_WPLI_is_minus_one(self):  # for: f = 52Hz and 100Hz
        """
        Test if WPLI is minus one at frequency f=52Hz and 100Hz
        for the multi-sine artificial LFP dataset.
        """
        freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
            self.lfps1_a, self.lfps2_a, self.sf1_a, absolute_value=False)
        mask = ((freq == 52) | (freq == 100))
        np.testing.assert_allclose(wpli[mask], -1, atol=self.tolerance,
                                   rtol=self.tolerance)

    def test_WPLI_raise_error_if_trial_number_is_different(self):
        """
        Test if a ValueError is raised, when the signals have different
        number of trails.
        """
        # different numbers of trails
        np.testing.assert_raises(
            ValueError, elephant.phase_analysis.weighted_phase_lag_index,
            self.simple_x, self.simple_y, 250)

    def test_WPLI_raise_error_if_trial_lengths_are_different(self):
        """
        Test if a ValueError is raised, when within a trail-pair of the signals
        the trial-lengths are different.
        """
        # different lengths in a trail pair
        np.testing.assert_raises(
            ValueError, elephant.phase_analysis.weighted_phase_lag_index,
            self.simple_y, self.simple_z, 250)


if __name__ == '__main__':
    unittest.main()
