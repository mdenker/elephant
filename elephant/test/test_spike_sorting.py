# -*- coding: utf-8 -*-
"""
Tests for the function sta module

:copyright: Copyright 2015-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import math
import numpy as np
from neo import Block, Segment, AnalogSignal, SpikeTrain
import quantities as pq
from elephant.spike_sorting import *

class SpikeSorterTestCase(object):

    def setUp(self):
        self.asiga0 = AnalogSignal(np.array([
            np.sin(np.arange(0, 20 * math.pi, 0.1))]).T,
            units='mV', sampling_rate=10 / pq.ms,
            channel_id=0)
        self.asiga1 = AnalogSignal(np.array([
            np.sin(np.arange(0, 20 * math.pi, 0.1)),
            np.cos(np.arange(0, 20 * math.pi, 0.1))]).T,
            units='mV', sampling_rate=10 / pq.ms,
            channel_id=1)
        self.asiga2 = AnalogSignal(np.array([
            np.sin(np.arange(0, 20 * math.pi, 0.1)),
            np.cos(np.arange(0, 20 * math.pi, 0.1)),
            np.tan(np.arange(0, 20 * math.pi, 0.1))]).T,
            units='mV', sampling_rate=10 / pq.ms,
            channel_id=2)

        self.block = Block()
        self.block.segments.append(Segment())
        self.block.segments[0].analogsignals.extend([self.asiga0, self.asiga1,
                                                     self.asiga2])

        self.block.create_relationship()

    # These tests need to be implemented for each subclass separately
    def test_spiketrains_exist(self):
        raise NotImplementedError()

    def test_spiketrain_relations(self):
        raise NotImplementedError()


class SpikeExtrationTestCase(SpikeSorterTestCase, unittest.TestCase):

    def setUp(self):
        super(SpikeExtrationTestCase, self).setUp()

    def default_sorting(self):
        extractor = SpikeExtractor()
        extractor.sort_block(self.block)

    def test_spiketrains_exist(self):
        self.default_sorting()
        self.assertEqual(len(self.block.segments[0].spiketrains), 6)

    def test_traincount(self):
        self.default_sorting()
        self.assertEqual(len(self.block.segments[0].spiketrains),
                         sum([a.shape[-1] for a in
                              self.block.segments[0].analogsignals]))

    def test_spiketrain_relations(self):
        self.default_sorting()
        self.assertEqual(len(self.block.channel_indexes),1)
        sts = self.block.segments[0].spiketrains
        chidx = self.block.channel_indexes[0]
        anasigs = self.block.segments[0].analogsignals
        self.assertEqual(len(chidx.units),
                         len(self.block.segments[0].analogsignals))
        for unit_idx in range(len(chidx.units)):
            self.assertEqual(len(chidx.units[unit_idx].spiketrains),
                             anasigs[unit_idx].shape[-1])
        for st in sts:
            self.assertTrue(st.unit is not None)

    def test_spiketrain_annotations(self):
        self.default_sorting()
        st = self.block.segments[0].spiketrains[0]
        self.assertTrue('sorting_hash' in st.annotations)
        self.assertTrue('sorter' in st.annotations)
        self.assertTrue(st.annotations['sorting_hash'] is not None)
        self.assertTrue(st.annotations['sorter'] == 'SpikeExtractor')

    def test_spike_times(self):
        self.default_sorting()
        expected_times = [np.array([0.0016, 0.0079, 0.0141, 0.0204, 0.0267,
                                    0.033 , 0.0393, 0.0456, 0.0518, 0.0581]),
                          np.array([0., 0.0063, 0.0126, 0.0188, 0.0251, 0.0314,
                                    0.0377, 0.044 , 0.0503, 0.0565, 0.0628]),
                          np.array([0.0015, 0.0047, 0.0078, 0.0109, 0.0141,
                                    0.0172, 0.0204, 0.0235, 0.0267, 0.0298,
                                    0.0329, 0.0361, 0.0392, 0.0424, 0.0455,
                                    0.0486, 0.0518, 0.0549, 0.0581, 0.0612])]
        sts = self.block.segments[0].spiketrains

        sts_id = 0
        for asig_id in range(3):
            for exp_id in range(asig_id+1):
                np.testing.assert_array_almost_equal(sts[sts_id].magnitude,
                                                     expected_times[exp_id])
                sts_id += 1


    def test_parameter_conservation(self):
        extractor = SpikeExtractor(filter_high=1*pq.Hz, filter_low=None)
        hash_before = extractor.sorting_hash

        extractor.sort_block(self.block)
        hash_after = extractor.sorting_hash

        self.assertEqual(hash_before, hash_after)

    def test_hash_uses_keys(self):
        parameters1 = {'a': 1, 'b': 1 * pq.m, 'c': [1, 2, 3]}
        parameters2 = {'a': 5, 'b': 1 * pq.mm, 'c': [1, 2, 3, 4]}

        hash1 = SpikeSorter.get_sorting_hash(parameters1)
        hash2 = SpikeSorter.get_sorting_hash(parameters2)

        self.assertNotEqual(hash1, hash2)

    def test_hash_is_reproducible(self):
        parameters1 = {'a': 1, 'b': 1 * pq.m, 'c': [1, 2, 3]}
        parameters2 = copy.deepcopy(parameters1)

        hash1 = SpikeSorter.get_sorting_hash(parameters1)
        hash2 = SpikeSorter.get_sorting_hash(parameters2)

        self.assertEqual(hash1, hash2)


class KMeansSorterTestCase(SpikeSorterTestCase, unittest.TestCase):
    def setUp(self):
        super(KMeansSorterTestCase, self).setUp()

        self.st0 = SpikeTrain(np.arange(10)*pq.s, t_start=0*pq.s,
                              t_stop=10*pq.s, sampling_rate=1*pq.kHz)
        waveforms = np.sin(np.arange(0, 20*math.pi, 0.02*math.pi))
        waveforms = waveforms.reshape((10, -1))*pq.V
        waveforms = waveforms[:, np.newaxis, :]
        self.st0.waveforms = waveforms
        self.st0.left_sweep = -2*pq.ms

        self.chidx = neo.ChannelIndex([0])
        self.unit0 = neo.Unit(name='unit0', channel_id=0)
        self.block.channel_indexes.append(self.chidx)
        self.chidx.units.append(self.unit0)
        self.unit0.spiketrains.append(self.st0)
        self.block.segments[0].spiketrains.append(self.st0)
        self.block.create_relationship()

    def default_sorting(self):
        sorter = KMeansSorter(n_clusters=3)
        sorter.sort_spiketrain(self.st0)

    def test_spiketrains_exist(self):
        self.default_sorting()
        self.assertEqual(len(self.block.segments[0].spiketrains), 3+1)

    def test_spiketrain_relations(self):
        self.default_sorting()
        self.assertEqual(len(self.block.channel_indexes), 2)
        sorted_chidx = self.block.channel_indexes[-1]
        self.assertEqual(len(sorted_chidx.units), 3)
        for u in sorted_chidx.units:
            self.assertEqual(len(u.spiketrains), 1)
            self.assertTrue(u.spiketrains[0].unit is u)





