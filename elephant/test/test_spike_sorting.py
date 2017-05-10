# -*- coding: utf-8 -*-
"""
Tests for the function sta module

:copyright: Copyright 2015-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import math
import numpy as np
import scipy
from numpy.testing import assert_array_equal
from numpy.testing.utils import assert_array_almost_equal
import neo
from neo import Block, Segment, AnalogSignal, SpikeTrain
import quantities as pq
from quantities import ms, mV, Hz
from elephant.spike_sorting import SpikeExtractor
import warnings

class spike_extraction_TestCase(unittest.TestCase):

    def setUp(self):
        self.asiga0 = AnalogSignal(np.array([
            np.sin(np.arange(0, 20 * math.pi, 0.1))]).T,
            units='mV', sampling_rate=10 / ms,
            channel_id=0)
        self.asiga1 = AnalogSignal(np.array([
            np.sin(np.arange(0, 20 * math.pi, 0.1)),
            np.cos(np.arange(0, 20 * math.pi, 0.1))]).T,
            units='mV', sampling_rate=10 / ms,
            channel_id=1)
        self.asiga2 = AnalogSignal(np.array([
            np.sin(np.arange(0, 20 * math.pi, 0.1)),
            np.cos(np.arange(0, 20 * math.pi, 0.1)),
            np.tan(np.arange(0, 20 * math.pi, 0.1))]).T,
            units='mV', sampling_rate=10 / ms,
            channel_id=2)

        self.block = Block()
        self.block.segments.append(Segment())
        self.block.segments[0].analogsignals.extend([self.asiga0, self.asiga1,
                                                     self.asiga2])

    #***********************************************************************
    #************************ Test for typical values **********************

    def Test_SpikeExtraction(self):

        extractor = SpikeExtractor({})

        extractor.sort_block(self.block)
        self.assertDictEqual(len(self.block.segments[0].spiketrains),
                             len(self.block.segments[0].analogsignals))

        # const = 13.8
        # x = const * np.ones(201)
        # asiga = AnalogSignal(
        #     np.array([x]).T, units='mV', sampling_rate=10 / ms)
        # st = SpikeTrain([3, 5.6, 7, 7.1, 16, 16.3], units='ms', t_stop=20)
        # window_starttime = -2 * ms
        # window_endtime = 2 * ms
        # STA = sta.spike_triggered_average(
        #     asiga, st, (window_starttime, window_endtime))
        # a = int(((window_endtime - window_starttime) *
        #         asiga.sampling_rate).simplified)
        # cutout = asiga[0: a]
        # cutout.t_start = window_starttime
        # assert_array_almost_equal(STA, cutout, 12)
