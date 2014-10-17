# needed for python 3 compatibility
from __future__ import absolute_import
import unittest
import numpy as np
import quantities as pq
from jelephant.core.neofilter import NeoFilter
from neo.core import Block, Segment, AnalogSignal, SpikeTrain, Unit, \
    RecordingChannelGroup, RecordingChannel


class NeoFilterTestCase(unittest.TestCase):
    def setUp(self):
        self.blk1 = Block()
        self.unit = Unit()
        self.rcg = RecordingChannelGroup(name='all channels')
        self.setup_block()
        self.nf = NeoFilter(self.blk1)

    def setup_block(self):
        """
        Initializes same neo.Block for every test function.

        """
        for ind in range(3):
            seg = Segment(name='segment %d' % ind, index=ind + 1)
            a = AnalogSignal(
                [0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7],
                sampling_rate=10 * pq.Hz,
                units='mV')
            st = SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
                            t_stop=10.0 * pq.s)
            chan = RecordingChannel(index=ind)
            self.rcg.recordingchannels.append(chan)
            chan.recordingchannelgroups.append(self.rcg)
            chan.analogsignals.append(a)
            chan.analogsignals.append(a)
            chan.analogsignals.append(a)
            a.recordingchannel = chan
            seg.analogsignals.append(a)
            seg.analogsignals.append(a)
            seg.analogsignals.append(a)
            st.unit = self.unit
            seg.spiketrains.append(st)
            seg.spiketrains.append(st)
            seg.spiketrains.append(st)
            self.unit.spiketrains.append(st)
            self.blk1.segments.append(seg)
        # Append
        self.unit.block = self.blk1
        self.rcg.units.append(self.unit)
        self.blk1.recordingchannelgroups.append(self.rcg)
        self.blk1.create_relationship()

    ############# Test Cases ####################
    def test_conditions_spiketrains(self):
        # At least n trains
        self.nf.set_conditions(at_least_n_trains=(True, {'n': 2}))
        self.assertTrue(np.array_equal(self.nf.filtered,
                                       self.blk1.list_children_by_class(
                                           'SpikeTrain')))
        # Exact n trains
        self.nf.reset_conditions()
        self.nf.set_conditions(exact_n_trains=(True, {'n': 3})),
        self.assertTrue(np.array_equal(self.nf.filtered,
                                       self.blk1.list_children_by_class(
                                           'SpikeTrain')))
        # At least n spikes
        self.nf.reset_conditions()
        self.nf.set_conditions(each_train_has_n_spikes=(True, {'n': 3}))
        self.assertTrue(np.array_equal(self.nf.filtered,
                                       self.blk1.list_children_by_class(
                                           'SpikeTrain')))
        # Exact n spikes
        self.nf.reset_conditions()
        self.nf.set_conditions(each_train_exact_n_spikes=(True, {'n': 7}))
        self.assertTrue(np.array_equal(self.nf.filtered,
                                       self.blk1.list_children_by_class(
                                           'SpikeTrain')))

    def test_conditions_spiketrains_failure(self):
        # At least n trains
        self.nf.set_conditions(at_least_n_trains=(True, {'n': 10}))
        self.assertFalse(np.array_equal(self.nf.filtered,
                                        self.blk1.list_children_by_class(
                                            'SpikeTrain')))
        # Exact n trains
        self.nf.reset_conditions()
        self.nf.set_conditions(exact_n_trains=(True, {'n': 1}))
        self.assertFalse(np.array_equal(self.nf.filtered,
                                        self.blk1.list_children_by_class(
                                            'SpikeTrain')))

        # At least n spikes
        self.nf.reset_conditions()
        self.nf.set_conditions(each_train_exact_n_spikes=(True, {'n': 2}))
        self.assertEqual(self.nf.filtered, [])

        self.nf.reset_conditions()
        self.assertRaises(ValueError, self.nf.set_conditions,
                          at_least_n_trains=(True, {'n': 0}))

        #Test exception statements
        self.nf.reset_conditions()
        self.assertRaises(ValueError, self.nf.set_conditions,
                          exact_n_trains=(True, {'n': 0}))

        self.nf.reset_conditions()
        self.assertRaises(ValueError, self.nf.set_conditions,
                          each_train_has_n_spikes=(True, {'n': 0}))

        self.nf.reset_conditions()
        self.assertRaises(ValueError, self.nf.set_conditions,
                          each_train_exact_n_spikes=(True, {'n': 0}))

    def test_conditions_analogsignals(self):
        # At least n signals
        self.nf.set_conditions(at_least_n_analogsignals=(True, {'n': 2}))
        self.assertTrue(np.array_equal(self.nf.filtered,
                                       self.blk1.list_children_by_class(
                                           'AnalogSignal')))

        # Exact n signals
        self.nf.reset_conditions()
        self.nf.set_conditions(exact_n_analogsignals=(True, {'n': 3}))
        self.assertTrue(np.array_equal(self.nf.filtered,
                                       self.blk1.list_children_by_class(
                                           'AnalogSignal')))

    def test_conditions_analogsignals_failure(self):
        # At least n signals
        self.nf.set_conditions(at_least_n_analogsignals=(True, {'n': 10}))
        self.assertFalse(len(self.nf.filtered),
                         len(self.blk1.list_children_by_class('AnalogSignal')))

        # Exact n signals
        self.nf.reset_conditions()
        self.nf.set_conditions(exact_n_analogsignals=(True, {'n': 5})),
        self.assertFalse(np.array_equal(self.nf.filtered,
                                        self.blk1.list_children_by_class(
                                            'AnalogSignal')))

        # Test exceptions
        self.nf.reset_conditions()
        self.assertRaises(ValueError, self.nf.set_conditions,
                          at_least_n_analogsignals=(True, {'n': 0}))
        self.nf.reset_conditions()
        self.assertRaises(ValueError, self.nf.set_conditions,
                          exact_n_analogsignals=(True, {'n': 0}))

    def test_conditions_recording_channel(self):
        # At least n rc
        self.nf.set_conditions(at_least_n_recordingchannels=(True, {'n': 2}))
        self.assertEqual(len(self.nf.filtered), 3)

        # Exact n rc
        self.nf.reset_conditions()
        self.nf.set_conditions(exact_n_recordingchannels=(True, {'n': 3}))
        self.assertEqual(len(self.nf.filtered), 3)

        # Contains each rc
        self.nf.reset_conditions()
        self.nf.set_conditions(contains_each_recordingchannel=(True, ))
        self.assertTrue(np.array_equal(self.nf.filtered,
                                       self.blk1.list_children_by_class(
                                           'AnalogSignal')))

    def test_conditions_recording_channel_failure(self):
        # At least n signals
        self.nf.set_conditions(at_least_n_recordingchannels=(True, {'n': 10}))
        self.assertFalse(len(self.nf.filtered),
                         len(self.blk1.list_children_by_class(
                             'RecordingChannel')))

        self.nf.reset_conditions()
        self.nf.set_conditions(exact_n_recordingchannels=(True, {'n': 10}))
        self.assertFalse(len(self.nf.filtered),
                         len(self.blk1.list_children_by_class(
                             'RecordingChannel')))

        # Test exceptions
        self.nf.reset_conditions()
        self.assertRaises(ValueError, self.nf.set_conditions,
                          at_least_n_recordingchannels=(True, {'n': 0}))
        self.nf.reset_conditions()
        self.assertRaises(ValueError, self.nf.set_conditions,
                          exact_n_recordingchannels=(True, {'n': 0}))

    def test_conditions_unit(self):
        # At least n signals
        self.nf.set_conditions(at_least_n_units=(True, {'n': 1}))
        self.assertTrue(np.array_equal(self.nf.filtered,
                                       self.blk1.list_children_by_class(
                                           'Unit')))

        # Exact n units
        self.nf.set_conditions(exact_n_units=(True, {'n': 1}))
        self.assertTrue(np.array_equal(self.nf.filtered,
                                       self.blk1.list_children_by_class(
                                           'Unit')))

        # Contains each unit
        self.nf.reset_conditions()
        self.nf.set_conditions(contains_each_unit=(True, ))
        self.assertTrue(np.array_equal(self.nf.filtered,
                                       self.blk1.list_children_by_class(
                                           'SpikeTrain')))

    def test_conditions_unit_failure(self):
        # At least n signals
        self.nf.set_conditions(
            at_least_n_units=(True, {'n': 10}))
        self.assertFalse(len(self.nf.filtered),
                         len(self.blk1.list_children_by_class(
                             'unit')))

        self.nf.reset_conditions()
        self.nf.set_conditions(exact_n_units=(True, {'n': 10}))
        self.assertFalse(len(self.nf.filtered),
                         len(self.blk1.list_children_by_class(
                             'unit')))

        # Test exceptions
        self.nf.reset_conditions()
        self.assertRaises(ValueError, self.nf.set_conditions,
                          at_least_n_units=(True, {'n': 0}))
        self.nf.reset_conditions()
        self.assertRaises(ValueError, self.nf.set_conditions,
                          exact_n_units=(True, {'n': 0}))

    def test_conditions_data_aligned(self):
        self.nf.set_conditions(data_aligned=(True, ))
        sigs = self.blk1.list_children_by_class('AnalogSignal')
        sts = self.blk1.list_children_by_class('SpikeTrain')
        sigs.extend(sts)
        self.assertTrue(np.array_equal(self.nf.filtered, sigs))

        # Create new block with just spiketrains and different start, stops
        blk2 = Block()
        blk2.segments.append(Segment())
        blk2.segments[0].spiketrains.append(SpikeTrain(
            [0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=7.0 * pq.s))
        blk2.segments[0].spiketrains.append(SpikeTrain(
            [1.2, 1.6, 2.3, 3.1, 4.3, 5.5, 7.7] * pq.s, t_start=1.0*pq.s,
            t_stop=8.0 * pq.s))
        blk2.segments[0].spiketrains.append(SpikeTrain(
            [2.1, 2.2, 2.3, 3.1, 4.3, 6.5, 8.7] * pq.s, t_start=2.0*pq.s,
            t_stop=9.0 * pq.s))
        nf2 = NeoFilter(blk2)
        nf2.set_conditions(data_aligned=(True, ))
        self.assertTrue(np.array_equal(nf2.filtered, []))

        # A list of analogsignals
        nf2 = NeoFilter(self.blk1.list_children_by_class('AnalogSignal'))
        nf2.set_conditions(data_aligned=(True, ))
        self.assertTrue(np.array_equal(nf2.filtered,
                                       self.blk1.list_children_by_class(
                                           'AnalogSignal')))

        # A list of spiketrains
        nf2 = NeoFilter(self.blk1.list_children_by_class('SpikeTrain'))
        nf2.set_conditions(data_aligned=(True, ))
        self.assertTrue(np.array_equal(nf2.filtered,
                                       self.blk1.list_children_by_class(
                                           'SpikeTrain')))

        # Block with analogsignals and different starts
        blk3 = Block()
        blk3.segments.append(Segment())
        blk3.segments[0].analogsignals.append(
            AnalogSignal(np.array([1.0, 2.0, 3.0]), t_start=50 * pq.ms,
                         sampling_rate=0.42 * pq.kHz, units=pq.uV))
        blk3.segments[0].analogsignals.append(
            AnalogSignal(np.array([1.0, 2.0, 3.0]), t_start=100 * pq.ms,
                         sampling_rate=0.42 * pq.kHz, units=pq.uV))
        blk3.segments[0].analogsignals.append(
            AnalogSignal(np.array([1.0, 2.0, 3.0]), t_start=200 * pq.ms,
                         sampling_rate=0.42 * pq.kHz, units=pq.uV))
        nf2 = NeoFilter(blk3)
        self.assertEqual(nf2.filtered, [])

        # Test out an empty Block
        nf2 = NeoFilter(Block())
        nf2.set_conditions(data_aligned=(True, ))
        self.assertEqual(nf2.filtered, [])

        # Test out a 'user' function with an empty block
        from jelephant.core.conditions import data_aligned
        nf2 = NeoFilter(blk2, data_aligned)
        self.assertEqual(nf2.filtered, [])

        # Test out 'user' function with non empty Block
        nf2 = NeoFilter(self.blk1)
        nf2.set_conditions(data_aligned=(True, ))
        self.assertTrue(np.array_equal(nf2.filtered, sigs))

    def test_no_overlap(self):
        self.nf.set_conditions(
            signals_no_overlap=(True, {'take_first': False}))
        self.assertEqual(self.nf.filtered, [])

        # Positive test
        blk = Block()
        j = 0
        for i in range(3):
            seg = Segment(name='segment %d' % i, index=i)
            s = np.array([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7])
            st = SpikeTrain((s + j) * pq.s,
                            t_start=j * pq.s, t_stop=(j + 10.0) * pq.s)
            seg.spiketrains.append(st)
            blk.segments.append(seg)
            j += 10
        nfilt = NeoFilter(blk)
        nfilt.set_conditions(
            signals_no_overlap=(True, {'take_first': False}))
        self.assertEqual(nfilt.filtered, [0, 1, 2])
        # test with take first element if there is overlap
        nfilt.reset_conditions()
        nfilt.set_conditions(signals_no_overlap=(True, {'take_first': True}))
        self.assertEqual(nfilt.filtered, [0, 1, 2])

        # Another negative case
        # ################################
        # # Structure of signals        #
        # # 0---                        #
        # #   1---                      #
        # #    2---                     #
        # #        3---                 #
        # # Only signal 3 has no overlap#
        #################################
        block = Block()
        seg1 = Segment()
        seg2 = Segment()
        seg3 = Segment()
        seg4 = Segment()
        s = np.array([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7])
        st1 = SpikeTrain(s * pq.s, t_start=0 * pq.s, t_stop=10.0 * pq.s)
        st2 = SpikeTrain((s + 7) * pq.s, t_start=7 * pq.s, t_stop=17.0 * pq.s)
        st3 = SpikeTrain((s + 11) * pq.s, t_start=11 * pq.s,
                         t_stop=21.0 * pq.s)
        st4 = SpikeTrain((s + 22) * pq.s, t_start=22 * pq.s,
                         t_stop=32.0 * pq.s)
        seg1.spiketrains.append(st1)
        seg2.spiketrains.append(st2)
        seg3.spiketrains.append(st3)
        seg4.spiketrains.append(st4)
        block.segments.append(seg1)
        block.segments.append(seg2)
        block.segments.append(seg3)
        block.segments.append(seg4)
        nfilt2 = NeoFilter(block)
        nfilt2.set_conditions(signals_no_overlap=(True, ))
        self.assertEqual(nfilt2.filtered, [3])

        # test with take first element if there is overlap
        nfilt2.reset_conditions()
        nfilt2.set_conditions(signals_no_overlap=(True, {'take_first': True}))
        self.assertEqual(nfilt2.filtered, [0, 3])

    def test_mixed_conditions(self):
        self.nf.set_conditions(data_aligned=(True, ),
                               at_least_n_trains=(True, {'n': 2}))
        self.assertTrue(np.array_equal(self.nf.filtered,
                                       self.blk1.list_children_by_class(
                                           'SpikeTrain')))

        nfilt = NeoFilter([])
        nfilt.set_conditions(data_aligned=(True, ))
        self.assertEqual(nfilt.filtered, [])

        # One conditions is met the other not
        self.nf.reset_conditions()
        self.nf.set_conditions(
            at_least_n_units=(True, {'n': 1}))
        self.assertTrue(len(self.nf.filtered),
                        len(self.blk1.list_children_by_class(
                            'unit')))
        self.nf.set_conditions(exact_n_units=(True, {'n': 3}))
        self.assertFalse(len(self.nf.filtered),
                         len(self.blk1.list_children_by_class('unit')))

        # Two conditions in one call
        self.nf.reset_conditions()
        self.nf.set_conditions(
            at_least_n_units=(True, {'n': 1}), exact_n_units=(True, {'n': 3}))
        self.assertEqual(self.nf.filtered, [])

    def test_user_function(self):
        def add_one(container):
            import jelephant.core.neo_tools as nt
            spkt = nt.get_all_spiketrains(container)
            return [spt + 1 * spt.units for spt in spkt]
        self.nf.apply_function(add_one)
        sts = [st + 1 * st.units for st in
               self.blk1.list_children_by_class('SpikeTrain')]
        self.assertTrue(np.array_equal(self.nf.filtered, sts))

        self.nf.set_conditions(at_least_n_trains=(True, {'n': 2}))
        self.assertTrue(np.array_equal(self.nf.filtered, sts))

    def test_error_keyword(self):
        self.assertRaises(ValueError, self.nf.set_conditions,
                          at_last_ntrains=(True, {'n': 3}))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(NeoFilterTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)