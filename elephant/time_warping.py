# -*- coding: utf-8 -*-
"""
Time warping 

Background
----------

- comparison to traditional trigger alignment

Functions overview
------------------

.. autosummary::
    :toctree: toctree/time_warping/

:copyright: Copyright 2015-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import warnings
from __future__ import division, print_function, unicode_literals

import neo
from neo import utils
import numpy as np
import quantities as pq
import copy

__all__ = [
    "warp_sequence_of_time_points"
    "warp_spiketrain_by_knots",
    "warp_list_of_spiketrains_by_knots",
    "warp_event_by_knots",
    "warp_list_of_events_by_knots",
    "warp_epoch_by_knots",
    "warp_list_of_epochs_by_knots",
    "warp_analogsingal_by_knots",
    "warp_list_of_analogsignals_by_knots",
    "warp_segment_by_events",
]

# TODO documentation


def warp_sequence_of_time_points(sequence_of_time_points,
                                 original_time_knots,
                                 warping_time_knots):
    """Short summary.

    Inspired from sparsewarp in Alex Williams affinewarp package.

    Parameters
    ----------
    sequence_of_time_points : type
        Description of parameter `sequence_of_time_points`.
    original_time_knots : type
        Description of parameter `original_time_knots`.
    warping_time_knots : type
        Description of parameter `warping_time_knots`.

    Returns
    -------
    type
        Description of returned object.

    """

    n_knots = len(original_time_knots)

    # loop over all spikes
    warped_sequence_of_time_points = []
    for time in sequence_of_time_points:
        for knot_idx in range(n_knots):
            if time <= original_time_knots[knot_idx]:
                break

        if knot_idx == 0:
            x_diff = (original_time_knots[1] - original_time_knots[0])
            y_diff = (warping_time_knots[1] - warping_time_knots[0])
            slope = y_diff / x_diff
        else:
            x_diff = (original_time_knots[knot_idx] -
                      original_time_knots[knot_idx-1])
            y_diff = (warping_time_knots[knot_idx] -
                      warping_time_knots[knot_idx-1])
            slope = y_diff / x_diff

        warped_sequence_of_time_points.append(
            warping_time_knots[knot_idx] +
            slope * (time - original_time_knots[knot_idx]))

    return np.array(warped_sequence_of_time_points).flatten()

# TODO update documentation


def warp_spiketrain_by_knots(spiketrain,
                             original_time_knots,
                             warping_time_knots):
    """Warps a single spiketrains by specifying warping knots.

    This function allows for linear and piecewise linear warping
    of a single spiketrain.
    
    Parameters
    ----------
    spiketrain : type
        Description of parameter `spiketrain`.
    original_time_knots : type
        Description of parameter `original_time_knots`.
    warping_time_knots : type
        Description of parameter `warping_time_knots`.

    Returns
    -------
    warped_spiketrain
        Warped neo.Spiketrain

    """
    warped_spike_times = warp_sequence_of_time_points(spiketrain,
                                                      original_time_knots,
                                                      warping_time_knots)

    warped_spiketrain = neo.SpikeTrain(
        name='Warped Spiketrain',
        times=warped_spike_times,
        t_start=warping_time_knots[0],
        t_stop=warping_time_knots[-1],
        units=spiketrain.units)

    warped_spiketrain.annotate(**copy.copy(spiketrain.annotations))
    warped_spiketrain.annotations.pop('nix_name')

    return warped_spiketrain

# TODO documentation
# TODO merge list of spiketrains and spiketrain warp?


def warp_list_of_spiketrains_by_knots(list_of_spiketrains,
                                      original_time_knots,
                                      warping_time_knots):

    list_of_warped_spiketrains = []
    for st in list_of_spiketrains:
        warped_st = warp_spiketrain_by_knots(st,
                                             original_time_knots,
                                             warping_time_knots)
        list_of_warped_spiketrains.append(warped_st)
    return list_of_warped_spiketrains

# TODO documentation


def warp_event_by_knots(event,
                        original_time_knots,
                        warping_time_knots):
    warped_event_times = warp_sequence_of_time_points(event,
                                                      original_time_knots,
                                                      warping_time_knots)

    warped_event = neo.Event(
        name=f'Warped {event.name}',
        times=warped_event_times,
        labels=event.labels,
        units=event.units)

    warped_event.annotate(**copy.copy(event.annotations))
    warped_event.array_annotate(**copy.copy(event.array_annotations))
    warped_event.annotations.pop('nix_name')

    return warped_event

# TODO documentation
# TODO merge list of spiketrains and spiketrain warp


def warp_list_of_events_by_knots(list_of_events,
                                 original_time_knots,
                                 warping_time_knots):

    list_of_warped_events = []
    for ev in list_of_events:
        warped_ev = warp_event_by_knots(ev,
                                        original_time_knots,
                                        warping_time_knots)
        list_of_warped_events.append(warped_ev)
    return list_of_warped_events

# TODO documentation


def warp_epoch_by_knots(epoch,
                        original_time_knots,
                        warping_time_knots):
    warped_epoch_start_times = warp_sequence_of_time_points(
        epoch,
        original_time_knots,
        warping_time_knots)

    # the warped duration is obtained by warping the original epoch stop
    epoch_stop_time = epoch.times + epoch.durations
    warped_epoch_stop_times = warp_sequence_of_time_points(epoch_stop_time,
                                                           original_time_knots,
                                                           warping_time_knots)

    warped_epoch_durations = warped_epoch_stop_times - warped_epoch_start_times

    warped_epoch = neo.Epoch(
        name=f'Warped {epoch.name}',
        times=warped_epoch_start_times,
        durations=warped_epoch_durations,
        labels=epoch.labels,
        units=epoch.units)

    warped_epoch.annotate(**copy.copy(epoch.annotations))
    warped_epoch.array_annotate(**copy.copy(epoch.array_annotations))
    warped_epoch.annotations.pop('nix_name')
    return warped_epoch

# TODO documentation
# TODO merge list of spiketrains and spiketrain warp


def warp_list_of_epochs_by_knots(list_of_epochs,
                                 original_time_knots,
                                 warping_time_knots):

    list_of_warped_epochs = []
    for ep in list_of_epochs:
        warped_ep = warp_epoch_by_knots(ep,
                                        original_time_knots,
                                        warping_time_knots)
        list_of_warped_epochs.append(warped_ep)
    return list_of_warped_epochs


def warp_analogsignal_by_knots(analogsignal,
                               original_time_knots,
                               warping_time_knots,
                               irregular_signal=False):

    warped_times = warp_sequence_of_time_points(analogsignal.times,
                                                original_time_knots,
                                                warping_time_knots)
    if irregular_signal:
        warped_analogsignal = neo.IrregularlySampledSignal(
            times=warped_times,
            signal=analogsignal,
            units=analogsignal.units,
            time_units=analogsignal.times.units)
    else:
        sampling_rate = 10**-3 * pq.s
        sample_count = int((
            (warping_time_knots[-1] - warping_time_knots[0]
             ).rescale(pq.s) / sampling_rate
        ).magnitude.item())
        print(sample_count)

        warped_analogsignal = neo.IrregularlySampledSignal(
            times=warped_times,
            signal=analogsignal,
            units=analogsignal.units,
            time_units=analogsignal.times.units).resample(
                sample_count=sample_count)

    warped_analogsignal.annotate(**copy.copy(analogsignal.annotations))
    warped_analogsignal.array_annotate(
        **copy.copy(analogsignal.array_annotations))
    warped_analogsignal.annotations.pop('nix_name')
    return warped_analogsignal

# TODO merge list of spiketrains and spiketrain warp


def warp_list_of_analogsignals_by_knots(list_of_analogsignals,
                                        original_time_knots,
                                        warping_time_knots,
                                        irregular_signal=False):

    list_of_warped_analogsignals = []
    for anasig in list_of_analogsignals:
        warped_anasig = warp_analogsignal_by_knots(anasig,
                                                   original_time_knots,
                                                   warping_time_knots,
                                                   irregular_signal)
        list_of_warped_analogsignals.append(warped_anasig)
    return list_of_warped_analogsignals

# TODO write another function for just warping t_stop!


def warp_segment_by_events(
    segment,
    event_name,
    new_events_dictionary,
    new_t_stop,
    irregular_signal=False
):
    """Warp a neo.Segment by specifying (warped) times of events.

    Parameters
    ----------
    segment : neo.Segment
        The neo.Segment that shall be warped.
    event_name : str
        Name of the neo.Event that `segment` contains and that the events
        for the `new_events_dictionary` are taken from.
    new_events_dictionary : dict
        The `new_events_dictionary` contains as keys the labels of
        the events that shall be warped. Their corresponding values
        are the new (pre-defined) event times.
        Internally the original event times will be obtained from the
        segment.
    new_t_stop : pq.Quantity
        New stop time of the segment

    Returns
    -------
    original_event_times : list
        List containing the original event times.
    new_event_times : list
        List containing the new event times.
    warped_segment : neo.Segment
        Warped neo.Segment.
    """

    # get original event times
    original_event_times = []
    for label, new_event_time in new_events_dictionary.items():
        # get_events returns a list of neo.Event
        neo_event = utils.get_events(container=segment,
                                     name=event_name,
                                     labels=label)[0]

        # TODO:
        # check that one unique event has been found
        # put this in a warning

        assert(len(neo_event) == 1)
        original_event_times.append(neo_event.times.squeeze())

    new_event_labels = list(new_events_dictionary.keys())
    new_event_times = list(new_events_dictionary.values())

    # start of real time and warp time should be the same (no shift!)
    # TODO allow for shifts
    # TODO check if seg.t_start < spiketrain.t_start is an issue
    original_event_times.insert(0, segment.t_start)
    new_event_times.insert(0, segment.t_start)
    new_event_labels.insert(0, 't_start')

    # TODO put this in a warning
    assert(new_t_stop >= np.max(list(new_events_dictionary.values())))
    if new_t_stop not in new_events_dictionary.values():
        # stop times need to be set explicitely to 
        # fully define the warping function
        original_event_times.append(segment.t_stop)
        new_event_times.append(new_t_stop)
        new_event_labels.append('t_stop')

    assert(len(original_event_times) == len(new_event_times))

    # create a new neo.Segment
    warped_segment = neo.Segment(
        name=f'Warped {segment.name}'
    )
    warped_segment.annotate(**copy.copy(segment.annotations))
    warped_segment.annotate(original_event_times=original_event_times,
                            new_event_times=new_event_times,
                            warped_event_labels=new_event_labels)
    warped_segment.annotations.pop('nix_name')

    warped_spiketrains = warp_list_of_spiketrains_by_knots(
        segment.spiketrains,
        original_event_times,
        new_event_times)
    warped_segment.spiketrains = warped_spiketrains

    warped_events = warp_list_of_events_by_knots(segment.events,
                                                 original_event_times,
                                                 new_event_times)
    warped_segment.events = warped_events

    warped_epochs = warp_list_of_epochs_by_knots(segment.epochs,
                                                 original_event_times,
                                                 new_event_times)
    warped_segment.epochs = warped_epochs

    warped_analogsignals = warp_list_of_analogsignals_by_knots(
        segment.analogsignals,
        original_event_times,
        new_event_times,
        irregular_signal)
    warped_segment.irregularlysampledsignals = warped_analogsignals

    warped_segment.create_relationship()

    return original_event_times, new_event_times, warped_segment


def warp_list_of_segments_by_events(
        list_of_segments,
        event_name,
        new_events_dictionary,
        new_t_stop,
        irregular_signal=False):

    new_block = neo.Block(name='Block warped by events')
    for seg_idx, segment in enumerate(list_of_segments):
        print(f'Currently warping segment {seg_idx}')
        knots_x, knots_y, warped_segment = warp_segment_by_events(
            segment,
            event_name,
            new_events_dictionary,
            new_t_stop,
            irregular_signal)
        new_block.segments.append(warped_segment)

    new_block.create_relationship()
    return new_block
