from __future__ import division
from __future__ import print_function

import copy
import csv
import gc
import inspect
import os
import re
import warnings

import neo
import numpy as np
import quantities as pq


def search_sequence_numpy(arr, seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create 2D array of sliding indices across entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        return True, np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[
            0]
    else:
        return False, np.empty(seq.shape)  # No match found


def decode_events(block, session_folder):
    # seg.t_stop and seg.t_start does not work in combination with proxyobjects
    # due to a bug in neo
    # TODO: check this and open an issue
    # seg_durations = [seg.t_stop - seg.t_start for seg in block.segments]
    # seg = block.segments[np.argmax(seg_durations)]
    seg = block.segments[
        1]  # hard-coded for now until the issue above is fixed
    behavioral_codes = extract_behavioral_codes(session_folder)
    behavioral_codes_inv = {val: key for key, val in behavioral_codes.items()}

    with open(session_folder + '/' + 'descriptor_codes_landing.csv',
              'r') as csvfile:
        code_reader = csv.reader(csvfile)
        next(code_reader)  # skip the header line
        landing_codes = {}
        for row in code_reader:
            landing_codes[row[2]] = row[1]

    events = get_events(seg, name='digital_input_port')[0]
    events.labels = events.labels.astype(int)

    times = []
    labels = []
    switch = 1
    for idx in range(len(events)):
        try:
            beh_code = behavioral_codes_inv[events.labels[idx]]
        except:
            continue
        if 'metadata' in beh_code:
            switch *= -1  # skip all metadata sections
            continue
        elif switch < 0:
            continue
        if beh_code in landing_codes:
            times.append(events.times[idx])
            labels.append(landing_codes[beh_code])

    new_ev = neo.Event(name='DecodedEvents', times=times, labels=labels,
                       units=times[0].units,
                       description='All relevant events of the experiment. '
                                   + 'Labels have been decoded to be human readable.')

    new_ev.segment = seg
    seg.events.append(new_ev)

    # annotate metadata
    metadata = {
        'analogsignal_min': -5000.0 * pq.mV,
        'analogsignal_max': 5000.0 * pq.mV,
        'xbias': 1.345 * pq.cm,
        'ybias': 4.688 * pq.cm,
    }

    # Extract metadata from events
    metadata.update(extract_metadata_from_events(events))

    block.annotate(metadata=metadata)

    return


def add_trial_epochs_to_block(block, session_folder):
    # seg.t_stop and seg.t_start does not work in combination with proxyobjects
    # due to a bug in neo
    # TODO: check this and open an issue
    # seg_durations = [seg.t_stop - seg.t_start for seg in block.segments]
    # seg = block.segments[np.argmax(seg_durations)]
    seg = block.segments[
        1]  # hard-coded for now until the issue above is fixed

    events = get_events(seg, name='digital_input_port')[0]
    event_labels = events.labels.astype(int)
    # Get trials from events
    behavioral_codes = extract_behavioral_codes(session_folder)

    trial_protocols = event_labels[np.where(event_labels ==
                                            behavioral_codes[
                                                'trial_protocol_start'])[
                                       0] + 1]

    trial_starts = get_events(seg, labels=np.array(
        (behavioral_codes['trial_start']),
        dtype=events.labels.dtype))
    trial_ends = get_events(seg,
                            labels=np.array((behavioral_codes['trial_end']),
                                            dtype=events.labels.dtype))

    if len(trial_starts[0]) == len(trial_ends[0]) + 1:
        trial_starts[0] = trial_starts[0][:-1]

    complete_trials = add_epoch(seg, trial_starts[0], trial_ends[0],
                                epoch_category='Complete Trials')

    attempt_starts = get_events(seg,
                                labels=np.array(
                                    (behavioral_codes['target_01_status_1']),
                                    dtype=events.labels.dtype))
    attempt_ends = get_events(seg,
                              labels=np.array(
                                  (behavioral_codes['automatic_reward_end'],
                                   behavioral_codes['anticipation'],
                                   behavioral_codes['time_out']),
                                  dtype=events.labels.dtype))

    if len(attempt_starts[0]) == len(attempt_ends[0]) + 1:
        attempt_starts[0] = attempt_starts[0][:-1]

    all_attempts = add_epoch(seg, attempt_starts[0], attempt_ends[0],
                             epoch_category='All Attempts')

    successful = np.array([np.array((behavioral_codes['automatic_reward_end']),
                                    dtype=events.labels.dtype).item() ==
                           attempt_ends[0].labels[i]
                           for i in range(len(attempt_starts[0]))])

    belongs_to_trial_number = np.array([sum(successful[:i]) + 1
                                        for i in
                                        range(len(attempt_starts[0]))])

    attempts_per_trial = np.bincount(belongs_to_trial_number)[1:]

    attempt_number_in_trial = np.concatenate(
        [np.arange(1, i + 1) for i in attempts_per_trial])

    complete_trials.array_annotate(
        trial_number=np.arange(1, len(trial_starts[0]) + 1),
        trial_protocol=trial_protocols[:-1],
        num_unsuccessful_attempts=attempts_per_trial - 1)

    all_attempts.array_annotate(
        attempt_number=np.arange(1, len(attempt_starts[0]) + 1),
        successful=successful,
        belongs_to_trial_number=belongs_to_trial_number,
        attempt_number_in_trial=attempt_number_in_trial)

    return


def annotate_sua_mua_info(spiketrains, sorting_csv_file):
    sua_mua_info = np.loadtxt(sorting_csv_file, delimiter=',')
    for ch_id, num_sua, id_mua in sua_mua_info:
        for st in spiketrains:
            if st.annotations['channel_id'] == ch_id:
                if st.annotations['unit_id'] == 0:
                    st.annotate(unit_type='noise')
                elif 0 < st.annotations['unit_id'] < id_mua:
                    st.annotate(unit_type='sua')
                elif 0 < st.annotations['unit_id'] < id_mua \
                        or id_mua == 0 < st.annotations['unit_id']:
                    st.annotate(unit_type='sua')
                elif st.annotations['unit_id'] >= id_mua:
                    st.annotate(unit_type='mua')


def extract_sequence_epochs(from_target, to_target, block, session_folder,
                            pre=0 * pq.s, post=0 * pq.s):
    """
            Compute an Epoch containing all time slices during which the monkey performs the
            movement from from_target to to_target. This corresponds to the time between
            to_target onset and to_target offset for which the previous target was from_target.

            Args:
                from_target (int):
                    The starting target of the requested movement.
                to_target (int):
                    The ending target of the requested movement.
                block (neo.Block):
                    The block containing the data to compute the epochs for.
                session_folder (str):
                    The folder containing the csv file with the event code translations.
                pre (Quantity):
                    Time buffer to include before to_target onset. Defaults to 0.
                post (Quantity):
                    Time buffer to include after to_target offset. Defaults to 0.

            Returns:
                epoch
                    neo.Epoch containing the requested epochs

            """

    # use the raw events and translate their codes
    # TODO: segment duration issue, see TODOs above
    events = get_events(block.segments[1], name='digital_input_port')[0]

    behavioral_codes = extract_behavioral_codes(session_folder)

    # get all target sequences
    sequence_dict = extract_sequence_dict(events, behavioral_codes)

    desired_sequence = np.array([from_target, to_target])
    protocols_with_desired_sequence = {
        key: search_sequence_numpy(sequence, desired_sequence)[1]
        for key, sequence in sequence_dict.items() if
        search_sequence_numpy(sequence, desired_sequence)[0]}

    complete_trials = \
        block.filter(epoch_category='Complete Trials', object=neo.Epoch)[0]
    all_attempts = \
        block.filter(epoch_category='All Attempts', object=neo.Epoch)[0]

    trials_with_desired_sequence = complete_trials[
        np.isin(complete_trials.array_annotations['trial_protocol'],
                list(protocols_with_desired_sequence))]

    successful_attempts_with_desired_sequence = all_attempts[
        np.logical_and(all_attempts.array_annotations['successful'],
                       np.isin(all_attempts.array_annotations[
                                   'belongs_to_trial_number'],
                               trials_with_desired_sequence.array_annotations[
                                   'trial_number']))]

    events_in_desired_attempts = [events.time_slice(start, start + dur) for
                                  start, dur in
                                  zip(
                                      successful_attempts_with_desired_sequence.times,
                                      successful_attempts_with_desired_sequence.durations)]

    trial_number = complete_trials.array_annotations['trial_number']

    # TODO: From here the code gets ugly due to long descriptive variable names and complex slicing

    # to_target onset
    # target onset event code is 200x1 where x is the number of the target in the chronological
    # target order of the corresponding trial
    # this number is the second entry of each protocol_with_desired_sequence value
    trigger_starts = [ev[ev.labels == np.array((20001 + 10 * int(
        protocols_with_desired_sequence[
            complete_trials[
                trial_number ==
                successful_attempts_with_desired_sequence[
                    idx].array_annotations[
                    'belongs_to_trial_number']].array_annotations[
                'trial_protocol'][0]][1] + 1)),
                                               dtype=events.labels.dtype).item()]
                      for idx, ev in enumerate(events_in_desired_attempts)]

    # to_target offset
    # target offset event code is 200x0 where x is the number of the target in the chronological
    # target order of the corresponding trial
    # this number is the second entry of each protocol_with_desired_sequence value
    trigger_stops = [ev[ev.labels == np.array((20000 + 10 * int(
        protocols_with_desired_sequence[complete_trials[
            trial_number ==
            successful_attempts_with_desired_sequence[
                idx].array_annotations[
                'belongs_to_trial_number']].array_annotations[
            'trial_protocol'][0]][1] + 1)),
                                              dtype=events.labels.dtype).item()]
                     for idx, ev in enumerate(events_in_desired_attempts)]

    starts = neo.Event(np.array([start.times.magnitude
                                 for start in trigger_starts]).flatten(),
                       units=trigger_starts[0].units,
                       name='Sequence {t1} to {t2} Starts'.format(
                           t1=from_target,
                           t2=to_target))
    ends = neo.Event(np.array([stop.times.magnitude
                               for stop in trigger_stops]).flatten(),
                     units=trigger_stops[0].units,
                     name='Sequence {t1} to {t2} Ends'.format(t1=from_target,
                                                              t2=to_target))

    epoch = add_epoch(block.segments[1], starts, ends, attach_result=False,
                      pre=pre, post=post,
                      name='Sequence {t1} to {t2}'.format(t1=from_target,
                                                          t2=to_target))
    return epoch


def detect_synchrofacts(block, segment='all', n=2, spread=2,
                        sampling_rate=30000. / pq.s, invert=False,
                        delete=False,
                        unit_type='all'):
    """
    Given block with spike trains, find all spikes engaged
    in synchronous events of size *n* or higher. Two events are considered
    synchronous if they occur within spread/sampling_rate of one another.

    *Args*
    ------
    block [list]:
        a block containing neo spike trains

    segment [int or iterable or str. Default: 1]:
        indices of segments in the block. Can be an integer, an iterable object
        or a string containing 'all'. Indicates on which segments of the block
        the synchrofact removal should be performed.

    n [int. Default: 2]:
        minimum number of coincident spikes to report synchrony

    spread [int. Default: 2]:
        number of bins of size 1/sampling_rate in which to check for synchronous spikes.
        *n* spikes within *spread* consecutive bins are considered synchronous.

    sampling_rate [quantity. Default: 30000/s]:
        Sampling rate of the spike trains. The spike trains are binned with binsize
        dt = 1/sampling_rate and *n* spikes within *spread* consecutive bins are
        considered synchronous.
        Groups of *n* or more synchronous spikes are deleted/annotated.

    invert [bool. Default: True]:
        invert the mask for annotation/deletion (Default:False). False annotates
        synchrofacts with False and other spikes with True / deletes everything
        except for synchrofacts for delete = True.

    delete [bool. Default: False]:
        delete spikes engaged in synchronous activity. If set to False the spiketrains are
        array-annotated and the spike times are kept unchanged.

    unit_type [list of strings. Default 'all']:
        selects only spiketrain of certain units / channels for synchrofact extraction.
        unit_type = 'all' considers all provided spiketrains
        Accepted unit types: 'sua','mua','idx' (where x is the id number requested)
    """

    if isinstance(segment, str):
        if 'all' in segment.lower():
            segment = range(len(block.segments))
        else:
            raise ValueError('Input parameter segment not understood.')

    elif isinstance(segment, int):
        segment = [segment]

    binsize = (spread / sampling_rate).rescale(
        pq.s)  # make sure all quantities have units s

    for seg in segment:
        # data check
        if len(block.segments[seg].spiketrains) == 0:
            warnings.warn(
                'Segment {0} does not contain any spiketrains!'.format(seg))
            continue

        neo_spiketrains, index = [], []

        # considering all spiketrains for unit_type == 'all'
        if isinstance(unit_type, str):
            if 'all' in unit_type.lower():
                # make sure all spiketrains have units s
                neo_spiketrains = [s.copy().rescale(pq.s) for s in
                                   block.segments[seg].spiketrains]
                index = range(len(block.segments[seg].spiketrains))

        else:
            # extracting spiketrains which should be used for synchrofact extraction
            # based on given unit type
            # possible improvement by using masks for different conditions
            # and adding them up
            for i in range(len(block.segments[seg].spiketrains)):
                take_it = False
                for utype in unit_type:
                    if utype[:2] == 'id' and \
                            block.segments[seg].spiketrains[i].annotations[
                                'unit_id'] == int(
                        utype[2:]):
                        take_it = True
                    elif (utype == 'sua' or utype == 'mua') and utype in \
                            block.segments[seg].spiketrains[
                                i].annotations and \
                            block.segments[seg].spiketrains[i].annotations[
                                utype]:
                        take_it = True
                if take_it:
                    # make sure all spiketrains have units s
                    neo_spiketrains.append(
                        block.segments[seg].spiketrains[i].copy().rescale(
                            pq.s))
                    index.append(i)

        # if no spiketrains were selected
        if len(neo_spiketrains) == 0:
            warnings.warn(
                'No matching spike trains for given unit selection criteria %s found' % unit_type)
            # we can skip to the next segment immediately since there are no spiketrains
            # to perform synchrofact detection on
            continue
        else:
            # find times of synchrony of size >=n
            bins_left = detect_syn_spikes(neo_spiketrains, n=n, spread=spread,
                                          sampling_rate=sampling_rate)
            # hstack only works for input != [], bins_left have implicit units s
            # boundaries of half-open intervals, all spikes within those intervals will be removed
            boundaries = np.hstack([(left_edge, left_edge + binsize.magnitude)
                                    for left_edge in bins_left]) if len(
                bins_left) else []
            if spread > 1 and len(bins_left):
                # find overlapping intervals
                # always keep the first and last entries
                mask = np.ones(boundaries.shape, np.bool)
                # reject entries with diff<0 and entries right before the ones with diff<0
                # this effectively merges overlapping intervals
                mask[1:-1] = np.all(
                    np.array([np.diff(boundaries[:-1]) > 0,
                              np.diff(boundaries[1:]) > 0]),
                    axis=0)
                boundaries = boundaries[mask]

        # j = index of pre-selected sts in neo_spiketrains
        # idx = index of pre-selected sts in original block.segments[seg].spiketrains
        for j, idx in enumerate(index):

            # all indices of spikes that are within the half-open intervals defined by the boundaries
            if invert:
                # every second entry in boundaries is an upper boundary
                mask = np.array(np.searchsorted(boundaries, neo_spiketrains[
                    j].times.magnitude, side='right') % 2 == 0,
                                dtype=np.bool)
            else:
                mask = np.array(np.searchsorted(boundaries, neo_spiketrains[
                    j].times.magnitude, side='right') % 2,
                                dtype=np.bool)

            if delete:
                block.segments[seg].spiketrains[idx] = neo_spiketrains[j][
                    np.logical_not(mask)]

            else:
                block.segments[seg].spiketrains[idx].array_annotate(
                    synchrofacts=mask)


def detect_syn_spikes(neo_spiketrains, n=2, spread=2,
                      sampling_rate=30000. / pq.s):
    """
        Given a list of spike trains, returns the left edges
        of the time bins of width *dt* with *n* or more spikes

        *Args*
        ------
        neo_spiketrains [list]:
            a list of neo spike trains

        n [int. Default: 2]:
            minimum number of coincident spikes to report synchrony

        spread [int. Default: 2]:
            number of consecutive bins of size 1/sampling_rate in which
            to check for synchrony

        sampling_rate [quantity. Default: 30000/s]:
            Sampling rate of the spike trains. The spike trains are binned with binsize
            dt = 1/sampling_rate and *n* spikes within *spread* consecutive bins are
            considered synchronous.
            Groups of *n* or more synchronous spikes are deleted/annotated.


        *Returns*
        ---------
        syn_left_edges [numpy array]
            left edges of bins with at least n spikes
        """

    binsize = (spread / sampling_rate).rescale(pq.s)
    offset = (.5 / sampling_rate).rescale(pq.s)

    # we can safely strip the units since we made sure everything has units s
    # most of the numpy functions we use here don't work with pyquantities
    bins = np.arange((neo_spiketrains[0].t_start - offset).magnitude,
                     (neo_spiketrains[0].t_stop + binsize).magnitude,
                     binsize.magnitude)
    # due to a binning issue in elephant.conversion.BinnedSpikeTrain I'm doing this manually for now.
    # We should use the elephant function once it works as intended.

    n_bins = len(bins) - 1
    norm = n_bins / (bins[-1] - bins[0])

    hist = np.zeros(n_bins, dtype=int)
    all_left_edges = bins[:-1]

    for st in neo_spiketrains:
        if len(st):
            spikes = st.times.magnitude

            # analogously to numpy/histograms.py:
            bin_indices = np.array((spikes - bins[0]) * norm, dtype=int)

            # for values that lie exactly on the last_edge we need to subtract one
            bin_indices[bin_indices == n_bins] -= 1

            # The index computation is not guaranteed to give exactly
            # consistent results within ~1 ULP of the bin edges.
            decrement = spikes < bins[bin_indices]
            bin_indices[decrement] -= 1
            # The last bin includes the right edge. The other bins do not.
            increment = ((spikes >= bins[bin_indices + 1])
                         & (bin_indices != n_bins - 1))
            bin_indices[increment] += 1

            hist += np.bincount(bin_indices, minlength=n_bins).astype(np.intp)
    syn_left_edges = all_left_edges[np.where(hist >= n)[0]]

    if spread > 1:
        # with just one binning we miss synchrofacts split by bin edges

        for shift in range(1, spread):
            # redo synchrony search with shifted bins, then merge the results
            # this guarantees we find all synchrofacts of size n
            bins_shifted = np.arange((neo_spiketrains[
                                          0].t_start - offset - shift / spread * binsize).magnitude,
                                     (neo_spiketrains[
                                          0].t_stop + binsize).magnitude,
                                     binsize.magnitude)

            n_bins_shifted = len(bins_shifted) - 1
            norm_shifted = n_bins_shifted / (
                    bins_shifted[-1] - bins_shifted[0])

            hist_shifted = np.zeros(n_bins_shifted, dtype=int)
            all_left_edges_shifted = bins_shifted[:-1]

            for st in neo_spiketrains:
                if len(st):
                    spikes = st.times.magnitude

                    # analogously to numpy/histograms.py:
                    bin_indices = np.array(
                        (spikes - bins_shifted[0]) * norm_shifted, dtype=int)

                    # for values that lie exactly on the last_edge we need to subtract one
                    bin_indices[bin_indices == n_bins_shifted] -= 1

                    # The index computation is not guaranteed to give exactly
                    # consistent results within ~1 ULP of the bin edges.
                    decrement = spikes < bins_shifted[bin_indices]
                    bin_indices[decrement] -= 1
                    # The last bin includes the right edge. The other bins do not.
                    increment = ((spikes >= bins_shifted[bin_indices + 1])
                                 & (bin_indices != n_bins_shifted - 1))
                    bin_indices[increment] += 1

                    hist_shifted += np.bincount(bin_indices,
                                                minlength=n_bins_shifted).astype(
                        np.intp)

            syn_left_edges_shifted = all_left_edges_shifted[
                np.where(hist_shifted >= n)[0]]

            syn_left_edges = np.concatenate((syn_left_edges,
                                             syn_left_edges_shifted))  # merge with previous results

        syn_left_edges.sort(kind='mergesort')  # sort the merged array

    return syn_left_edges  # the result now has implicit units s since we calculate everything in s


# TODO: The following function should be added to elephant.
#  Import it from there once it has been added.


def snr_hatsopoulos(spiketrain):
    """
    :param spiketrain: neo.SpikeTrain
    :return: snr: float signal to noise ratio according to Hatsopoulos 2007
    """

    # average over all waveforms for each bin
    mean_waveform = np.mean(spiketrain.waveforms.magnitude, axis=0)[0]
    # standard deviation over all waveforms for each bin
    std_waveform = np.std(spiketrain.waveforms.magnitude, axis=0)[0]
    mean_std = np.mean(std_waveform)

    # signal
    peak_to_trough_voltage = abs(np.min(mean_waveform) - np.max(mean_waveform))
    # noise
    noise = 2 * mean_std

    if noise == 0:
        return np.nan
    else:
        snr = peak_to_trough_voltage / noise
    return snr


# TODO: The following functions will be included in neo.utils, import that once it's released.


def get_events(container, **properties):
    """
    This function returns a list of Event objects, corresponding to given
    key-value pairs in the attributes or annotations of the Event.

    Parameter:
    ---------
    container: Block or Segment
        The Block or Segment object to extract data from.

    Keyword Arguments:
    ------------------
    The Event properties to filter for.
    Each property name is matched to an attribute or an
    (array-)annotation of the Event. The value of property corresponds
    to a valid entry or a list of valid entries of the attribute or
    (array-)annotation.

    If the value is a list of entries of the same
    length as the number of events in the Event object, the list entries
    are matched to the events in the Event object. The resulting Event
    object contains only those events where the values match up.

    Otherwise, the value is compared to the attribute or (array-)annotation
    of the Event object as such, and depending on the comparison, either the
    complete Event object is returned or not.

    If no keyword arguments is passed, all Event Objects will
    be returned in a list.

    Returns:
    --------
    events: list
        A list of Event objects matching the given criteria.

    Example:
    --------
        >>> event = neo.Event(
                times = [0.5, 10.0, 25.2] * pq.s)
        >>> event.annotate(
                event_type = 'trial start',
                trial_id = [1, 2, 3])
        >>> seg = neo.Segment()
        >>> seg.events = [event]

        # Will return a list with the complete event object
        >>> get_events(seg, properties={'event_type': 'trial start'})

        # Will return an empty list
        >>> get_events(seg, properties={'event_type': 'trial stop'})

        # Will return a list with an Event object, but only with trial 2
        >>> get_events(seg, properties={'trial_id': 2})

        # Will return a list with an Event object, but only with trials 1 and 2
        >>> get_events(seg, properties={'trial_id': [1, 2]})
    """
    if isinstance(container, neo.Segment):
        return _get_from_list(container.events, prop=properties)

    elif isinstance(container, neo.Block):
        event_lst = []
        for seg in container.segments:
            event_lst += _get_from_list(seg.events, prop=properties)
        return event_lst
    else:
        raise TypeError(
            'Container needs to be of type Block or Segment, not %s '
            'in order to extract Events.' % (type(container)))


def get_epochs(container, **properties):
    """
    This function returns a list of Epoch objects, corresponding to given
    key-value pairs in the attributes or annotations of the Epoch.

    Parameters:
    -----------
    container: Block or Segment
        The Block or Segment object to extract data from.

    Keyword Arguments:
    ------------------
    The Epoch properties to filter for.
    Each property name is matched to an attribute or an
    (array-)annotation of the Epoch. The value of property corresponds
    to a valid entry or a list of valid entries of the attribute or
    (array-)annotation.

    If the value is a list of entries of the same
    length as the number of epochs in the Epoch object, the list entries
    are matched to the epochs in the Epoch object. The resulting Epoch
    object contains only those epochs where the values match up.

    Otherwise, the value is compared to the attribute or (array-)annotation
    of the Epoch object as such, and depending on the comparison, either the
    complete Epoch object is returned or not.

    If no keyword arguments is passed, all Epoch Objects will
    be returned in a list.

    Returns:
    --------
    epochs: list
        A list of Epoch objects matching the given criteria.

    Example:
    --------
        >>> epoch = neo.Epoch(
                times = [0.5, 10.0, 25.2] * pq.s,
                durations = [100, 100, 100] * pq.ms)
        >>> epoch.annotate(
                event_type = 'complete trial',
                trial_id = [1, 2, 3]
        >>> seg = neo.Segment()
        >>> seg.epochs = [epoch]

        # Will return a list with the complete event object
        >>> get_epochs(seg, prop={'epoch_type': 'complete trial'})

        # Will return an empty list
        >>> get_epochs(seg, prop={'epoch_type': 'error trial'})

        # Will return a list with an Event object, but only with trial 2
        >>> get_epochs(seg, prop={'trial_id': 2})

        # Will return a list with an Event object, but only with trials 1 and 2
        >>> get_epochs(seg, prop={'trial_id': [1, 2]})
    """
    if isinstance(container, neo.Segment):
        return _get_from_list(container.epochs, prop=properties)

    elif isinstance(container, neo.Block):
        epoch_list = []
        for seg in container.segments:
            epoch_list += _get_from_list(seg.epochs, prop=properties)
        return epoch_list
    else:
        raise TypeError(
            'Container needs to be of type Block or Segment, not %s '
            'in order to extract Epochs.' % (type(container)))


def _get_from_list(input_list, prop=None):
    """
    Internal function
    """
    output_list = []
    # empty or no dictionary
    if not prop or bool([b for b in prop.values() if b == []]):
        output_list += [e for e in input_list]
    # dictionary is given
    else:
        for ep in input_list:
            if isinstance(ep, neo.Epoch) or isinstance(ep, neo.Event):
                sparse_ep = ep.copy()
            elif hasattr(neo.io, 'proxyobjects') and isinstance(ep,
                                                                (
                                                                        neo.io.proxyobjects.EpochProxy,
                                                                        neo.io.proxyobjects.EventProxy)):
                # need to load the Event/Epoch in order to be able to filter by array annotations
                sparse_ep = ep.load()
                # ugly fix for labels of loaded events/epochs being zero-terminated bytes
                sparse_ep.labels = sparse_ep.labels.astype(str)
            for k in prop.keys():
                sparse_ep = _filter_event_epoch(sparse_ep, k, prop[k])
                # if there is nothing left, it cannot filtered
                if sparse_ep is None:
                    break
            if sparse_ep is not None:
                output_list.append(sparse_ep)
    return output_list


def _filter_event_epoch(obj, annotation_key, annotation_value):
    """
    Internal function.

    This function returns a copy of a Event or Epoch object, which only
    contains attributes or annotations corresponding to requested key-value
    pairs.

    Parameters:
    -----------
    obj : Event
        The Event or Epoch object to modify.
    annotation_key : string, int or float
        The name of the annotation used to filter.
    annotation_value : string, int, float, list or np.ndarray
        The accepted value or list of accepted values of the attributes or
        annotations specified by annotation_key. For each entry in obj the
        respective annotation defined by annotation_key is compared to the
        annotation value. The entry of obj is kept if the attribute or
        annotation is equal or contained in annotation_value.

    Returns:
    --------
    obj : Event or Epoch
        The Event or Epoch object with every event or epoch removed that does
        not match the filter criteria (i.e., where none of the entries in
        annotation_value match the attribute or annotation annotation_key.
    """
    valid_ids = _get_valid_ids(obj, annotation_key, annotation_value)

    if len(valid_ids) == 0:
        return None

    return _event_epoch_slice_by_valid_ids(obj, valid_ids)


def _event_epoch_slice_by_valid_ids(obj, valid_ids):
    """
    Internal function
    """
    # modify annotations
    sparse_annotations = _get_valid_annotations(obj, valid_ids)

    # modify array annotations
    sparse_array_annotations = {key: value[valid_ids]
                                for key, value in obj.array_annotations.items()
                                if len(value)}

    if type(obj) is neo.Event:
        sparse_obj = neo.Event(
            times=copy.deepcopy(obj.times[valid_ids]),
            units=copy.deepcopy(obj.units),
            name=copy.deepcopy(obj.name),
            description=copy.deepcopy(obj.description),
            file_origin=copy.deepcopy(obj.file_origin),
            array_annotations=sparse_array_annotations,
            **sparse_annotations)
    elif type(obj) is neo.Epoch:
        sparse_obj = neo.Epoch(
            times=copy.deepcopy(obj.times[valid_ids]),
            durations=copy.deepcopy(obj.durations[valid_ids]),
            units=copy.deepcopy(obj.units),
            name=copy.deepcopy(obj.name),
            description=copy.deepcopy(obj.description),
            file_origin=copy.deepcopy(obj.file_origin),
            array_annotations=sparse_array_annotations,
            **sparse_annotations)
    else:
        raise TypeError('Can only slice Event and Epoch objects by valid IDs.')

    return sparse_obj


def _get_valid_ids(obj, annotation_key, annotation_value):
    """
    Internal function
    """
    # wrap annotation value to be list
    if not type(annotation_value) in [list, np.ndarray]:
        annotation_value = [annotation_value]

    # get all real attributes of object
    attributes = inspect.getmembers(obj)
    attributes_names = [t[0] for t in attributes if not (
            t[0].startswith('__') and t[0].endswith('__'))]
    attributes_ids = [i for i, t in enumerate(attributes) if not (
            t[0].startswith('__') and t[0].endswith('__'))]

    # check if annotation is present
    value_avail = False
    if annotation_key in obj.annotations:
        check_value = obj.annotations[annotation_key]
        value_avail = True
    elif annotation_key in obj.array_annotations:
        check_value = obj.array_annotations[annotation_key]
        value_avail = True
    elif annotation_key in attributes_names:
        check_value = attributes[attributes_ids[
            attributes_names.index(annotation_key)]][1]
        value_avail = True

    if value_avail:
        # check if annotation is list and fits to length of object list
        if not _is_annotation_list(check_value, len(obj)):
            # check if annotation is single value and fits to requested value
            if check_value in annotation_value:
                valid_mask = np.ones(obj.shape)
            else:
                valid_mask = np.zeros(obj.shape)
                if type(check_value) != str:
                    warnings.warn(
                        'Length of annotation "%s" (%s) does not fit '
                        'to length of object list (%s)' % (
                            annotation_key, len(check_value), len(obj)))

        # extract object entries, which match requested annotation
        else:
            valid_mask = np.zeros(obj.shape)
            for obj_id in range(len(obj)):
                if check_value[obj_id] in annotation_value:
                    valid_mask[obj_id] = True
    else:
        valid_mask = np.zeros(obj.shape)

    valid_ids = np.where(valid_mask)[0]

    return valid_ids


def _get_valid_annotations(obj, valid_ids):
    """
    Internal function
    """
    sparse_annotations = copy.deepcopy(obj.annotations)
    for key in sparse_annotations:
        if _is_annotation_list(sparse_annotations[key], len(obj)):
            sparse_annotations[key] = list(np.array(sparse_annotations[key])[
                                               valid_ids])
    return sparse_annotations


def _is_annotation_list(value, exp_length):
    """
    Internal function
    """
    return (
            (isinstance(value, list) or (
                    isinstance(value, np.ndarray) and value.ndim > 0)) and (
                    len(value) == exp_length))


def add_epoch(
        segment, event1, event2=None, pre=0 * pq.s, post=0 * pq.s,
        attach_result=True, **kwargs):
    """
    Create Epochs around a single Event, or between pairs of events. Starting
    and end time of the Epoch can be modified using pre and post as offsets
    before the and after the event(s). Additional keywords will be directly
    forwarded to the Epoch intialization.

    Parameters:
    -----------
    segment : Segment
        The segment in which the final Epoch object is added.
    event1 : Event
        The Event objects containing the start events of the epochs. If no
        event2 is specified, these event1 also specifies the stop events, i.e.,
        the Epoch is cut around event1 times.
    event2: Event
        The Event objects containing the stop events of the epochs. If no
        event2 is specified, event1 specifies the stop events, i.e., the Epoch
        is cut around event1 times. The number of events in event2 must match
        that of event1.
    pre, post: Quantity (time)
        Time offsets to modify the start (pre) and end (post) of the resulting
        Epoch. Example: pre=-10*ms and post=+25*ms will cut from 10 ms before
        event1 times to 25 ms after event2 times
    attach_result: bool
        If True, the resulting Epoch object is added to segment.

    Keyword Arguments:
    ------------------
    Passed to the Epoch object.

    Returns:
    --------
    epoch: Epoch
        An Epoch object with the calculated epochs (one per entry in event1).

    See also:
    ---------
    Event.to_epoch()
    """
    if event2 is None:
        event2 = event1

    if not isinstance(segment, neo.Segment):
        raise TypeError(
            'Segment has to be of type Segment, not %s' % type(segment))

    # load the full event if a proxy object has been given as an argument
    if hasattr(neo.io, 'proxyobjects') and isinstance(event1,
                                                      neo.io.proxyobjects.EventProxy):
        event1 = event1.load()
    if hasattr(neo.io, 'proxyobjects') and isinstance(event2,
                                                      neo.io.proxyobjects.EventProxy):
        event2 = event2.load()

    for event in [event1, event2]:
        if not isinstance(event, neo.Event):
            raise TypeError(
                'Events have to be of type Event, not %s' % type(event))

    if len(event1) != len(event2):
        raise ValueError(
            'event1 and event2 have to have the same number of entries in '
            'order to create epochs between pairs of entries. Match your '
            'events before generating epochs. Current event lengths '
            'are %i and %i' % (len(event1), len(event2)))

    times = event1.times + pre
    durations = event2.times + post - times

    if any(durations < 0):
        raise ValueError(
            'Can not create epoch with negative duration. '
            'Requested durations %s.' % durations)
    elif any(durations == 0):
        raise ValueError('Can not create epoch with zero duration.')

    if 'name' not in kwargs:
        kwargs['name'] = 'epoch'
    if 'labels' not in kwargs:
        # this needs to be changed to '%s_%i' % (kwargs['name'], i) for i in range(len(times))]
        # when labels become unicode
        kwargs['labels'] = [
            ('%s_%i' % (kwargs['name'], i)).encode('ascii') for i in
            range(len(times))]

    ep = neo.Epoch(times=times, durations=durations, **kwargs)

    ep.annotate(**event1.annotations)

    if attach_result:
        segment.epochs.append(ep)
        segment.create_relationship()

    return ep


def match_events(event1, event2):
    """
    Finds pairs of Event entries in event1 and event2 with the minimum delay,
    such that the entry of event1 directly precedes the entry of event2.
    Returns filtered two events of identical length, which contain matched
    entries.

    Parameters:
    -----------
    event1, event2: Event
        The two Event objects to match up.

    Returns:
    --------
    event1, event2: Event
        Event objects with identical number of events, containing only those
        events that could be matched against each other. A warning is issued if
        not all events in event1 or event2 could be matched.
    """
    # load the full event if a proxy object has been given as an argument
    if hasattr(neo.io, 'proxyobjects') and isinstance(event1,
                                                      neo.io.proxyobjects.EventProxy):
        event1 = event1.load()
    if hasattr(neo.io, 'proxyobjects') and isinstance(event2,
                                                      neo.io.proxyobjects.EventProxy):
        event2 = event2.load()

    id1, id2 = 0, 0
    match_ev1, match_ev2 = [], []
    while id1 < len(event1) and id2 < len(event2):
        time1 = event1.times[id1]
        time2 = event2.times[id2]

        # wrong order of events
        if time1 >= time2:
            id2 += 1

        # shorter epoch possible by later event1 entry
        elif id1 + 1 < len(event1) and event1.times[id1 + 1] < time2:
            # there is no event in 2 until the next event in 1
            id1 += 1

        # found a match
        else:
            match_ev1.append(id1)
            match_ev2.append(id2)
            id1 += 1
            id2 += 1

    if id1 < len(event1):
        warnings.warn(
            'Could not match all events to generate epochs. Missed '
            '%s event entries in event1 list' % (len(event1) - id1))
    if id2 < len(event2):
        warnings.warn(
            'Could not match all events to generate epochs. Missed '
            '%s event entries in event2 list' % (len(event2) - id2))

    event1_matched = _event_epoch_slice_by_valid_ids(
        obj=event1, valid_ids=match_ev1)
    event2_matched = _event_epoch_slice_by_valid_ids(
        obj=event2, valid_ids=match_ev2)

    return event1_matched, event2_matched


def cut_block_by_epochs(block, properties=None, reset_time=False):
    """
    This function cuts Segments in a Block according to multiple Neo
    Epoch objects.

    The function alters the Block by adding one Segment per Epoch entry
    fulfilling a set of conditions on the Epoch attributes and annotations. The
    original segments are removed from the block.

    A dictionary contains restrictions on which Epochs are considered for
    the cutting procedure. To this end, it is possible to
    specify accepted (valid) values of specific annotations on the source
    Epochs.

    The resulting cut segments may either retain their original time stamps, or
    be shifted to a common starting time.

    Parameters
    ----------
    block: Block
        Contains the Segments to cut according to the Epoch criteria provided
    properties: dictionary
        A dictionary that contains the Epoch keys and values to filter for.
        Each key of the dictionary is matched to an attribute or an an
        annotation of the Event. The value of each dictionary entry corresponds
        to a valid entry or a list of valid entries of the attribute or
        annotation.

        If the value belonging to the key is a list of entries of the same
        length as the number of epochs in the Epoch object, the list entries
        are matched to the epochs in the Epoch object. The resulting Epoch
        object contains only those epochs where the values match up.

        Otherwise, the value is compared to the attributes or annotation of the
        Epoch object as such, and depending on the comparison, either the
        complete Epoch object is returned or not.

        If None or an empty dictionary is passed, all Epoch Objects will
        be considered

    reset_time: bool
        If True the times stamps of all sliced objects are set to fall
        in the range from 0 to the duration of the epoch duration.
        If False, original time stamps are retained.
        Default is False.

    Returns:
    --------
    None
    """
    if not isinstance(block, neo.Block):
        raise TypeError(
            'block needs to be a Block, not %s' % type(block))

    old_segments = copy.copy(block.segments)
    for seg in old_segments:
        epochs = _get_from_list(seg.epochs, prop=properties)
        if len(epochs) > 1:
            warnings.warn(
                'Segment %s contains multiple epochs with '
                'requested properties (%s). Sub-segments can '
                'have overlapping times' % (seg.name, properties))

        elif len(epochs) == 0:
            warnings.warn(
                'No epoch is matching the requested epoch properties %s. '
                'No cutting of segment %s performed.' % (properties, seg.name))

        for epoch in epochs:
            new_segments = cut_segment_by_epoch(
                seg, epoch=epoch, reset_time=reset_time)
            block.segments += new_segments

        block.segments.remove(seg)
    block.create_many_to_one_relationship(force=True)


def cut_segment_by_epoch(seg, epoch, reset_time=False):
    """
    Cuts a Segment according to an Epoch object

    The function returns a list of Segments, where each segment corresponds
    to an epoch in the Epoch object and contains the data of the original
    Segment cut to that particular Epoch.

    The resulting segments may either retain their original time stamps,
    or can be shifted to a common time axis.

    Parameters
    ----------
    seg: Segment
        The Segment containing the original uncut data.
    epoch: Epoch
        For each epoch in this input, one segment is generated according to
         the epoch time and duration.
    reset_time: bool
        If True the times stamps of all sliced objects are set to fall
        in the range from 0 to the duration of the epoch duration.
        If False, original time stamps are retained.
        Default is False.

    Returns:
    --------
    segments: list of Segments
        Per epoch in the input, a Segment with AnalogSignal and/or
        SpikeTrain Objects will be generated and returned. Each Segment will
        receive the annotations of the corresponding epoch in the input.
    """
    if not isinstance(seg, neo.Segment):
        raise TypeError(
            'Seg needs to be of type Segment, not %s' % type(seg))

    if type(seg.parents[0]) != neo.Block:
        raise ValueError(
            'Segment has no block as parent. Can not cut segment.')

    if not isinstance(epoch, neo.Epoch):
        raise TypeError(
            'Epoch needs to be of type Epoch, not %s' % type(epoch))

    segments = []
    for ep_id in range(len(epoch)):
        subseg = seg_time_slice(seg,
                                epoch.times[ep_id],
                                epoch.times[ep_id] + epoch.durations[ep_id],
                                reset_time=reset_time)

        # Add annotations of Epoch
        for a in epoch.annotations:
            if type(epoch.annotations[a]) is list \
                    and len(epoch.annotations[a]) == len(epoch):
                subseg.annotations[a] = copy.copy(epoch.annotations[a][ep_id])
            else:
                subseg.annotations[a] = copy.copy(epoch.annotations[a])

        # Add array-annotations of Epoch
        for key, val in epoch.array_annotations.items():
            if len(val):
                subseg.annotations[key] = copy.copy(val[ep_id])

        segments.append(subseg)

    return segments


def seg_time_slice(seg, t_start=None, t_stop=None, reset_time=False, **kwargs):
    """
    Creates a time slice of a Segment containing slices of all child
    objects.

    Parameters:
    -----------
    seg: Segment
        The Segment object to slice.
    t_start: Quantity
        Starting time of the sliced time window.
    t_stop: Quantity
        Stop time of the sliced time window.
    reset_time: bool
        If True the time stamps of all sliced objects are set to fall
        in the range from t_start to t_stop.
        If False, original time stamps are retained.
        Default is False.

    Keyword Arguments:
    ------------------
        Additional keyword arguments used for initialization of the sliced
        Segment object.

    Returns:
    --------
    seg: Segment
        Temporal slice of the original Segment from t_start to t_stop.
    """
    subseg = neo.Segment(**kwargs)

    for attr in [
        'file_datetime', 'rec_datetime', 'index',
        'name', 'description', 'file_origin']:
        setattr(subseg, attr, getattr(seg, attr))

    subseg.annotations = copy.deepcopy(seg.annotations)

    t_shift = - t_start

    # cut analogsignals and analogsignalarrays
    for ana_id in range(len(seg.analogsignals)):
        if isinstance(seg.analogsignals[ana_id], neo.AnalogSignal):
            ana_time_slice = seg.analogsignals[ana_id].time_slice(t_start,
                                                                  t_stop)
        elif (hasattr(neo.io, 'proxyobjects') and
              isinstance(seg.analogsignals[ana_id],
                         neo.io.proxyobjects.AnalogSignalProxy)):
            ana_time_slice = seg.analogsignals[ana_id].load(
                time_slice=(t_start, t_stop))
        if reset_time:
            ana_time_slice.t_start = ana_time_slice.t_start + t_shift
        subseg.analogsignals.append(ana_time_slice)

    # cut spiketrains
    for st_id in range(len(seg.spiketrains)):
        if isinstance(seg.spiketrains[st_id], neo.SpikeTrain):
            st_time_slice = seg.spiketrains[st_id].time_slice(t_start, t_stop)
        elif (hasattr(neo.io, 'proxyobjects') and
              isinstance(seg.spiketrains[st_id],
                         neo.io.proxyobjects.SpikeTrainProxy)):
            st_time_slice = seg.spiketrains[st_id].load(
                time_slice=(t_start, t_stop))
        if reset_time:
            st_time_slice = shift_spiketrain(st_time_slice, t_shift)
        subseg.spiketrains.append(st_time_slice)

    # cut events
    for ev_id in range(len(seg.events)):
        if isinstance(seg.events[ev_id], neo.Event):
            ev_time_slice = event_time_slice(seg.events[ev_id], t_start,
                                             t_stop)
        elif (hasattr(neo.io, 'proxyobjects') and
              isinstance(seg.events[ev_id], neo.io.proxyobjects.EventProxy)):
            ev_time_slice = seg.events[ev_id].load(
                time_slice=(t_start, t_stop))
        if reset_time:
            ev_time_slice = shift_event(ev_time_slice, t_shift)
        # appending only non-empty events
        if len(ev_time_slice):
            subseg.events.append(ev_time_slice)

    # cut epochs
    for ep_id in range(len(seg.epochs)):
        if isinstance(seg.epochs[ep_id], neo.Epoch):
            ep_time_slice = epoch_time_slice(seg.epochs[ep_id], t_start,
                                             t_stop)
        elif (hasattr(neo.io, 'proxyobjects') and
              isinstance(seg.epochs[ep_id], neo.io.proxyobjects.EpochProxy)):
            ep_time_slice = seg.epochs[ep_id].load(
                time_slice=(t_start, t_stop))
        if reset_time:
            ep_time_slice = shift_epoch(ep_time_slice, t_shift)
        # appending only non-empty epochs
        if len(ep_time_slice):
            subseg.epochs.append(ep_time_slice)

    return subseg


def shift_spiketrain(spiketrain, t_shift):
    """
    Shifts a spike train to start at a new time.

    Parameters:
    -----------
    spiketrain: SpikeTrain
        Spiketrain of which a copy will be generated with shifted spikes and
        starting and stopping times
    t_shift: Quantity (time)
        Amount of time by which to shift the SpikeTrain.

    Returns:
    --------
    spiketrain: SpikeTrain
        New instance of a SpikeTrain object starting at t_start (the original
        SpikeTrain is not modified).
    """
    new_st = spiketrain.duplicate_with_new_data(
        signal=spiketrain.times.view(pq.Quantity) + t_shift,
        t_start=spiketrain.t_start + t_shift,
        t_stop=spiketrain.t_stop + t_shift)
    return new_st


def event_time_slice(event, t_start=None, t_stop=None):
    """
    Slices an Event object to retain only those events that fall in a certain
    time window.

    Parameters:
    -----------
    event: Event
        The Event to slice.
    t_start, t_stop: Quantity (time)
        Time window in which to retain events. An event at time t is retained
        if t_start <= t < t_stop.

    Returns:
    --------
    event: Event
        New instance of an Event object containing only the events in the time
        range.
    """
    if t_start is None:
        t_start = -np.inf
    if t_stop is None:
        t_stop = np.inf

    valid_ids = np.where(np.logical_and(
        event.times >= t_start, event.times < t_stop))[0]

    new_event = _event_epoch_slice_by_valid_ids(event, valid_ids=valid_ids)

    return new_event


def epoch_time_slice(epoch, t_start=None, t_stop=None):
    """
    Slices an Epoch object to retain only those epochs that fall in a certain
    time window.

    Parameters:
    -----------
    epoch: Epoch
        The Epoch to slice.
    t_start, t_stop: Quantity (time)
        Time window in which to retain epochs. An epoch at time t and
        duration d is retained if t_start <= t < t_stop - d.

    Returns:
    --------
    epoch: Epoch
        New instance of an Epoch object containing only the epochs in the time
        range.
    """
    if t_start is None:
        t_start = -np.inf
    if t_stop is None:
        t_stop = np.inf

    valid_ids = np.where(np.logical_and(
        epoch.times >= t_start, epoch.times + epoch.durations < t_stop))[0]

    new_epoch = _event_epoch_slice_by_valid_ids(epoch, valid_ids=valid_ids)

    return new_epoch


def shift_event(ev, t_shift):
    """
    Shifts an event by an amount of time.

    Parameters:
    -----------
    event: Event
        Event of which a copy will be generated with shifted times
    t_shift: Quantity (time)
        Amount of time by which to shift the Event.

    Returns:
    --------
    epoch: Event
        New instance of an Event object starting at t_shift later than the
        original Event (the original Event is not modified).
    """
    return _shift_time_signal(ev, t_shift)


def shift_epoch(epoch, t_shift):
    """
    Shifts an epoch by an amount of time.

    Parameters:
    -----------
    epoch: Epoch
        Epoch of which a copy will be generated with shifted times
    t_shift: Quantity (time)
        Amount of time by which to shift the Epoch.

    Returns:
    --------
    epoch: Epoch
        New instance of an Epoch object starting at t_shift later than the
        original Epoch (the original Epoch is not modified).
    """
    return _shift_time_signal(epoch, t_shift)


def _shift_time_signal(sig, t_shift):
    """
    Internal function.
    """
    if not hasattr(sig, 'times'):
        raise AttributeError(
            'Can only shift signals, which have an attribute'
            ' "times", not %s' % type(sig))
    new_sig = sig.duplicate_with_new_data(signal=sig.times + t_shift)
    return new_sig


# TODO: End of block of functions that have to be imported from neo.utils
#  once the corresponding PR has been merged


def extract_metadata_from_events(events):
    def _extract_table(codes, code_start, code_end):
        idx_start = np.where(codes == code_start)[0][0]
        idx_end = np.where(codes == code_end)[0][0]
        # Table content is everything between start and end indices
        # without them included, organized in a 2D matrix
        # with consecutive triplets of codes as rows. Column 1
        # contains the values while columns 0 and 2 contain flankers.
        assert ((idx_end - idx_start - 1) % 3 == 0)
        return codes[idx_start + 1:idx_end].reshape([-1, 3])

    def _parse_target_table(table):
        metadata = {}

        target_id = None
        for lflank, value, rflank in table:
            if lflank == 65531 and rflank == 65530:
                start = value % 10  # 1st digit of the value
                if start:
                    target_id = value % 100 // 10  # 2nd digit of the value
                else:
                    target_id = None
                continue

            if target_id is None:
                continue

            if lflank == 2021 and rflank == 2020:
                key = "target{:02d}_xpos".format(target_id)
                metadata[key] = (value - 2 ** 15) / 1000. * pq.cm
            elif lflank == 2031 and rflank == 2030:
                key = "target{:02d}_ypos".format(target_id)
                metadata[key] = (value - 2 ** 15) / 1000. * pq.cm
            elif lflank == 2041 and rflank == 2040:
                key = "target{:02d}_radius".format(target_id)
                metadata[key] = value / 1000. * pq.cm

        return metadata

    def _parse_load_table(table):
        metadata = {}
        return metadata

    def _parse_tp_table(table):
        metadata = {}
        return metadata

    def _parse_twp_table(table):
        metadata = {}
        return metadata

    def _parse_range_table(table):
        metadata = {}
        for lflank, value, rflank in table:
            if lflank == 8011 and rflank == 8010:
                metadata["EyeXcm_max"] = (value - 2 ** 15) * pq.cm
            elif lflank == 8021 and rflank == 8020:
                metadata["EyeXcm_min"] = (value - 2 ** 15) * pq.cm
            elif lflank == 8031 and rflank == 8030:
                metadata["EyeYcm_max"] = (value - 2 ** 15) * pq.cm
            elif lflank == 8041 and rflank == 8040:
                metadata["EyeYcm_min"] = (value - 2 ** 15) * pq.cm
            elif lflank == 8051 and rflank == 8050:
                metadata["HandXcm_max"] = (value - 2 ** 15) * pq.cm
            elif lflank == 8061 and rflank == 8060:
                metadata["HandXcm_min"] = (value - 2 ** 15) * pq.cm
            elif lflank == 8071 and rflank == 8070:
                metadata["HandYcm_max"] = (value - 2 ** 15) * pq.cm
            elif lflank == 8081 and rflank == 8080:
                metadata["HandYcm_min"] = (value - 2 ** 15) * pq.cm
            elif lflank == 8091 and rflank == 8090:
                metadata["TargetXcm_max"] = (value - 2 ** 15) * pq.cm
            elif lflank == 8101 and rflank == 8100:
                metadata["TargetXcm_min"] = (value - 2 ** 15) * pq.cm
            elif lflank == 8111 and rflank == 8110:
                metadata["TargetYcm_max"] = (value - 2 ** 15) * pq.cm
            elif lflank == 8121 and rflank == 8120:
                metadata["TargetYcm_min"] = (value - 2 ** 15) * pq.cm
        return metadata

    metadata = {}
    event_codes = np.array([int(x) for x in events.labels])

    # Metadata consists of 5 tables : target, load, trial protocol (tp),
    # task wide parameter (twp), and range.
    target_table = _extract_table(event_codes, 65211, 65210)
    metadata.update(_parse_target_table(target_table))

    load_table = _extract_table(event_codes, 65221, 65220)
    metadata.update(_parse_load_table(load_table))

    tp_table = _extract_table(event_codes, 65231, 65230)
    metadata.update(_parse_tp_table(tp_table))

    twp_table = _extract_table(event_codes, 65241, 65240)
    metadata.update(_parse_twp_table(twp_table))

    range_table = _extract_table(event_codes, 65251, 65250)
    metadata.update(_parse_range_table(range_table))

    return metadata


def rescale_by_range(data, range_orig, range_rescaled):
    return (data - range_orig[0]) / (range_orig[1] - range_orig[0]) * \
           (range_rescaled[1] - range_rescaled[0]) \
           + range_rescaled[0]


def convert_mv_to_cm(anasig, metadata, correct_bias=True):
    signame = anasig.name
    if 'bundle' in signame.lower():
        return convert_anasig_bundle(anasig, metadata, correct_bias)
    elif signame not in ['EyeXcm', 'EyeYcm', 'HandXcm', 'HandYcm', 'TargetXcm',
                         'TargetYcm']:
        print("No conversion is necessary for {} data.".format(signame))
        return
    range_mv = (metadata['analogsignal_min'].magnitude,
                metadata['analogsignal_max'].magnitude)
    range_cm = (metadata['{}_min'.format(signame)].magnitude,
                metadata['{}_max'.format(signame)].magnitude)
    if signame[-3] == 'X':
        bias = metadata['xbias'].magnitude
    elif signame[-3] == 'Y':
        bias = metadata['ybias'].magnitude
    else:
        bias = 0.
    data = anasig.rescale('mV').magnitude
    data_new = rescale_by_range(data, range_mv, range_cm)
    if correct_bias:
        data_new -= bias
    anasig_new = neo.AnalogSignal(data_new, units='cm',
                                  sampling_rate=anasig.sampling_rate)
    anasig_new._copy_data_complement(anasig)
    return anasig_new


def convert_anasig_bundle(anasig, metadata, correct_bias=True):
    signames = anasig.array_annotations['channel_names']
    signals_to_convert = ['EyeXcm', 'EyeYcm', 'HandXcm', 'HandYcm',
                          'TargetXcm',
                          'TargetYcm']
    conversion_indices = [idx for idx, name in enumerate(signames)
                          if name in signals_to_convert]
    if not len(conversion_indices):
        print("No conversion is necessary for {} data.".format(signames))
        return
    data = anasig.rescale('mV').magnitude[:, conversion_indices].copy()
    for idx, signame in enumerate(signames[conversion_indices]):
        range_mv = (metadata['analogsignal_min'].magnitude,
                    metadata['analogsignal_max'].magnitude)
        range_cm = (metadata['{}_min'.format(signame)].magnitude,
                    metadata['{}_max'.format(signame)].magnitude)
        data[:, idx] = rescale_by_range(data[:, idx], range_mv, range_cm)
        if signame[-3] == 'X':
            bias = metadata['xbias'].magnitude
        elif signame[-3] == 'Y':
            bias = metadata['ybias'].magnitude
        else:
            bias = 0.
        if correct_bias:
            data[:, idx] -= bias
    anasig_new = neo.AnalogSignal(data, units='cm',
                                  sampling_rate=anasig.sampling_rate)
    anasig_new._copy_data_complement(anasig)
    anasig_new.name = 'Converted channel bundle ({names})'.format(
        names=','.join(signals_to_convert))
    arr_ann_new = {key: val[conversion_indices]
                   for key, val in anasig.array_annotations.items()}
    anasig_new.array_annotate(**arr_ann_new)
    return anasig_new


def extract_behavioral_codes(path, filename='descriptor_codes_global.csv'):
    """
    Extracts behavioral codes from the .csv file specific to each session

    Args:
        path:
            Path to the .csv file
        filename:
            the name of the .csv file (NOTE: for now, the name of the .csv
            file is "codes_landing.csv". You may need to
            change this name according to the .csv file name)

    returns:
        behavioral_codes:
            a dictionary containing all the behavioral codes in the .csv file
            and their corresponding descriptions
    """
    with open(path + '/' + filename, 'r') as csvfile:
        code_reader = csv.reader(csvfile)
        next(code_reader)  # skip the header line
        behavioral_codes = {}
        for row in code_reader:
            behavioral_codes[row[1]] = int(row[2])
    return behavioral_codes


def cut_events_into_trials(events, behavioral_codes, trial_type):
    """
    Extracts list of event@time for all trials within a session

    Args:
        events:
            a neo.Event object containing all event codes@times for a session

        behavioral_codes:
            the output of the function "extract_behavioral_codes",
            which is a dictionary containing all the
            behavioral codes defined in the .csv file

        trial_type:
            either 'successful' or 'all'.
            This indicates whether to select successful or all monkey attempts.

    returns:
        event_list:
            a list of neo.Event objects, each of which contains all
            events@times in a corresponding trial

    NOTE:
        A trial defined by 'trial_start' and 'trial_end' events can contain
        multiple attempts of the monkey to perform
        the complete sequence of target reaches. In such a case, all attempts
        except the last one are failure attempts.
        A successful trial should start with the appearance of the central
        target ('target_01_on'), and end with reward off ('reward_off').
        If trial_type is set to 'all', this function returns all monkey
        attempts including the unsuccessful ones. For unsuccessfull attempts,
        each trial beings with the appearance of the central target and ends
        with an error code meaning that the trial was not successful.
    """

    event_labels = events.labels.astype(int)
    event_times = events.times

    trial_start_idx = np.where(event_labels ==
                               int(behavioral_codes['trial_start']))[0]
    # returns indices of trial starts
    trial_end_idx = np.where(event_labels ==
                             int(behavioral_codes['trial_end']))[0]
    # returns indices of trial ends

    event_list = []
    for idx_start, idx_end in zip(trial_start_idx, trial_end_idx):
        trial_event_labels = event_labels[idx_start:idx_end + 1]
        idxs_central_target_on = \
            np.where(trial_event_labels ==
                     int(behavioral_codes['target_01_status_1']))[
                0] + idx_start
        idx_reward_off = \
            np.where(trial_event_labels ==
                     int(behavioral_codes['automatic_reward_end']))[
                0] + idx_start
        idxs_trial_edges = np.hstack([idxs_central_target_on,
                                      idx_reward_off + 1])

        temp_sequence_tries = []
        for idx1, idx2 in zip(idxs_trial_edges[:-1], idxs_trial_edges[1:]):
            # slice the data into sequence_tries (contains failures and successes)
            temp_sequence_tries.append(neo.Event(event_times[idx1:idx2],
                                                 labels=event_labels[
                                                        idx1:idx2]))

        if trial_type == 'successful':
            # just pick the complete one (which is naturally the last sequence try for one 'trial')
            event_list.append(temp_sequence_tries[-1])


        elif trial_type == 'all':
            event_list.extend(temp_sequence_tries)

        else:
            print('incorrect trial_type input')

    return event_list


def construct_spiketrains_dict(spiketrains, ch_ids=None):
    """
    Extracts all spike trains recorded from an array

    Args:
        spiketrains:
            list of neo.Spiketrain objects containing all recorded spike trains
        ch_ids:
            a list of integers indicating the channel IDs of the spike trains
            to be stored in the returned dictionary.
            When set to None, all channels (i.e., channel IDs from 1 to 128)
            are stored. Defaults to None.

    returns:
        spikes:
            a dictionary of lists, indexed by channel IDs as keys, containing
            all spike trains of the corresponding channel.
    """
    if ch_ids is None:
        ch_ids = range(1, 129)
    spiketrains_dict = {x: [] for x in ch_ids}
    for st in spiketrains:
        ch_id = st.annotations['channel_id']
        if ch_id in ch_ids:
            spiketrains_dict[ch_id].append(st)
    return spiketrains_dict


def construct_analogsignals_dict(analogsignals):
    return {sig.name: sig for sig in analogsignals}


def slice_spiketrains(spiketrains, time_range):
    t_start, t_stop = time_range
    spiketrains_sliced = {ch_id: [] for ch_id in spiketrains}
    for ch_id, sts in spiketrains.items():
        for st in sts:
            spiketrains_sliced[ch_id].append(st.time_slice(t_start, t_stop))
    return spiketrains_sliced


def slice_analogsignals(anasigs, time_range):
    t_start, t_stop = time_range
    anasigs_sliced = {}
    for signame, anasig in anasigs.items():
        anasigs_sliced[signame] = anasig.time_slice(t_start, t_stop)
    return anasigs_sliced


def slice_ns6_analogsignals(ns6_anasigs, time_range):
    t_start, t_stop = time_range
    anasigs_sliced = []
    for anasig in ns6_anasigs:
        anasigs_sliced.append(anasig.time_slice(t_start, t_stop))
    return anasigs_sliced


def wavelet_transform(signal, freq, nco, fs):
    """
    Compute the wavelet transform of a given signal with Morlet mother wavelet.
    The definition of the wavelet is based on Le van Quyen et al. J
    Neurosci Meth 111:83-98 (2001).

    Parameters
    ----------
    signal : 1D array_like
        Signal to be wavelet-transformed
    freq : float
        Center frequency of the Morlet wavelet.
    nco : float
        Size of the mother wavelet (approximate number of cycles within a
        wavelet). A larger nco value leads to a higher frequency resolution but
        a lower temporal resolution, and vice versa. Typically used values are
        in a range of 3 - 8.
    fs : float
        Sampling rate of the signal.

    Returns
    -------
    signal_trans: 1D complex array_like
        Wavelet-transformed signal
    """

    # Morlet wavelet generator (c.f. Le van Quyen et al. J Neurosci Meth
    # 111:83-98 (2001))
    def _morlet_wavelet(freq, nco, Fs, N):
        sigma = nco / (6. * freq)
        t = (np.arange(N, dtype='float') - N / 2) / Fs
        if N % 2 == 0:
            t = np.roll(t, int(N / 2))
        else:
            t = np.roll(t, int(N / 2) + 1)
        return np.sqrt(freq) * np.exp(
            -(t * t) / (2 * sigma ** 2) + 1j * 2 * np.pi * freq * t)

    data = np.asarray(signal)
    # When the input is AnalogSignal, the axis for time index (i.e. the
    # first axis) needs to be rolled to the last
    if isinstance(signal, neo.AnalogSignal):
        data = np.rollaxis(data, 0, len(data.shape))[0]

    # check whether the given central frequency is less than the Nyquist
    # frequency of the signal
    if freq >= fs / 2:
        raise ValueError(
            "freq must be less than the half of Fs " +
            "(sampling rate of the original signal)")

    # check if nco is positive
    if nco <= 0:
        raise ValueError("nco must be positive")

    N = len(data)
    # the least power of 2 greater than N
    N_pow2 = 2 ** (int(np.log2(N)) + 1)

    # zero-padding to a power of 2 for efficient convolution
    tmpdata = np.zeros(N_pow2)
    tmpdata[0:N] = data

    # generate Morlet wavelet
    wavelet = _morlet_wavelet(freq, nco, fs, N_pow2)

    # convolution of the signal with the wavelet
    signal_trans = np.fft.ifft(np.fft.fft(tmpdata) * np.fft.fft(wavelet))[0:N]

    if isinstance(signal, neo.AnalogSignal):
        return signal.duplicate_with_new_array(signal_trans.T)
    elif isinstance(signal, pq.quantity.Quantity):
        return signal_trans * signal.units
    else:
        return signal_trans


def compute_specgram(event_related_signals, nco):
    # event_nr = len(event_related_signals)
    # specgram_dict = {}
    freq_range = np.arange(1, 101)
    specgram_final = np.zeros((len(freq_range), len(event_related_signals[0])))
    for sig in event_related_signals:
        specgram = np.zeros((len(sig)))
        zero_mean_sig = sig - np.mean(sig)
        for fr in freq_range:
            transformed_sig = 10 * np.log10(np.abs(wavelet_transform(
                zero_mean_sig, fr, nco, 1000)))
            specgram = np.vstack((specgram, transformed_sig))
        specgram = specgram[1:, :]
        specgram_final += specgram
    specgram_final = specgram_final / float(len(event_related_signals))
    # specgram_dict[anasig.annotations['channel_id']] = specgram
    return specgram_final


def downsample_ns6_anasigs(anasigs_trial, params):
    desired_sr = params['desired_sr']
    scaling_factor = int(anasigs_trial[0].sampling_rate.magnitude / desired_sr)
    down_sampled_anasigs = []
    time_vec = anasigs_trial[0].times.magnitude[::scaling_factor]
    for anasig in anasigs_trial:
        down_sampled_anasigs.append(anasig[::scaling_factor])
    return down_sampled_anasigs, time_vec


def construct_specgram_dict(analog_signals, events_trial, params, ch_ids):
    t_start = events_trial[0] - params['time_margin_wavelet_pre']
    t_stop = events_trial[-1] + params['time_margin_wavelet_post']
    anasigs_trial = slice_ns6_analogsignals(analog_signals, [t_start, t_stop])
    anasigs_trial_ds, time_vec_ds = downsample_ns6_anasigs(
        anasigs_trial, params)
    specgrams = compute_specgram(anasigs_trial_ds, ch_ids)
    return specgrams, time_vec_ds


def remove_powerline_noise(signals):
    def denoise_chunk(chunk, win_len, win_num):
        mean_sig = chunk.reshape([win_num, win_len]).mean(axis=0)
        noise_removed_chunk = chunk.reshape([win_num, win_len]) - mean_sig
        noise_removed_chunk = noise_removed_chunk.reshape(
            [win_len * win_num, 1])
        return noise_removed_chunk

    win_len = int(signals[0].sampling_rate.magnitude * 20 / 1000)
    win_num = 500
    chunk_len = win_num * win_len
    start_points = np.arange(0, len(signals[0]), chunk_len)
    for signal in signals:
        for stpoint in start_points[:-1]:
            chunk = signal[stpoint:stpoint + chunk_len]
            denoised_chunk = denoise_chunk(chunk, win_len, win_num)
            signal[stpoint:stpoint + chunk_len] = denoised_chunk
    return signals


def remove_analogsignals(block, keep=lambda x: False):
    '''
    Removes all AnalogSignals from a neo.Block according to keep function

        Parameters
    ----------
    block : neo.Block containing neo.AnalogSignals
    keep : function accepting neo.AnalogSignals as input and returning bool.
           Default removes all AnalogSignals contained.
           Default: lambda x: False
    '''

    for ana in block.filter(objects=neo.AnalogSignal):
        if not keep(ana):

            # removing link segment -> analogsignal
            seg = ana.segment
            ids = [i for i, a in enumerate(seg.analogsignals) if a is ana]
            for i in ids[::-1]:
                ana.segment.analogsignals.pop(i)

            # removing link channel_index -> analogsignal
            chid = ana.channel_index
            ids = [i for i, a in enumerate(chid.analogsignals) if a is ana]
            for i in ids[::-1]:
                ana.channel_index.analogsignals.pop(i)

            # removing links from analogsignal to containers
            ana.segment = None
            ana.channel_index = None

            del ana
    gc.collect()


def gen_keep_func(cond, textfile):
    sua_mua_info = np.loadtxt(textfile, delimiter=',')
    id_keep = {}
    for ch_id, num_sua, id_mua in sua_mua_info:

        if np.isnan(ch_id):
            continue

        ch_id = int(ch_id)
        num_sua = int(num_sua)
        id_mua = int(id_mua)

        if cond == 'sua':
            id_keep[ch_id] = list(range(1, num_sua + 1))

        elif cond == 'mua':
            if id_mua == 0:
                id_keep[ch_id] = []
            else:
                id_keep[ch_id] = [id_mua]

        elif cond == 'sua_mua':
            id_keep[ch_id] = list(range(1, num_sua + 1))
            if id_mua != 0:
                id_keep[ch_id].append(id_mua)

        elif cond == 'noise':
            id_keep[ch_id] = 0

        else:
            raise ValueError(
                'cond need to be either "sua", "mua", "sua_mua", or "noise"')

    def keep(spiketrain):
        ch_id = spiketrain.annotations['channel_id']
        unit_id = spiketrain.annotations['unit_id']
        if unit_id in id_keep[ch_id]:
            return True
        else:
            return False

    return keep


def remove_spiketrains(block, keep=lambda x: False, lazy=False):
    '''
    Removes all spiketrains from a neo.Block according to keep function

        Parameters
    ----------
    block : neo.Block containing neo.SpikeTrain
    keep : function accepting neo.SpikeTrain as input and returning bool.
           Default removes all SpikeTrains contained.
           Default: lambda x: False
    '''

    if lazy:
        obj = neo.io.proxyobjects.SpikeTrainProxy
    else:
        obj = neo.SpikeTrain

    for st in block.filter(objects=obj):
        if not keep(st):

            # removing link segment -> spiketrain
            seg = st.segment
            ids = [i for i, a in enumerate(seg.spiketrains) if a is st]
            for i in ids[::-1]:
                st.segment.spiketrains.pop(i)

            # removing link unit -> spiketrain
            unit = st.unit
            ids = [i for i, a in enumerate(unit.spiketrains) if a is st]
            for i in ids[::-1]:
                st.unit.spiketrains.pop(i)

            # removing links from spiketrain to containers
            st.segment = None
            st.unit = None

            del st
    gc.collect()


def extract_table(labels, code_start, code_end):
    """
    Convenience function to extract all event labels between code_start and code_end.
    For each occurence of code_start and code_end, the result contains a table with all labels between
    code_start and code_end without them included, organized in a 2D matrix with consecutive triplets of codes as rows.
    When using this for reading information tables, Column 1 contains the values while columns 0 and 2 contain flankers.

    Parameters:
    -----------
    labels: array of ints
        an array of all event labels as integers
    code_start: int
        first code of the desired sequence
    code_start: int
        last code of the desired sequence

    Returns:
    --------
    labels: array of ints
        For each occurence of code_start and code_end, the result contains a table with all labels between code_start
        and code_end without them included, organized in a 2D matrix with consecutive triplets of codes as rows. When
        using this for reading information tables, Column 1 contains the values while columns 0 and 2 contain flankers.
    """

    tables = []

    for idx_start, idx_end in zip(np.where(labels == code_start)[0],
                                  np.where(labels == code_end)[0]):
        assert ((idx_end - idx_start - 1) % 3 == 0)

        tables.append(labels[idx_start + 1:idx_end].reshape([-1, 3]))

    return np.array(tables)


def extract_sequence_dict(events, behavioral_codes):
    """
    Convenience function to extract all target sequences for one session. Target numbering matches the target
    descriptors of the session.

    Parameters:
    -----------
    events: neo.Event
        neo Event containing at least the trial protocol table part of the metadata section
    behavioral_codes: dict
        a dictionary containing all the behavioral codes in the .csv file of the session and their corresponding
        descriptions, as output by extract_behavioral_codes


    Returns:
    --------
    sequence_dict: dict
        dictionary containing all sequence IDs and the corresponding target sequences
    """

    tp_table = extract_table(events.labels.astype(int),
                             behavioral_codes['tp_table_start'],
                             behavioral_codes['tp_table_end'])
    assert len(tp_table) == 1
    tp_table = tp_table[0]

    sequence_dict = {}

    num_sequences = np.count_nonzero(
        tp_table == behavioral_codes['tag_start']) // 2

    sequence_length = len(tp_table) // num_sequences - 2

    for i in range(num_sequences):

        sequence = tp_table[:, 1][i * (sequence_length + 2) + 1:i * (
                sequence_length + 2) + sequence_length]

        if 0 in sequence:
            break

        sequence_dict[i + 1] = sequence

    return sequence_dict


class SessionInfoCollector(object):
    def __init__(self, data_dir, sorted_data_dir=None):
        self.data_dir = data_dir
        self.sorted_data_dir = sorted_data_dir
        subdir_names = [x for x in os.listdir(data_dir) if
                        os.path.isdir(os.path.join(data_dir, x))]
        self.session_info = []
        for subdir_name in sorted(subdir_names):
            info = self.parse_subdir_name(subdir_name)
            if info:
                info['filenames'] = self.collect_filenames(info)
                if 'v-nev-sorted' in info['filenames'] or 'm-nev-sorted' in \
                        info['filenames']:
                    info['units'] = 'sorted'
                self.session_info.append(info)

    def parse_subdir_name(self, subdir_name):
        tokens = re.split(r"[_\-]", subdir_name)
        if len(
                tokens) < 3:  # a valid subdir name must be composed of 3 parts separated by "_" or "-"
            return
        elif not re.match(r".\d{6}", tokens[
            0]):  # the first part must be monkey's initial + 6 digit number (date)
            return
        elif tokens[1] in ["calib"]:  # we skip calibration sessions
            return
        elif not re.match(r"\d{3}", tokens[
            2]):  # the third part must be a 3 digit number (recording order)
            return
        else:
            return dict(session_name=subdir_name, monkey=tokens[0][0],
                        year=int(tokens[0][1:3]), month=int(tokens[0][3:5]),
                        day=int(tokens[0][5:7]), task=tokens[1],
                        order=int(tokens[2]), units='unsorted')

    def collect_filenames(self, info):
        basename = '{monkey}{year:02d}{month:02d}{day:02d}-{task}-{{array}}-{order:03d}'.format(
            **info)
        subdir = os.path.join(self.data_dir, info['session_name'])
        datafiles = [x for x in os.listdir(subdir) if
                     os.path.isfile(os.path.join(subdir, x))]
        filenames = {}
        for ext in ['ns2', 'ns6', 'nev']:
            for array in ['v', 'm']:
                target_file = '.'.join([basename.format(array=array), ext])
                if target_file in datafiles:
                    filenames['-'.join([array, ext])] = target_file

        if self.sorted_data_dir is None:
            return filenames

        for array in ['v', 'm']:
            nev_files = {}
            for filename in os.listdir(self.sorted_data_dir):
                if re.match(r"{}-\d{{2}}\.nev".format(
                        basename.format(array=array)), filename):
                    tokens = re.split(r"[_\-\.]", filename)
                    nev_files[int(tokens[4])] = filename
            if len(nev_files.keys()) > 0:
                filenames['-'.join([array, ext, "sorted"])] = nev_files[
                    min(nev_files.keys())]
        return filenames

    def get_tasks(self):
        return list(set([x['task'] for x in self.session_info]))

    def get_names(self, **kwargs):
        info = self.get_info(**kwargs)
        return [x['session_name'] for x in info]

    def get_info(self, monkey="ANY", task="ANY", units="ANY"):
        info = []
        for session_info in self.session_info:
            session_monkey = session_info['monkey']
            session_task = session_info['task']
            session_units = session_info['units']
            if not (session_monkey == monkey or monkey == "ANY"):
                continue
            elif not (session_task == task or task == "ANY"):
                continue
            elif not (session_units == units or units == "ANY"):
                continue
            else:
                info.append(session_info)
        return info
