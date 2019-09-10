import os

import neo
import quantities as pq
import pickle

from elephant.load_routine import add_epoch, cut_segment_by_epoch,\
    get_events, get_epochs
from elephant.granger import pairwise_granger_causality

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def load_blocks():
    blocks = {}
    for fname in ['motor', 'visual']:
        fpath = os.path.join(DATA_DIR, fname + '.nix')
        with neo.io.NixIO(fpath, 'ro') as f:
            blocks[fname] = f.read_block()
    return blocks


def get_attempts(block):
    # get epochs of all successful attempts
    successful_attempts = get_epochs(block,
                                     successful=True,
                                     epoch_category='All Attempts')[0]

    # create a new block of cut attempts
    # todo: why block.segments[1] ?
    attempts_cut = cut_segment_by_epoch(block.segments[1], successful_attempts)
    attempts_block = neo.Block()
    attempts_block.annotations = block.annotations
    attempts_block.segments = attempts_cut
    attempts_block.create_relationship()

    return attempts_block


def get_interesting_events(block, labels_of_interest):
    # fixme: saving and loading a neo block in nixIO drops the information
    # needed to run get_events properly (returns an empty list).
    with open(os.path.join(DATA_DIR, 'interesting_events.pkl'), 'rb') as f:
        interesting_events = pickle.load(f)
    return interesting_events

    interesting_events = []
    for segment in block.segments:
        _events = get_events(segment, name='DecodedEvents',
                             labels=labels_of_interest)[0]
        interesting_events.append(_events)
    return interesting_events


def granger_example():
    channels = {
        'motor': 2,  # M1
        'visual': 24  # V1
    }

    # fixme: does not work if labels_of_interest is a tuple!
    labels_of_interest = ['target_02_on', 'target_03_on', 'target_04_on']

    blocks = load_blocks()

    # no matter which block segments to use: motor or visual
    interesting_events = get_interesting_events(
        block=blocks['visual'],
        labels_of_interest=labels_of_interest)
    print("Interesting events ({n_trials} trials): ".format(
        n_trials=len(interesting_events)) +
        "{interesting_events}".format(interesting_events=interesting_events))

    t_start_target_on = 0 * pq.ms
    time_window_target_on = 200 * pq.ms
    t_post = t_start_target_on + time_window_target_on + 1 * pq.ms

    segments_cut = {}
    for area_name, block in blocks.items():
        segments_cut[area_name] = []
        assert len(interesting_events) == len(block.segments), \
            "Number of trials in interesting_events does not match with the" \
            "number of trials in block.segments"
        for segment_id, segment_trial in enumerate(block.segments):
            target_on_epochs = add_epoch(segment_trial, attach_result=False,
                                         event1=interesting_events[segment_id],
                                         pre=t_start_target_on,
                                         post=t_post,
                                         name='targets_on')
            # target_on_epochs has 3 epochs (time points)

            target_on_cut = cut_segment_by_epoch(segment_trial,
                                                 epoch=target_on_epochs,
                                                 reset_time=False)
            # target_on_cut has 3 cut segments
            segments_cut[area_name].extend(target_on_cut)
    granger = pairwise_granger_causality(segments_cut['motor'],
                                         segments_cut['visual'])
    return granger


if __name__ == '__main__':
    gr = granger_example()
