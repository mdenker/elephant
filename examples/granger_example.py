import os
import warnings

import matplotlib.pyplot as plt
import neo
import quantities as pq

from elephant.granger import pairwise_granger_causality
from elephant.load_routine import add_epoch, cut_segment_by_epoch, \
    get_events

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def load_blocks():
    blocks = {}
    for fname in ['motor', 'visual']:
        fpath = os.path.join(DATA_DIR, fname + '.pkl')
        f = neo.io.PickleIO(fpath)
        blocks[fname] = f.read_block()
    return blocks


def block_from_segment_epochs(segments, epochs, annotations=None,
                              reset_time=False):
    segments_cut = cut_segment_by_epoch(segments, epochs,
                                        reset_time=reset_time)
    block = neo.Block()
    if annotations is not None:
        block.annotations = annotations
    block.segments = segments_cut
    block.create_relationship()
    return block


def get_interesting_events(block, labels_of_interest):
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
            block_cut = block_from_segment_epochs(
                segment_trial, epochs=target_on_epochs,
                annotations=block.annotations)
            lfps = block_cut.filter(signal_type="LFP", objects="AnalogSignal")
            # select the channel
            lfps = [lfp[:, channels[area_name]] for lfp in lfps]
            segments_cut[area_name].extend(lfps)
    analog_signal_trials = zip(*segments_cut.values())
    granger, freq = pairwise_granger_causality(analog_signal_trials)
    plot_causalities(granger, freq)


def granger_resting_state():
    fpath = os.path.join(DATA_DIR, "block_rst.pkl")
    f = neo.io.PickleIO(fpath)
    block = f.read_block()
    analog_signals = block.filter(objects="AnalogSignal")
    # take first 2 signals (channels)
    analog_signals = analog_signals[:2]
    analog_signals = [asig[:10000] for asig in analog_signals]
    granger, freq = pairwise_granger_causality([analog_signals])
    plot_causalities(granger, freq)


def plot_causalities(granger, frequencies):
    n_signals = granger.shape[2]
    if granger.shape[0] > 1:
        warnings.warn(
            "Granger causality has multiple time windows (n_time_windows > 1)."
            "Plotting the causalities in the first window only. Consider"
            "computing Granger causality across all time points at once.")
    for signal1_id in range(granger.shape[-1]):
        for signal2_id in range(granger.shape[-1]):
            ax = plt.subplot(n_signals, n_signals,
                             signal1_id * n_signals + signal2_id + 1)
            ax.plot(frequencies, granger[0, :, signal1_id, signal2_id].T)
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Causality')
            ax.set_xlim([0, 100])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # granger_resting_state()
    granger_example()
