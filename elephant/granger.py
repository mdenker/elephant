import pickle as pkl
from collections import defaultdict

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
from tqdm import tqdm
from spectral_connectivity import Multitaper, Connectivity

from elephant.load_routine import get_epochs, cut_segment_by_epoch, \
    get_events, add_epoch


class CausalBlock(neo.Block):
    def __init__(self, block, causal_labels, t_start=0 * pq.ms, name=None):
        super(CausalBlock, self).__init__(name=name)
        self.block = block
        self.causal_labels = causal_labels
        self.t_start = t_start
        if name is None:
            name = ' '.join(causal_labels)
        self.name = name

        # get epochs of all successful attempts
        successful_attempts = \
            get_epochs(block, successful=True, epoch_category='All Attempts')[
                0]

        attempts_cut = cut_segment_by_epoch(block.segments[1],
                                            successful_attempts)
        self.annotations = block.annotations
        self.segments = attempts_cut
        self.create_relationship()


def pairwise_granger_causality(*signal_segments):
    n_signals = len(signal_segments)
    if n_signals < 2:
        raise ValueError(
            "Input list should have at least 2 analog signals/arrays")
    sampl_freq = signal_segments[0].sampling_rate.rescale('Hz')
    n_trials = len(signal_segments[0])  # n_trials per segment
    n_time_samples = len(signal_segments[0][0])
    # combined_matrix = np.zeros((n_time_samples, n_trials, n_signals))
    combined_matrix = []
    for segments in signal_segments:
        lfps = segments.filter(signal_type="LFP", objects="AnalogSignal")[0]
        lfps = lfps.magnitude
        combined_matrix.append(lfps)
    combined_matrix = np.stack(combined_matrix, axis=1)
    multitaper = Multitaper(combined_matrix,
                            sampling_frequency=sampl_freq)
    connectivity = Connectivity.from_multitaper(multitaper)
    granger = connectivity.pairwise_spectral_granger_prediction()
    return granger
