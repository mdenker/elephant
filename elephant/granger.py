import warnings

import neo
import numpy as np
from spectral_connectivity import Multitaper, Connectivity


def pairwise_granger_causality(analog_signals):
    """
    Parameters
    ----------
    analog_signals: iterable object
        Iterable of trials of length n_trials.
        Each trial is either:
            * a list of AnalogSignals of length n_signals;
              each AnalogSignal is a list of n_time_samples time points.
            * AnalogSignal matrix of shape (n_time_samples, n_signals).
        In either case, each trial should have at least two signals
        (n_signals > 1).

    Returns
    -------
    granger: np.ndarray
        Pairwise Granger Causality matrix of shape
        (n_time_windows, n_frequencies, n_signals, n_signals).
        `n_frequencies` depends on the arguments, specified for Multitaper.

    """
    sampl_freq = set()
    rate_in_hz = lambda rate: float(rate.rescale('Hz').magnitude)

    # combined_matrix = np.zeros((n_time_samples, n_trials, n_signals))
    combined_matrix = []
    for trial in analog_signals:
        # some AnalogSignals might have a shape of (n_time_samples, 1)
        if isinstance(trial, neo.AnalogSignal) and trial.ndim == 2 \
                and trial.shape[1] > 1:
            lfps = trial
            sampl_freq.add(rate_in_hz(trial.sampling_rate))
        else:
            # stack a list of AnalogSignals column wise
            sampl_freq.add(rate_in_hz(trial[0].sampling_rate))
            lfps = np.hstack(trial)
        # lfps shape: (n_time_samples, n_signals)
        combined_matrix.append(lfps.magnitude)
    sampl_freq = sorted(sampl_freq, reverse=True)
    if len(sampl_freq) > 1:
        warnings.warn("AnalogSignals have different sampling rates: "
                      "{rates}. Chose the highest one.".format(
                       rates=sampl_freq))
    combined_matrix = np.stack(combined_matrix, axis=1)
    if combined_matrix.shape[1] == 1:
        combined_matrix = np.squeeze(combined_matrix, axis=1)
    print("Time series Multitaper input shape "
          "(n_time_samples, n_trials, n_signals): {shape}".format(
           shape=combined_matrix.shape))
    multitaper = Multitaper(combined_matrix,
                            sampling_frequency=sampl_freq[0],
                            time_halfbandwidth_product=3)
    connectivity = Connectivity.from_multitaper(multitaper)
    granger = connectivity.pairwise_spectral_granger_prediction()
    return granger, connectivity.frequencies
