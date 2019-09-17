import neo
import numpy as np
from spectral_connectivity import Multitaper


def multitaper_from_analog_signals(analog_signals, **kwargs):
    """
    Parameters
    ----------
    analog_signals: list
        List of trials of length n_trials.
        Each trial is either:
            * a list of AnalogSignals, each of length n_signals;
              each AnalogSignal is a list of n_time_samples time points.
            * AnalogSignal matrix of shape (n_time_samples, n_signals).
        In either case, each trial should have at least two signals
        (n_signals > 1).
    **kwargs:
        `Multitaper` constructor arguments.
        Consider changing the `time_halfbandwidth_product` (default is 3).

    Returns
    -------
    multitaper: Multitaper
        Multitaper object, used next in `Connectivity.from_multitaper()` to
        estimate power spectrum, Granger causality, etc., in frequency domain.
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
    sampl_freq = sorted(sampl_freq)
    if len(sampl_freq) > 1:
        # todo: consider rescaling all AnalogSignals to a fixed sampling rate
        raise ValueError("AnalogSignals have different sampling rates: "
                         "{rates}. Required unique.".format(rates=sampl_freq))
    combined_matrix = np.stack(combined_matrix, axis=1)
    if combined_matrix.shape[1] == 1:
        combined_matrix = np.squeeze(combined_matrix, axis=1)
    multitaper = Multitaper(combined_matrix,
                            sampling_frequency=sampl_freq[0],
                            **kwargs)
    return multitaper
