import neo
import numpy as np
from spectral_connectivity import Multitaper


def multitaper_from_analog_signals(analog_signals, **kwargs):
    """
    Constructs a matrix of `(n_time_samples, n_trials, n_signals)` and passes
    it to `Multitaper`.

    Parameters
    ----------
    analog_signals: list
        List of trials of length `n_trials`.
        Each trial is either:
            * a list of `n_signals` AnalogSignals, where
              each AnalogSignal is a list of `n_time_samples` time points.
            * AnalogSignal matrix of shape `(n_time_samples, n_signals)`.
        In either case, each trial should have at least two signals
        (n_signals > 1).
    **kwargs:
        `Multitaper` constructor arguments.
        Consider changing the `time_halfbandwidth_product` (default is 3).

    Returns
    -------
    multitaper: Multitaper
        `Multitaper` object, used next in `Connectivity.from_multitaper()` to
        estimate power spectrum, Granger causality, etc., in the frequency
        domain.
    """
    def rate_in_hz(rate):
        return float(rate.rescale('Hz').magnitude)

    sample_freq = set()
    combined_matrix = []
    for trial in analog_signals:
        if isinstance(trial, neo.AnalogSignal):
            # a matrix of shape (n_time_samples, n_signals)
            sample_freq.add(rate_in_hz(trial.sampling_rate))
            lfps = trial.magnitude
        elif len(trial) > 0 and isinstance(trial[0], neo.AnalogSignal):
            # a list of AnalogSignals of shape (n_time_samples, 1)
            sample_freq.add(rate_in_hz(trial[0].sampling_rate))
            lfps = np.stack([sig.magnitude for sig in trial], axis=1)
            lfps = np.squeeze(lfps)
        else:
            raise ValueError("Trials should be of type neo.AnalogSignal")
        combined_matrix.append(lfps)
    sample_freq = sorted(sample_freq)
    if len(sample_freq) > 1:
        # as an alternative, rescale all AnalogSignals to a fixed sampling rate
        raise ValueError("AnalogSignals have different sampling rates: "
                         "{rates}. Required unique.".format(rates=sample_freq))
    combined_matrix = np.stack(combined_matrix, axis=1)
    if combined_matrix.shape[1] == 1:
        # remove trial dimension in case of a single trial
        combined_matrix = np.squeeze(combined_matrix, axis=1)
    multitaper = Multitaper(combined_matrix,
                            sampling_frequency=sample_freq[0],
                            **kwargs)
    return multitaper
