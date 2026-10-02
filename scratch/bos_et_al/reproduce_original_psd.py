"""
Reproduce the simulated population-rate spectra of Fig. 1E in

    Bos H, Diesmann M, Helias M (2016) Identifying Anatomical Origins of
    Coexisting Oscillations in the Cortical Microcircuit.
    PLoS Comput Biol 12(10): e1005132. doi:10.1371/journal.pcbi.1005132

using the original analysis code in ``bos_et_al_code/`` (``read_sim.py`` and
``microcircuit.get_spec_sim``), ported to Python 3 with the following
changes only:

* ``pop/2`` -> ``pop // 2`` (Python 2 integer division), ``print`` syntax;
* every spike file is parsed once with ``numpy.loadtxt`` and then filtered
  with exactly the same (strict / half-open) conditions as the original
  line-by-line loop, instead of re-reading all files for every window;
* the original ``powerspec`` is kept verbatim (minus the unused smoothing
  branch).

The result is checked against the values that the authors stored in
``bos_et_al_code/results.h5`` (group ``fig_microcircuit``).

Saved to ``original_psd.npz``:

* inputs: per-population binned, population-normalised spike counts
  (``x_full``: one 9700 ms window as in the 1-window spectrum;
  ``x_windows``: the 20 windows of 485 ms as in the 20-window spectrum),
  bin size, population sizes, window boundaries;
* outputs: frequencies and spectra for 1 and 20 windows, exactly as in the
  original code, plus the analytical (mean-field) spectra from results.h5.
"""
import glob
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(HERE, '..', '..', 'bos_et_al_code')
DATA_DIR = os.path.join(CODE_DIR, 'data', 'data_microcircuit')
RESULTS_H5 = os.path.join(CODE_DIR, 'results.h5')
OUT_FILE = os.path.join(HERE, 'original_psd.npz')

POPULATIONS = ['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I']

# parameters of microcircuit.plot_fig() / get_spec_sim()
T_SIM = 10000.0      # ms
T_MIN = 300.0        # ms, initial transient discarded
BIN_SIZE = 1.0       # ms
FMAX = 500.0         # Hz
NR_WINDOWS = (1, 20)


# --------------------------------------------------------------------------
# Port of read_sim.py
# --------------------------------------------------------------------------
def get_gids():
    data = np.loadtxt(os.path.join(DATA_DIR, 'population_GIDs.dat'))
    gid_mins = data[:, 0].astype(int)
    pop_sizes = data[:, 1].astype(int) - gid_mins + 1
    return gid_mins, pop_sizes


def load_pop_spikes(pop):
    """All (gid, time) pairs of one population, all files concatenated."""
    files = sorted(glob.glob(os.path.join(
        DATA_DIR, 'spikes_%d_%d*' % (pop // 2, pop % 2))))
    data = [np.loadtxt(f, ndmin=2) for f in files]
    data = np.concatenate([d for d in data if d.size > 0])
    return data[:, 0], data[:, 1]


def read_all_neurons(gids, times, pop, T, tmin):
    """Same selection as read_sim.read_all_neurons (gid_lims=None).

    Note: the original uses ``gid_max = gid_min + nr_gids - 1`` together
    with ``gid < gid_max``, i.e. the last neuron of each population is
    excluded. This is reproduced as is.
    """
    gid_mins, pop_sizes = get_gids()
    gid_min = gid_mins[pop]
    gid_max = gid_mins[pop] + pop_sizes[pop] - 1
    gids_int = gids.astype(int)
    sel = (gids_int >= gid_min) & (gids_int < gid_max) & \
        (times > tmin) & (times < T)
    return times[sel]


def cut_arrays(x, y, xmin, xmax):
    y = y[np.where(x < xmax)]
    x = x[np.where(x < xmax)]
    y = y[np.where(x >= xmin)]
    x = x[np.where(x >= xmin)]
    return x, y


def instantaneous_spike_count(data, tbin, tmin=None, tmax=None):
    assert tmin < tmax
    bins = np.arange(tmin, tmax + tbin, tbin)
    hist = np.array([np.histogram(x, bins)[0] for x in data])
    return bins[:-1], hist


def calculate_fft(data, tbin):
    n = len(data[0])
    return np.fft.fftfreq(n, tbin * 1e-3), np.fft.fft(data, axis=1)


def powerspec(data, tbin):
    freq, DATA = calculate_fft(data, tbin)
    T = tbin * len(freq)
    POW = np.power(np.abs(DATA), 2)
    POW *= 1. / T * 1e3  # normalization, power independent of T
    return freq, POW


def create_spec(spike_times, T, tmin, pop_size=1, bin_size=1.0):
    """read_sim.create_spec with mode=None; also returns the input signal."""
    bins, hist = instantaneous_spike_count([spike_times], bin_size,
                                           tmin, tmax=T)
    hist = [hist[0] / float(pop_size)]
    freq, p = powerspec(hist, bin_size)
    return freq, p[0], hist[0]


def get_spec(gids, times, pop, T, tmin, fmax, bin_size):
    _, pop_sizes = get_gids()
    N = float(pop_sizes[pop])
    spike_times = read_all_neurons(gids, times, pop, T, tmin)
    freq, p, x = create_spec(spike_times, T, tmin, N, bin_size=bin_size)
    freq, p = cut_arrays(freq, p, 0.1, fmax)
    return freq, p, x


# --------------------------------------------------------------------------
# Port of microcircuit.get_spec_sim
# --------------------------------------------------------------------------
def get_spec_sim(spikes, nr_window, bin_size):
    power_sim, inputs = [], []
    dt = (T_SIM - T_MIN) / nr_window
    Ts = [T_MIN + i * dt for i in range(nr_window)]
    for pop in range(8):
        gids, times = spikes[pop]
        psim = 0
        x_pop = []
        for T in Ts:
            fsim, p_sim, x = get_spec(gids, times, pop, T + dt, T,
                                      fmax=FMAX, bin_size=bin_size)
            psim += np.asarray(p_sim) / float(len(Ts))
            x_pop.append(x)
        power_sim.append(psim)
        inputs.append(np.array(x_pop))
    return np.array(power_sim), np.asarray(fsim), np.array(inputs), Ts, dt


def load_stored_results():
    out = {}
    with h5py.File(RESULTS_H5, 'r') as f:
        g = f['fig_microcircuit']
        for nw in NR_WINDOWS:
            out[nw] = (g['%d/freq_sim' % nw][()],
                       np.array([g['%d/power%d' % (nw, p)][()]
                                 for p in range(8)]))
        out['freq_ana'] = g['freq_ana'][()]
        out['power_ana'] = g['power_ana'][()]
    return out


def main():
    _, pop_sizes = get_gids()
    print('Reading spike files ...')
    spikes = [load_pop_spikes(pop) for pop in range(8)]
    stored = load_stored_results()

    results = {}
    for nw in NR_WINDOWS:
        power, freq, x, Ts, dt = get_spec_sim(spikes, nw, BIN_SIZE)
        f_ref, p_ref = stored[nw]
        assert np.array_equal(freq.shape, f_ref.shape)
        rel = np.max(np.abs(power - p_ref) / np.abs(p_ref))
        print('%2d window(s): %d frequencies, max. rel. deviation from '
              'results.h5: %.2e' % (nw, len(freq), rel))
        assert np.allclose(freq, f_ref, rtol=1e-12, atol=0)
        assert np.allclose(power, p_ref, rtol=1e-10, atol=0)
        results[nw] = dict(freq=freq, power=power, x=x, Ts=np.array(Ts),
                           dt=dt)

    # x_windows concatenated vs. x_full: they differ only by spikes that
    # fall exactly on a window boundary (excluded by the strict '<', '>')
    x_full = results[1]['x'][:, 0, :]
    x_windows = results[20]['x']
    n_diff = np.sum(x_windows.reshape(8, -1) != x_full)
    print('Bins differing between 1-window and concatenated 20-window '
          'input: %d of %d' % (n_diff, x_full.size))

    np.savez(
        OUT_FILE,
        populations=np.array(POPULATIONS),
        pop_sizes=pop_sizes,
        bin_size_ms=BIN_SIZE, t_min_ms=T_MIN, t_stop_ms=T_SIM, fmax_hz=FMAX,
        # inputs
        x_full=x_full,                      # (8, 9700)
        x_windows=x_windows,                # (8, 20, 485)
        window_starts_ms=results[20]['Ts'],
        window_length_ms=results[20]['dt'],
        # outputs
        freq_1w=results[1]['freq'], power_1w=results[1]['power'],
        freq_20w=results[20]['freq'], power_20w=results[20]['power'],
        freq_ana=stored['freq_ana'], power_ana=stored['power_ana'],
    )
    print('Saved', OUT_FILE)


if __name__ == '__main__':
    main()
