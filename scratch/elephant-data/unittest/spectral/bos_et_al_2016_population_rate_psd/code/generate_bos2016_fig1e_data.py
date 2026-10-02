"""
Generate the ground-truth data for validating Elephant's PSD estimators
against the simulated population-rate spectra of Fig. 1E in

    Bos H, Diesmann M, Helias M (2016) Identifying Anatomical Origins of
    Coexisting Oscillations in the Cortical Microcircuit.
    PLoS Comput Biol 12(10): e1005132. doi:10.1371/journal.pcbi.1005132

The spectra are computed with the original analysis code of the authors
(module ``read_sim.py`` of the original code and data package). The script

1. obtains the original code and data package, either from a local checkout
   (``--bos-code``) or from Zenodo (``BOS_CODE_ZENODO_DOI``, see below);
2. computes the 1-window and 20-window spectra of all 8 populations with the
   original functions ``read_sim.create_spec`` (binning, normalisation and
   FFT-based power spectrum) and ``read_sim.cut_arrays``, following the loop
   of ``microcircuit.get_spec_sim``;
3. verifies the result against the spectra the authors stored in
   ``results.h5`` of the package;
4. writes the input (binned population activity) and the intended output
   (spectra) to ``../data/``;
5. plots Fig. 1E (simulated and theoretical spectra; the theory is
   plotted as sqrt of the stored ``power_ana`` = |C_ii|^2, as in the
   original ``microcircuit.plot_fig``) to
   ``../fig1E_reproduction.png``.

Only two parts of the original code had to be re-implemented, because they
are Python 2 code that does not run under Python 3:

* ``read_sim.read_all_neurons`` (reads the spike files; uses ``pop/2`` as
  integer division and indexes the result of ``map``). The port applies
  exactly the same selection, including the original exclusion of the last
  neuron of each population (``gid < gid_min + N - 1``);
* the loop over windows in ``microcircuit.get_spec_sim`` and the plotting
  code of panel E in ``microcircuit.plot_fig`` (the module ``microcircuit``
  cannot be imported: Python 2 ``print`` statements, ``h5py_wrapper``).

Usage::

    python generate_bos2016_fig1e_data.py --bos-code /path/to/bos_et_al_code
    python generate_bos2016_fig1e_data.py      # download from Zenodo
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
import zipfile

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --------------------------------------------------------------------------
# Location of the original code and data package
# --------------------------------------------------------------------------
# DOI of the Zenodo record holding the original code and data package of
# Bos et al. (2016) (content of ``bos_et_al_code``: *.py, results.h5,
# data/data_microcircuit/). Not yet published -- set this once available.
BOS_CODE_ZENODO_DOI = None
# Name of the top-level folder of the package inside the Zenodo archive, if
# the archive does not unpack directly into the package root.
BOS_CODE_ARCHIVE_ROOT = None

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_OUT = os.path.join(HERE, '..', 'data')
FIG_OUT = os.path.join(HERE, '..', 'fig1E_reproduction.png')
INPUT_FILE = os.path.join(DATA_OUT, 'bos2016_fig1e_input.npz')
OUTPUT_FILE = os.path.join(DATA_OUT, 'bos2016_fig1e_output.npz')

POPULATIONS = ['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I']

# parameters of microcircuit.plot_fig() / microcircuit.get_spec_sim()
FOLDER = 'data_microcircuit'
T_SIM = 10000.0      # ms
T_MIN = 300.0        # ms
BIN_SIZE = 1.0       # ms
FMAX = 500.0         # Hz
NR_WINDOWS = (1, 20)


# --------------------------------------------------------------------------
# Obtaining the original package
# --------------------------------------------------------------------------
def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch_from_zenodo(doi, cache_dir):
    """Download and unpack all files of the Zenodo record with `doi`.

    Returns the path of the unpacked package root.
    """
    if doi is None:
        raise RuntimeError(
            'BOS_CODE_ZENODO_DOI is not set yet. Provide a local checkout '
            'of the original code and data package with --bos-code.')
    # resolve the DOI to the Zenodo record, then query the record via API
    with urllib.request.urlopen('https://doi.org/' + doi) as resp:
        record_url = resp.geturl()
    record_id = record_url.rstrip('/').split('/')[-1]
    api_url = 'https://zenodo.org/api/records/' + record_id
    with urllib.request.urlopen(api_url) as resp:
        record = json.load(resp)

    target = os.path.join(cache_dir, 'zenodo_' + record_id)
    os.makedirs(target, exist_ok=True)
    for entry in record['files']:
        name = entry['key']
        path = os.path.join(target, name)
        if not os.path.exists(path):
            print('Downloading', name, '...')
            urllib.request.urlretrieve(entry['links']['self'], path)
        algo, _, expected = entry.get('checksum', '').partition(':')
        if algo == 'md5' and _md5(path) != expected:
            raise ValueError('MD5 mismatch for downloaded file ' + path)
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                z.extractall(target)
    if BOS_CODE_ARCHIVE_ROOT is not None:
        target = os.path.join(target, BOS_CODE_ARCHIVE_ROOT)
    return target


def import_read_sim(bos_code):
    """Import the original module ``read_sim`` from the package root."""
    sys.path.insert(0, os.path.abspath(bos_code))
    import read_sim
    read_sim.path_base = os.path.join(os.path.abspath(bos_code), 'data', '')
    return read_sim


# --------------------------------------------------------------------------
# Python 3 port of read_sim.read_all_neurons (spike file reader)
# --------------------------------------------------------------------------
def load_pop_spikes(rs, pop):
    """All (gid, time) pairs of population `pop`, all files concatenated."""
    files = sorted(glob.glob(rs.path_base + FOLDER + '/spikes_%d_%d*'
                             % (pop // 2, pop % 2)))
    data = [np.loadtxt(f, ndmin=2) for f in files]
    data = np.concatenate([d for d in data if d.size > 0])
    return data[:, 0], data[:, 1]


def read_all_neurons(rs, spikes, pop, T, tmin):
    """Selection of read_sim.read_all_neurons(FOLDER, pop, T, tmin)."""
    gids, times = spikes
    gid_mins, pop_sizes = rs.get_gids(FOLDER)
    gid_min = gid_mins[pop]
    gid_max = gid_mins[pop] + pop_sizes[pop] - 1
    gids_int = gids.astype(int)
    sel = (gids_int >= gid_min) & (gids_int < gid_max) & \
        (times > tmin) & (times < T)
    return list(times[sel])


# --------------------------------------------------------------------------
# microcircuit.get_spec_sim, with read_sim.get_spec inlined so that the
# input of the spectrum calculation can be stored as well
# --------------------------------------------------------------------------
def get_spec_sim(rs, spikes, nr_window, bin_size):
    _, pop_sizes = rs.get_gids(FOLDER)
    power_sim, inputs = [], []
    dt = (T_SIM - T_MIN) / nr_window
    Ts = [T_MIN + i * dt for i in range(nr_window)]
    for pop in range(8):
        N = float(pop_sizes[pop])
        psim = 0
        x_pop = []
        for T in Ts:
            # read_sim.get_spec(folder, pop, T+dt, T, fmax, bin_size)
            spike_times = read_all_neurons(rs, spikes[pop], pop, T + dt, T)
            fsim, p_sim, _ = rs.create_spec(spike_times, T + dt, T, N,
                                            bin_size=bin_size)
            fsim, p_sim = rs.cut_arrays(fsim, p_sim, 0.1, FMAX)
            psim += np.asarray(p_sim) / float(len(Ts))
            # input of the spectrum: identical to the binning and
            # normalisation in read_sim.create_spec
            _, hist = rs.instantaneous_spike_count([spike_times], bin_size,
                                                   T, tmax=T + dt)
            x_pop.append(hist[0] / N)
        power_sim.append(psim)
        inputs.append(np.array(x_pop))
    return (np.asarray(fsim), np.array(power_sim), np.array(inputs),
            np.array(Ts), dt)


def load_stored_results(bos_code):
    out = {}
    with h5py.File(os.path.join(bos_code, 'results.h5'), 'r') as f:
        g = f['fig_microcircuit']
        for nw in NR_WINDOWS:
            out[nw] = (g['%d/freq_sim' % nw][()],
                       np.array([g['%d/power%d' % (nw, p)][()]
                                 for p in range(8)]))
        out['freq_ana'] = g['freq_ana'][()]
        out['power_ana'] = g['power_ana'][()]
    return out


# --------------------------------------------------------------------------
# Figure: panel E of microcircuit.plot_fig
# --------------------------------------------------------------------------
def plot_fig1e(freq_1w, power_1w, freq, power, freq_ana, psd_ana):
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.5), sharex=True,
                             sharey=True)
    for layer in [0, 1, 2, 3]:
        for pop in [0, 1]:
            j = layer * 2 + pop
            ax = axes[pop, layer]
            ax.plot(freq_1w, power_1w[j], color=(0.8, 0.8, 0.8))
            ax.plot(freq, power[j], color=(0.5, 0.5, 0.5))
            ax.plot(freq_ana, psd_ana[j], color='black',
                    zorder=2)
            ax.set_xlim([10.0, 400.0])
            ax.set_ylim([1e-6, 1e0])
            ax.set_yscale('log')
            ax.set_xticks([100, 200, 300])
            ax.set_title(POPULATIONS[j])
            if pop == 1:
                ax.set_xlabel(r'frequency (1/$s$)')
    for ax in axes[:, 0]:
        ax.set_yticks([1e-5, 1e-3, 1e-1])
        ax.set_ylabel(r'$|C(\omega)|$')
    fig.suptitle('Bos et al. (2016), Fig. 1E, recomputed with the original '
                 'code', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    print('Saved', os.path.normpath(FIG_OUT))


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--bos-code', default=None,
                        help='root of a local copy of the original code and '
                             'data package (default: download from Zenodo)')
    parser.add_argument('--cache-dir',
                        default=os.path.join(tempfile.gettempdir(),
                                             'bos2016_code'),
                        help='download directory for the Zenodo record')
    args = parser.parse_args()

    bos_code = args.bos_code or fetch_from_zenodo(BOS_CODE_ZENODO_DOI,
                                                  args.cache_dir)
    rs = import_read_sim(bos_code)
    _, pop_sizes = rs.get_gids(FOLDER)

    print('Reading spike files ...')
    spikes = [load_pop_spikes(rs, pop) for pop in range(8)]
    stored = load_stored_results(bos_code)

    res = {}
    for nw in NR_WINDOWS:
        freq, power, x, Ts, dt = get_spec_sim(rs, spikes, nw, BIN_SIZE)
        f_ref, p_ref = stored[nw]
        rel = np.max(np.abs(power - p_ref) / np.abs(p_ref))
        print('%2d window(s): %4d frequencies, max. relative deviation from '
              'results.h5: %.1e' % (nw, len(freq), rel))
        np.testing.assert_allclose(freq, f_ref, rtol=1e-12)
        np.testing.assert_allclose(power, p_ref, rtol=1e-10)
        res[nw] = dict(freq=freq, power=power, x=x, Ts=Ts, dt=dt)

    os.makedirs(DATA_OUT, exist_ok=True)
    np.savez_compressed(
        INPUT_FILE,
        populations=np.array(POPULATIONS),
        population_sizes=pop_sizes,
        bin_size_ms=BIN_SIZE,
        t_start_ms=T_MIN,
        t_stop_ms=T_SIM,
        x_full=res[1]['x'][:, 0, :],
        x_windows=res[20]['x'],
        window_starts_ms=res[20]['Ts'],
        window_length_ms=res[20]['dt'],
    )
    np.savez_compressed(
        OUTPUT_FILE,
        populations=np.array(POPULATIONS),
        freq_1w=res[1]['freq'], power_1w=res[1]['power'],
        freq_20w=res[20]['freq'], power_20w=res[20]['power'],
        # mean-field theory: meanfield/analytics.py (spec) stores
        # |C_ii(omega)|^2, the square of the theoretical spectrum
        # C = Q D Q^+ with D = diag(rate / N); psd_ana = C_ii is the
        # two-sided population-rate PSD (1/s), same scale as power_*w
        freq_ana=stored['freq_ana'], psd_ana=np.sqrt(stored['power_ana']),
        power_ana_stored=stored['power_ana'],
    )
    for path in (INPUT_FILE, OUTPUT_FILE):
        print('Saved %s (md5 %s)' % (os.path.normpath(path), _md5(path)))

    plot_fig1e(res[1]['freq'], res[1]['power'], res[20]['freq'],
               res[20]['power'], stored['freq_ana'],
               np.sqrt(stored['power_ana']))


if __name__ == '__main__':
    main()
