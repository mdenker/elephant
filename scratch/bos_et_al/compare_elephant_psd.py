"""
Compare the original Fig. 1E spectra of Bos et al. (2016), recreated by
``reproduce_original_psd.py``, with the PSD estimators of
``elephant.spectral``.

Normalisation
-------------
The original code computes, for the population-normalised spike count
x[n] (counts per neuron per 1 ms bin, n = 0..M-1),

    P_orig(f_k) = |sum_n x[n] exp(-2 pi i k n / M)|^2 / (M * dt),

with dt in seconds, at the positive frequencies 0.1 Hz <= f_k < 500 Hz
(no mean subtraction, rectangular window). With the population rate
r[n] = x[n] / dt (in 1/s, sampled at fs = 1/dt = 1 kHz) this is exactly the
*two-sided* periodogram density of r. Elephant returns one-sided densities,
hence

    P_orig = P_elephant(r) / 2      (for 0 < f < fs/2).

Estimators compared
-------------------
exact equivalents (must agree to floating-point precision):
  * welch_psd, boxcar window, no detrending, len_segment = M
    -> 1-window spectrum (input: x_full)
  * welch_psd, boxcar window, no detrending, no overlap, len_segment = 485
    -> 20-window spectrum (input: concatenated x_windows)
statistical equivalents (only expected to agree in shape / level):
  * welch_psd with default window (Hann), 50 % overlap, constant detrending,
    len_segment = 485
  * multitaper_psd on the full signal (NW = 4)
  * segmented_multitaper_psd, len_segment = 485, no overlap (NW = 4)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq

from elephant import spectral

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, 'original_psd.npz'))

POPS = list(d['populations'])
DT_S = float(d['bin_size_ms']) * 1e-3
FS = 1.0 / DT_S
FMAX = float(d['fmax_hz'])
LEN_WIN = d['x_windows'].shape[-1]

# inputs as population rates in 1/s, shape (time, populations)
r_full = d['x_full'].T / DT_S
r_win = d['x_windows'].reshape(8, -1).T / DT_S


def to_signal(r):
    return neo.AnalogSignal(r, units=1 / pq.s, sampling_rate=FS * pq.Hz)


def cut(freqs, psd):
    """Original frequency range and one- to two-sided conversion."""
    freqs = np.asarray(getattr(freqs, 'magnitude', freqs))
    psd = np.asarray(getattr(psd, 'magnitude', psd))
    sel = (freqs >= 0.1) & (freqs < FMAX)
    return freqs[sel], psd[:, sel] / 2.0


def run_elephant():
    est = {}
    est['welch boxcar, 1 seg (exact)'] = ('1w', cut(*spectral.welch_psd(
        to_signal(r_full), len_segment=r_full.shape[0], overlap=0,
        window='boxcar', detrend=False)))
    est['welch boxcar, 485 ms, no overlap (exact)'] = ('20w', cut(
        *spectral.welch_psd(to_signal(r_win), len_segment=LEN_WIN, overlap=0,
                            window='boxcar', detrend=False)))
    est['welch hann, 485 ms, 50% overlap'] = ('20w', cut(*spectral.welch_psd(
        to_signal(r_full), len_segment=LEN_WIN)))
    est['multitaper, full signal, NW=4'] = ('1w', cut(
        *spectral.multitaper_psd(to_signal(r_full), nw=4)))
    f, p = spectral.segmented_multitaper_psd(
        to_signal(r_full), len_segment=LEN_WIN, overlap=0, nw=4)
    if p.ndim == 3:
        # on main, segmented_multitaper_psd returns an (n, n, f) complex64
        # array (bug fixed on feature/spike-psd); take the diagonal
        p = np.real(np.diagonal(np.asarray(p), axis1=0, axis2=1).T)
    est['segmented multitaper, 485 ms, NW=4'] = ('20w', cut(f, p))
    return est


def compare(est):
    ref = {'1w': (d['freq_1w'], d['power_1w']),
           '20w': (d['freq_20w'], d['power_20w'])}
    f20, p20 = ref['20w']
    band = (f20 >= 10) & (f20 <= 400)
    print('%-42s %-4s %8s %14s %22s' % (
        'estimator', 'ref', 'n_freq', 'max rel. dev.',
        'median ratio to 20w'))
    print('%-42s %-4s %8s %14s %22s' % ('', '', '', '(vs. ref)',
                                         '(10-400 Hz, all pops)'))
    for name, (which, (f, p)) in est.items():
        f_ref, p_ref = ref[which]
        rel = np.max(np.abs(p - p_ref) / p_ref)
        p_i = np.array([np.interp(f20, f, pp) for pp in p])
        ratio = np.median(p_i[:, band] / p20[:, band])
        print('%-42s %-4s %8d %14.2e %22.3f' % (name, which, len(f), rel,
                                                ratio))
    # the exact equivalents
    f, p = est['welch boxcar, 1 seg (exact)'][1]
    assert np.allclose(f, d['freq_1w']) and np.allclose(p, d['power_1w'],
                                                        rtol=1e-8)
    f, p = est['welch boxcar, 485 ms, no overlap (exact)'][1]
    assert np.allclose(f, d['freq_20w']) and np.allclose(p, d['power_20w'],
                                                         rtol=1e-8)
    print('Exact equivalents agree (rtol=1e-8).')


def _panel_grid(title):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    fig.suptitle(title)
    for j in range(8):
        layer, ei = divmod(j, 2)
        ax = axes[ei, layer]
        ax.set_title(POPS[j])
        ax.set_xlim(10, 400)
        ax.set_yscale('log')
        if ei == 1:
            ax.set_xlabel('frequency (1/s)')
    return fig, [axes[j % 2, j // 2] for j in range(8)]


def plot_original():
    """Fig. 1E as in microcircuit.plot_fig (gray: simulation, black:
    theory, which the original code plots as sqrt(power_ana))."""
    fig, axes = _panel_grid('Recreated Bos et al. (2016) Fig. 1E '
                            '(original code, ported)')
    for j, ax in enumerate(axes):
        ax.plot(d['freq_1w'], d['power_1w'][j], color='0.8')
        ax.plot(d['freq_20w'], d['power_20w'][j], color='0.5')
        ax.plot(d['freq_ana'], np.sqrt(d['power_ana'][j]), color='k')
        ax.set_ylim(1e-6, 1)
    axes[0].set_ylabel(r'$|C(\omega)|$')
    axes[1].set_ylabel(r'$|C(\omega)|$')
    fig.tight_layout()
    out = os.path.join(HERE, 'fig1E_original_recreated.png')
    fig.savefig(out, dpi=150)
    print('Saved', out)


def plot_comparison(est):
    fig, axes = _panel_grid(
        'Fig. 1E: original 20-window spectrum vs. elephant.spectral '
        '(Elephant one-sided PSD / 2)')
    styles = {
        'welch boxcar, 485 ms, no overlap (exact)':
            dict(color='tab:red', ls='--', lw=1.2),
        'welch hann, 485 ms, 50% overlap': dict(color='tab:orange', lw=0.9),
        'multitaper, full signal, NW=4': dict(color='tab:green', lw=0.6,
                                              alpha=0.7),
        'segmented multitaper, 485 ms, NW=4': dict(color='tab:purple',
                                                   lw=0.9),
    }
    for j, ax in enumerate(axes):
        ax.plot(d['freq_20w'], d['power_20w'][j], color='0.4', lw=3,
                label='original, 20 windows')
        for name, kw in styles.items():
            f, p = est[name][1]
            ax.plot(f, p[j], label='Elephant: ' + name, **kw)
        ax.set_ylim(1e-5, 1)
    axes[0].set_ylabel('two-sided PSD of population rate (1/s)')
    axes[1].set_ylabel('two-sided PSD of population rate (1/s)')
    axes[0].legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    out = os.path.join(HERE, 'fig1E_original_vs_elephant.png')
    fig.savefig(out, dpi=150)
    print('Saved', out)


def plot_exact_deviation(est):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for k, (name, which) in enumerate([
            ('welch boxcar, 1 seg (exact)', '1w'),
            ('welch boxcar, 485 ms, no overlap (exact)', '20w')]):
        f, p = est[name][1]
        p_ref = d['power_' + which]
        for j in range(8):
            ax[k].plot(f, np.abs(p[j] / p_ref[j] - 1) + 1e-18, lw=0.6,
                       label=POPS[j])
        ax[k].set_yscale('log')
        ax[k].set_title('Elephant %s vs. original %s' % (name, which),
                        fontsize=9)
        ax[k].set_xlabel('frequency (1/s)')
    ax[0].set_ylabel('|P_elephant / P_original - 1|')
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(HERE, 'exact_equivalents_deviation.png')
    fig.savefig(out, dpi=150)
    print('Saved', out)


if __name__ == '__main__':
    estimates = run_elephant()
    compare(estimates)
    plot_original()
    plot_comparison(estimates)
    plot_exact_deviation(estimates)
