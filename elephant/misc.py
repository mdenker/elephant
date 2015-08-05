import numpy
import quantities as pq
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit
from functools import partial
import neo
import timeit
import unitary_event_analysis as ue
import elephant.spike_train_generation as stg


def load(filename):
    '''
    Load a .npy or .npz archive and return the internal item.

    Parameters
    ----------
    filename : str
        the file to be loaded, including the path to the file.


    Returns
    -------
    dict
        a python dictionary, with the following key:
        * 'st': a list of SpikeTrains (one per trial)
        * '__doc__': a description of the data
        * '__version__': the Matlab version used to generate the data
        * '__header__': None
        * 'sptrmat': matrix of binned spike trains
    '''
    return numpy.load(filename).item()


def _rescale(x, q):
    '''
    Rescale a list of quantity objects to the desired quantity

    arguments:
    ----------
        x: list of Quantity objects
            the list of elements to rescale

        q: a Quantity
            the quantity to which to rescale the elements of x

    returns:
    --------
        a copy of x with all elements rescaled to quantity q
    '''
    x_rescaled = copy.deepcopy(x)
    for i, xx in enumerate(x):
        x_rescaled[i] = xx.rescale(q)
    return x_rescaled


def add_raster(fig, (n_row, n_col, panel), sts, ms=4, title='',
               xlabel='', ylabel=''):
    '''
    Add a raster plot of spike trains to a figure, at the specified position.

    parameters:
    ----------
    sts: list of SpikeTrains
        the list of spike trains for which to make a raster plot

    fig: matplotlib.pyplot.figure object
        the figure to which to add the raster plot

    (n_row, n_col, panel): tuple of integers
        the number of rows and columns in the figure, and the panel id in
        which to plot the raster display.

    ms: float, optional
        the size of the spike markers in the raster plot

    returns:
    -------
    Returns the figure fig, enriched with the raster plot at the specified
    position
    '''

    sts_ms = _rescale(sts, 'ms')
    ax = fig.add_subplot(n_row, n_col, panel)
    for i, st in enumerate(sts_ms):
        ax.plot(st.magnitude, [i + 1] * len(st), '.', ms=ms, color='k')

    t0 = min([st.t_start for st in sts_ms]).rescale('ms').magnitude
    T = max([st.t_stop for st in sts_ms]).rescale('ms').magnitude
    ax.set_xlim((t0, T))
    ax.set_ylim(0, len(sts) + 1)
    ax.set_xlabel(xlabel + ' (ms)', size=12)
    ax.set_ylabel(ylabel, size=12)
    ax.set_title(title)
    fig.add_axes(ax)

    return fig


def raster(sts, ms=4, title='', xlabel='', ylabel='', color='k'):
    '''
    Make a raster plot of a list of spike trains.

    Arguments
    ----------
    sts: list of SpikeTrains
        the list of spike trains for which to make a raster plot

    ms: float, optional
        the size of the spike markers in the raster plot

    Returns
    -------
    None
    '''

    sts_ms = _rescale(sts, 'ms')
    t0 = min([st.t_start for st in sts_ms]).rescale('ms').magnitude
    T = max([st.t_stop for st in sts_ms]).rescale('ms').magnitude

    for i, st in enumerate(sts_ms):
        plt.plot(st.magnitude, [i + 1] * len(st), '.', ms=ms, color=color)

    plt.xlim((t0, T))
    plt.ylim(0, len(sts) + 1)
    plt.xlabel(xlabel + ' (ms)', size=12)
    plt.ylabel(ylabel, size=12)
    plt.title(title)


def add_hist(fig, (n_row, n_col, panel), left, height, width=None,
             bottom=None, title='', xlabel='', ylabel=''):
    '''
    Add an axis to the specified figure, containing a bar plot

    Arguments
    ----------
    fig: matplotlib.pyplot.figure object
        the figure to which to add the raster plot
    (n_row, n_col, panel): tuple of integers
        the number of rows and columns in the figure, and the panel id in
        which to plot the raster display.
    left: array or Quantity array
        the left ends of the histogram bins
    height: array or Quantity array
        the left ends of the histogram bins
    width: float or array (also as a Quantity). Optional, default is None
        the width of each histogram bar
    bottom: float or array (also as a Quantity). Optional, default is None
        the bottom of each histogram bar
    title: str, optional. Default is ''
        title to assign to the generated axis

    Returns
    -------
        None
    '''
    if isinstance(left, pq.Quantity):
        left_dl = left.magnitude
        if left.units == pq.dimensionless:
            x_unit = ''
        else:
            x_unit = ' (%s)' % (left.units.__str__().split(' ')[-1])
    else:
        left_dl = left
        x_unit = ''

    if isinstance(height, pq.Quantity):
        height_dl = height.magnitude
        if height.units == pq.dimensionless:
            y_unit = ''
        else:
            y_unit = ' (%s)' % (height.units.__str__().split(' ')[-1])
    else:
        height_dl = height
        y_unit = ''

    width_dl = 0 if width == None else width.rescale(left.units).magnitude
    bottom_dl = 0 if bottom == None else bottom.rescale(height.units).magnitude

    ax = fig.add_subplot(n_row, n_col, panel)
    ax.bar(left=left_dl, height=height_dl, color='.5', width=width_dl,
           bottom=bottom_dl)
    ax.set_title(title)

    x0, x1 = min(left_dl), max(left_dl + width_dl)
    y0, y1 = min(height_dl) * .9, max(height_dl + bottom_dl)
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))
    ax.set_xlabel(xlabel + x_unit, size=12)
    ax.set_ylabel(ylabel + y_unit, size=12)


def hist(left, height, width=None, bottom=None, title='',
         xlabel='', ylabel=''):
    '''
    Plot Quantity objects in a bar plot

    Arguments
    ---------
    left: array or Quantity array
        the left ends of the histogram bins
    height: array or Quantity array
        the left ends of the histogram bins
    width: float or array (also as a Quantity). Optional, default is None
        the width of each histogram bar
    bottom: float or array (also as a Quantity). Optional, default is None
        the bottom of each histogram bar
    title: str, optional. Default is ''
        title to assign to the generated axis
    xlabel: str
        label of the x-axis
    ylabel: str
        label of the y-axis

    Returns
    -------
    None
    '''

    if isinstance(left, pq.Quantity):
        left_dl = left.magnitude
        if left.units == pq.dimensionless:
            x_unit = ''
        else:
            x_unit = ' (%s)' % (left.units.__str__().split(' ')[-1])
    else:
        left_dl = left
        x_unit = ''

    if isinstance(height, pq.Quantity):
        height_dl = height.magnitude
        if height.units == pq.dimensionless:
            y_unit = ''
        else:
            y_unit = ' (%s)' % (height.units.__str__().split(' ')[-1])
    else:
        height_dl = height
        y_unit = ''

    width_dl = 0 if width == None else width.rescale(left.units).magnitude
    bottom_dl = 0 if bottom == None else bottom.rescale(height.units).magnitude

    plt.bar(left=left_dl, height=height_dl, color='.5',
            width=width_dl, bottom=bottom_dl)
    plt.title(title)

    x0, x1 = min(left_dl), max(left_dl + width_dl)
    y0, y1 = min(height_dl) * .9, max(height_dl + bottom_dl)
    plt.xlim((x0, x1))
    plt.ylim((y0, y1))
    plt.xlabel(xlabel + x_unit, size=12)
    plt.ylabel(ylabel + y_unit, size=12)
    # return ax


def _expo(x, r):
    if isinstance(x, pq.Quantity) or isinstance(r, pq.Quantity):
        return r * numpy.exp(-(r * x).rescale(pq.dimensionless).magnitude)
    else:
        return r * numpy.exp(-(r * x))


def fit_to_exp(x, y):
    '''
    Fit points (x[i], y[i]) to an exponential probability density function

                f(x) = m*exp(-m*x)

    by fitting the mean parameter m.

    Arguments
    ---------
    x: array or Quantity array
        points' abscissa
    y: array or Quantity array
        points' ordinate

    Returns
    -------
    m: float or Quantity
        estimate of the distribution's mean parameter
    z: array or Quantity array
        value taken by the fit distribution in each abscissa x[i]
    '''

    x_dl = x if not isinstance(x, pq.Quantity) else x.simplified.magnitude
    y_dl = y if not isinstance(y, pq.Quantity) else y.simplified.magnitude

    r, cov = curve_fit(_expo, x_dl, y_dl)

    if isinstance(y, pq.Quantity): r = (r * pq.Hz).rescale(y.units)

    return r, _expo(x, r)


def gen_rate_profile(b, A, w, t, t0):
    """
    generate rate profile as sum of two exponential function
    (see Nawrot et al. 1999)
    """
    tau2 = float(w) / numpy.sqrt(5)
    if t < t0:
        return b
    else:
        return b + float(A) * ((1. / (tau2)) * (
        numpy.exp(-(t - t0) / (2 * tau2)) - numpy.exp(-(t - t0) / tau2)))


def isi_pdf(spiketrain, bins=10, rng=None, density=False):
    """
    Evaluate the empirical inter-spike-interval (ISI) probability density
    function (pdf) from a list of spike trains.

    Parameters:
    ----------
    spiketrain : neo.core.SpikeTrain or list of neo.core.SpikeTrain objects
        One or more spike trains for which to compute the ISI pdf

    bins : int or time Quantity. (Optional)
        If int, number of bins of the pdf histogram.
        If single-value time Quantity, width of each time bin.
        Default is 10.

    rng : Quantity array or None. (Optional)
        Range (in time unit) over which to compute the histogram:
        * if None (default) computes the histogram in the full rng
        * if pair of Quantities [r0, r1], ignores ISIs outside the rng. r0
          or r1 can also be None (min and max ISI used instead, respectively)
        Default is False.

    density : bool. (Optional)
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the pdf at the bin, normalized
        such that the *integral* over the rng is 1.
        Note that the sum of the histogram values will not be equal to 1
        unless bins of unity width are chosen.
        Default is False.

    Returns
    -------
    analogSignal: neo.core.AnalogSignal
        A neo.core.AnalogSignal containing the values of the ISI distribution.
        AnalogSignal[j] represents the ISI's pdf computed between
        `t_start + j * sampling_period and t_start + (j + 1) * sampling_period`

    """

    # Convert spiketrain to a list if it is a SpikeTrain
    if type(spiketrain) == neo.core.SpikeTrain:
        spiketrain = [spiketrain]

    # Collect all ISIs from all SpikeTrains in spiketrain
    # (as dimensionless array, meant in seconds)
    isis = []
    for st in spiketrain:
        isis = numpy.hstack([isis, numpy.diff(st.simplified.magnitude)])

    # Set histogram rng [isi_min, isi_max]
    if rng is None:
        isi_min, isi_max = min(isis), max(isis)
    elif len(rng) == 2:
        if rng[0] is None:
            isi_min = min(isis)
        else:
            try:
                isi_min = rng[0].rescale('s').magnitude
            except:
                raise ValueError('rng[0] must be a time Quantity')
        if rng[1] is None:
            isi_max = max(isis)
        else:
            try:
                isi_max = rng[1].rescale('s').magnitude
            except:
                raise ValueError('rng[1] must be a time Quantity')
    else:
        raise ValueError('Range can only be None or sequence of length two')

    # If bins is a Quantity, convert it to a dimensionless array
    # of bin edges (meant in seconds)
    if type(bins) == pq.quantity.Quantity:
        binsize = bins.simplified.magnitude
        # If bins has length 1, interpret it as binsize and create bin array.
        # Otherwise return error
        if binsize.ndim == 0:
            bins = numpy.arange(isi_min, isi_max + binsize / 2, binsize)
        else:
            raise ValueError(
                'bins can be either int or single-value Quantity. Quantity '
                'array of shape ' + bins.shape + ' given instead')

    vals, edges = numpy.histogram(isis, bins, density=density)

    # Add unit (inverse of time) to the histogram values if normalized
    if density is True:
        vals = (vals / pq.s).rescale(1. / spiketrain[0].units)
    else:
        vals = vals * pq.dimensionless

    # Rescale edges to the 1st spike train's unit and compute bin size
    edges = (edges * pq.s).rescale(spiketrain[0].units)
    w = edges[1] - edges[0]

    # Return histogram values and bins; the latter are r
    return neo.AnalogSignal(signal=vals, sampling_period=w, t_start=edges[0])


def cv(spiketrains):
    """
    Evaluate the empirical coefficient of variation (CV) of the inter-spike
    intervals (ISIs) collected from one or more spike trains.

    Given the vector v containing the observed ISIs of one spike train,
    the CV is defined as

                    CV := std(v)/mean(v).

    The CV of a list of spike trains is computed collecting the ISIs of all
    spike trains.

    The CV represents a measure of irregularity in the spiking activity. For
    For a time-stationary Poisson process, the theoretical CV=1.

    Parameters
    ---------
    spiketrains: SpikeTrain or list of SpikeTrains
        A `neo.SpikeTrain` object or a list of `neo.core.SpikeTrain` objects,
        for which to compute the CV.

    Returns
    -------
    CV : float
        The CV of all ISIs in the input SpikeTrain(s).  If no ISI can be
        calculated (less than 2 spikes in each SpikeTrains), then CV=0.
    """

    # Convert input to a list if it is a SpikeTrain object
    if isinstance(spiketrains, neo.core.SpikeTrain):
        spiketrains = [spiketrains]

    # Collect the ISIs of all trains in spiketrains, and return their CV
    isis = numpy.array([])
    for st in spiketrains:
        if len(st) > 1:
            isis = numpy.hstack([isis, numpy.diff(st.simplified.base)])

    # Compute CV of ISIs
    if len(isis) == 0:
        CV = 0.
    else:
        CV = isis.std() / isis.mean()
    return CV


def _gen_rate_profile_analysis_spike_train_book_fig17p1(delta):
    A = 11
    b = 10
    w = 0.07
    t_lst = numpy.arange(0, 1, 0.001)
    t00 = 0.1
    t01 = 0.6
    y = []
    for t in t_lst:
        if t < t00:
            y.append(b)
        elif t <= t01 + 0.008:
            y.append(gen_rate_profile(b, A * float(delta), w, t, t00) + b)
        else:
            y.append(gen_rate_profile(b, A * float(delta) / 2., w, t, t01))
    return numpy.array(y)


def _gen_data_analysis_spike_train_book_fig17p2(num_trials):
    y_low = _gen_rate_profile_analysis_spike_train_book_fig17p1(0.5)
    y = _gen_rate_profile_analysis_spike_train_book_fig17p1(1.)
    y_high = _gen_rate_profile_analysis_spike_train_book_fig17p1(1.5)

    from jelephant.core.stocmod import gamma_nonstat_rate as gnr
    import neo

    sts = []
    rand_num = (numpy.random.rand(num_trials) * 2).astype(int)
    for rnd in rand_num:
        if rnd == 0:
            y_sel = neo.AnalogSignal(y_low, sampling_period=1 * pq.ms,
                                     units=pq.Hz, t_start=0 * pq.ms)
        elif rnd == 1:
            y_sel = neo.AnalogSignal(y_high, sampling_period=1 * pq.ms,
                                     units=pq.Hz, t_start=0 * pq.ms)
        gamma_indep = gnr(y_sel, 3, N=2)
        t_start = gamma_indep[0].t_start
        t_stop = gamma_indep[0].t_stop
        coinc_rand = numpy.random.rand(2)
        for i in coinc_rand:
            gamma_indep0 = numpy.append(gamma_indep[0].magnitude,
                                        (i * pq.s).rescale(
                                            gamma_indep[0].units).magnitude) * \
                           gamma_indep[0].units
            gamma_indep1 = numpy.append(gamma_indep[1].magnitude,
                                        (i * pq.s).rescale(
                                            gamma_indep[1].units).magnitude) * \
                           gamma_indep[1].units
        sts.append(
            [neo.SpikeTrain(gamma_indep0, t_start=t_start, t_stop=t_stop)] + [
                neo.SpikeTrain(gamma_indep1, t_start=t_start, t_stop=t_stop)])
    return sts


def generate_data(lambda_b, lambda_c, T, T_h, nTrials, RateJitter=0 * pq.Hz):
    t_start_c, t_stop_c = (T - T_h) / 2., (T - T_h) / 2. + T_h
    N = 2
    sts = []
    for tr in range(nTrials):
        sp_interval1 = [stg.homogeneous_poisson_process(
            lambda_b, t_stop=t_start_c) for n in range(N)]
        sp_interval2_tmp = [stg.homogeneous_poisson_process(
            lambda_b - lambda_c, t_start_c, t_stop_c) for n in range(N)]
        coinc_sp = stg.homogeneous_poisson_process(lambda_c, t_start_c,
                                                   t_stop_c)
        sp_interval3 = [stg.homogeneous_poisson_process(
            lambda_b, t_start=t_stop_c, t_stop=T) for n in range(N)]

        s1 = numpy.sort(
            numpy.append(sp_interval2_tmp[0].rescale('ms').magnitude,
                         coinc_sp.rescale('ms').magnitude))
        s2 = numpy.sort(
            numpy.append(sp_interval2_tmp[1].rescale('ms').magnitude,
                         coinc_sp.rescale('ms').magnitude))
        sp_interval2 = [neo.SpikeTrain(
            s1 * pq.ms, t_start=sp_interval2_tmp[0].t_start,
            t_stop=sp_interval2_tmp[0].t_stop)] + [neo.SpikeTrain(
            s2 * pq.ms, t_start=sp_interval2_tmp[1].t_start,
            t_stop=sp_interval2_tmp[1].t_stop)]
        spiketrain1 = numpy.append(
            sp_interval1[0].rescale('ms').magnitude,
            sp_interval2[0].rescale('ms').magnitude)
        spiketrain1 = numpy.append(
            spiketrain1, sp_interval3[0].rescale('ms').magnitude)

        spiketrain2 = numpy.append(
            sp_interval1[1].rescale('ms').magnitude,
            sp_interval2[1].rescale('ms').magnitude)
        spiketrain2 = numpy.append(
            spiketrain2, sp_interval3[1].rescale('ms').magnitude)
        sts.append(
            [neo.SpikeTrain(
                spiketrain1 * pq.ms, t_start=0 * pq.ms, t_stop=T),
                neo.SpikeTrain(
                    spiketrain2 * pq.ms, t_start=0 * pq.ms, t_stop=T)])

    return sts


def generate_data_oscilatory(nTrials, N, T, freq_coinc, amp_coinc,
                             offset_coinc, freq_bg, amp_bg, offset_bg,
                             RateJitter=10 * pq.Hz):
    """
    generate non-stationary poisson spiketrains with injected coincidences
    """
    from jelephant.core.stocmod import poisson_nonstat as pn
    import neo

    h = 1 * pq.ms
    # modulatory coincidence rate
    tc = numpy.arange(0, T.rescale('ms').magnitude,
                      h.rescale('ms').magnitude) * pq.ms
    bbc = (2 * numpy.pi * freq_coinc * tc).simplified
    coincrate = offset_coinc + amp_coinc * numpy.sin(bbc) * offset_coinc.units
    coincrate[coincrate < 0 * coincrate.units] = 0 * coincrate.units

    # background rate
    tb = numpy.arange(0, T.rescale('ms').magnitude,
                      h.rescale('ms').magnitude) * pq.ms
    bbb = (2 * numpy.pi * freq_bg * tb).simplified
    backgroundrate = offset_bg + amp_bg * numpy.sin(bbb) * offset_bg.units
    backgroundrate[
        backgroundrate < 0 * backgroundrate.units] = 0 * backgroundrate.units

    # inhomogenious rate across trials
    rndRateJitter = (numpy.random.rand(nTrials) - 0.5) * RateJitter
    spiketrain = []
    for i in range(nTrials):
        rate_signal_bg = neo.AnalogSignal(
            (backgroundrate.rescale('Hz') + rndRateJitter[i]).magnitude,
            sampling_period=h, units=pq.Hz, t_start=0 * pq.ms)
        rate_signal_coinc = neo.AnalogSignal(coincrate.rescale('Hz').magnitude,
                                             sampling_period=h, units=pq.Hz,
                                             t_start=0 * pq.ms)
        sts_bg = pn(rate_signal_bg, N=N)
        # inserting coincidences
        sts_coinc = pn(rate_signal_coinc, N=1)
        sts_bg_coinc = []
        for j in sts_bg:
            sts_bg_coinc.append(
                neo.SpikeTrain(numpy.sort(numpy.append(j.magnitude, sts_coinc[
                    0].magnitude)) * j.units,
                               t_start=j.t_start, t_stop=j.t_stop))
        spiketrain.append(sts_bg_coinc)
    return {'st': spiketrain, 'backgroundrate': backgroundrate,
            'coincrate': coincrate}


def run_benchmark_plot(func, spiketrain, binsize, winsize, winstep,
                       pattern_hash):
    """
    Runs the benchmarks and initializes the plot benchmark function.

    """
    benchmark = list()
    # Multiprocessing
    benchmark.append(
        timeit.Timer(partial(func, spiketrain, binsize, winsize, winstep,
                             pattern_hash, parallel=True)).timeit(1))
    # Sequential
    benchmark.append(
        timeit.Timer(partial(func, spiketrain, binsize, winsize, winstep,
                             pattern_hash)).timeit(1))
    _plot_benchmark_seaborn(benchmark)


def _plot_benchmark(benchmarks):
    """
    Plot the benchmarks

    :param benchmarks: list of timed benchmarks to plot

    """
    bar_labels = ['Multiprocessing', 'Sequential']
    plt.figure(figsize=(16, 9))
    y_pos = numpy.arange(len(benchmarks))
    plt.yticks(y_pos, bar_labels, fontsize=16)
    bars = plt.barh(y_pos, benchmarks, align='center', alpha=0.7, color='g')

    for ba, be in zip(bars, benchmarks):
        plt.text(ba.get_width() + .1, ba.get_y() + ba.get_height() / 2,
                 '{0:.2%}'.format(be / benchmarks[0]),
                 ha='center', va='bottom', fontsize=11)
    plt.xlabel(
        'time in seconds ', fontsize=14)
    plt.ylabel('Methods', fontsize=14)
    plt.title('Serial vs. Multiprocessing ', fontsize=18)
    plt.ylim([-1, len(benchmarks) + 0.5])
    plt.xlim([0, max(benchmarks) * 1.1])
    plt.vlines(benchmarks[1], -1, len(benchmarks) + 0.5, linestyles='dashed')
    plt.grid()
    plt.tight_layout()
    # plt.savefig('benchmark.png')
    plt.show()


def _plot_benchmark_seaborn(benchmarks):
    try:
        import seaborn as sns
    except ImportError:
        import warnings

        warnings.warn("Plotting module seaborn is not installed!")
    else:
        # seaborn settings
        sns.set(style="darkgrid")
        sns.set_context('poster')
        sns.set_context({"figure.figsize": (17, 7)})
        sns.set_color_codes('pastel')
        bar_labels = ['Multiprocessing', 'Sequential']
        # bar = sns.barplot(data=[[benchmarks[0]], [benchmarks[1]]],
        #                   orient='h', alpha=0.7)
        y_pos = numpy.arange(len(benchmarks))
        # plt.yticks(y_pos, bar_labels, fontsize=16)

        # Create horizontal bar plot
        bar = plt.barh(y_pos, benchmarks, align='center', alpha=0.9,
                       color=sns.color_palette("Paired"))

        # Set text inside bar plot
        for ba, be in zip(bar, benchmarks):
            plt.text(ba.get_width() + .1, ba.get_y() + ba.get_height() / 2,
                     '{0:.2%}'.format(be / benchmarks[0]),
                     ha='center', va='bottom', fontsize=11)
        # Set labels etc.
        plt.xlabel(
            'time in seconds ', fontsize=14)
        # plt.ylabel('Methods', fontsize=14)
        plt.yticks([])
        plt.title('Unitary Event Analysis \n Sequential- & Multiprocessing',
                  fontsize=18)
        plt.vlines(benchmarks[0], -1, len(benchmarks) + 0.5,
                   linestyles='dashed', linewidths=0.5)
        # bar.set(yticklabels=bar_labels)
        # bar.legend(bar, bar_labels, frameon=True)

        # Display the legend in correct order
        legend = plt.legend(bar[::-1], bar_labels[::-1], title='Methods',
                            frameon=True)
        # Emphasize the legend's frame box
        frame = legend.get_frame()
        frame.set_facecolor('gray')
        # plt.savefig('benchmark.png')
        plt.tight_layout()
        plt.show()


def _plot_UE(data, Js_dict, sig_level, binsize, winsize, winstep, pattern_hash,
             N, args, add_epochs=[]):
    """
    Examples:
    ---------
    dict_args = {'events':{'SO':[100*pq.ms]},
     'save_fig': True,
     'path_filename_format':'UE1.pdf',
     'showfig':True,
     'suptitle':True,
     'figsize':(12,10),
    'unit_ids':[10, 19, 20],
    'ch_ids':[1,3,4],
    'fontsize':15,
    'linewidth':2,
    'set_xticks' :False'}
    'marker_size':8,
    """
    import matplotlib.pylab as plt

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop

    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)
    Js_sig = ue.jointJ(sig_level)
    num_tr = len(data)
    pat = ue.inv_hash(pattern_hash, N)
    events = args['events']

    # figure format
    figsize = args['figsize']
    if 'top' in args.keys():
        top = args['top']
    else:
        top = .90
    if 'bottom' in args.keys():
        bottom = args['bottom']
    else:
        bottom = .05
    if 'right' in args.keys():
        right = args['right']
    else:
        right = .95
    if 'left' in args.keys():
        left = args['left']
    else:
        left = .1

    if 'hspace' in args.keys():
        hspace = args['hspace']
    else:
        hspace = .5
    if 'wspace' in args.keys():
        wspace = args['wspace']
    else:
        wspace = .5

    if 'fontsize' in args.keys():
        fsize = args['fontsize']
    else:
        fsize = 20
    if 'unit_ids' in args.keys():
        unit_real_ids = args['unit_ids']
        if len(unit_real_ids) != N:
            raise ValueError(
                'length of unit_ids should be equal to number of neurons!')
    else:
        unit_real_ids = numpy.arange(1, N + 1, 1)
    if 'ch_ids' in args.keys():
        ch_real_ids = args['ch_ids']
        if len(ch_real_ids) != N:
            raise ValueError(
                'length of ch_ids should be equal to number of neurons!')
    else:
        ch_real_ids = []

    if 'showfig' in args.keys():
        showfig = args['showfig']
    else:
        showfig = False
    if 'linewidth' in args.keys():
        lw = args['linewidth']
    else:
        lw = 2

    if 'S_ylim' in args.keys():
        S_ylim = args['S_ylim']
    else:
        S_ylim = [-3, 3]

    if 'marker_size' in args.keys():
        ms = args['marker_size']
    else:
        ms = 8

    if add_epochs != []:
        coincrate = add_epochs['coincrate']
        backgroundrate = add_epochs['backgroundrate']
        num_row = 6
    else:
        num_row = 5
    num_col = 1
    ls = '-'
    alpha = 0.5
    plt.figure(1, figsize=figsize)
    if args['suptitle'] == True:
        plt.suptitle("Spike Pattern:" + str((pat.T)[0]), fontsize=20)
    print 'plotting UEs ...'
    plt.subplots_adjust(top=top, right=right, left=left, bottom=bottom,
                        hspace=hspace, wspace=wspace)
    ax = plt.subplot(num_row, 1, 1)
    ax.set_title('Unitary Events', fontsize=20, color='r')
    for n in range(N):
        for tr, data_tr in enumerate(data):
            plt.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) * tr + n * (
                     num_tr + 1) + 1, ',', color='k')
            sig_idx_win = numpy.where(Js_dict['Js'] >= Js_sig)[0]
            if len(sig_idx_win) > 0:
                x = numpy.unique(Js_dict['indices']['trial' + str(tr)])
                if len(x) > 0:
                    xx = []
                    for j in sig_idx_win:
                        xx = numpy.append(xx, x[numpy.where(
                            (x * binsize >= t_winpos[j]) & (
                            x * binsize < t_winpos[j] + winsize))])
                    plt.plot(
                        numpy.unique(xx) * binsize,
                        numpy.ones_like(numpy.unique(xx)) * tr + n * (
                        num_tr + 1) + 1,
                        ms=ms, marker='s', ls='', mfc='none', mec='r')
        plt.axhline((tr + 2) * (n + 1), lw=2, color='k')
    y_ticks_pos = numpy.arange(num_tr / 2 + 1, N * (num_tr + 1), num_tr + 1)
    plt.yticks(y_ticks_pos)
    plt.gca().set_yticklabels(unit_real_ids, fontsize=fsize)
    for ch_cnt, ch_id in enumerate(ch_real_ids):
        print ch_id
        plt.gca().text((max(t_winpos) + winsize).rescale('ms').magnitude,
                       y_ticks_pos[ch_cnt], 'CH-' + str(ch_id), fontsize=fsize)

    plt.ylim(0, (tr + 2) * (n + 1) + 1)
    plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    plt.xticks([])
    plt.ylabel('Unit ID', fontsize=fsize)
    for key in events.keys():
        for e_val in events[key]:
            plt.axvline(e_val, ls=ls, color='r', lw=2, alpha=alpha)
    if 'set_xticks' in args.keys() and args['set_xticks'] == False:
        plt.xticks([])
    print 'plotting Raw Coincidences ...'
    ax1 = plt.subplot(num_row, 1, 2, sharex=ax)
    ax1.set_title('Raw Coincidences', fontsize=20, color='c')
    for n in range(N):
        for tr, data_tr in enumerate(data):
            plt.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) * tr + n * (
                     num_tr + 1) + 1,
                     ',', color='k')
            plt.plot(
                numpy.unique(Js_dict['indices']['trial' + str(tr)]) * binsize,
                numpy.ones_like(numpy.unique(
                    Js_dict['indices']['trial' + str(tr)])) * tr + n * (
                num_tr + 1) + 1,
                ls='', ms=ms, marker='s', markerfacecolor='none',
                markeredgecolor='c')
        plt.axhline((tr + 2) * (n + 1), lw=2, color='k')
    plt.ylim(0, (tr + 2) * (n + 1) + 1)
    plt.yticks(numpy.arange(num_tr / 2 + 1, N * (num_tr + 1), num_tr + 1))
    plt.gca().set_yticklabels(unit_real_ids, fontsize=fsize)
    plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    plt.xticks([])
    plt.ylabel('Unit ID', fontsize=fsize)
    for key in events.keys():
        for e_val in events[key]:
            plt.axvline(e_val, ls=ls, color='r', lw=2, alpha=alpha)

    print 'plotting PSTH ...'
    plt.subplot(num_row, 1, 3, sharex=ax)
    # max_val_psth = 0.*pq.Hz
    for n in range(N):
        # data_psth = []
        # for tr,data_tr in enumerate(data):
        #    data_psth.append(data_tr[p])
        # psth = ss.peth(data_psth, w = psth_width)
        # plt.plot(psth.times,psth.base/float(num_tr)/psth_width.rescale('s'), label = 'unit '+str(unit_real_ids[p]))
        # max_val_psth = max(max_val_psth, max((psth.base/float(num_tr)/psth_width.rescale('s')).magnitude))
        plt.plot(t_winpos + winsize / 2.,
                 Js_dict['rate_avg'][:, n].rescale('Hz'),
                 label='unit ' + str(unit_real_ids[n]), lw=lw)
        # max_val_psth = max(max_val_psth, max(Js_dict['rate_avg'][:,n].rescale('Hz')))
    plt.ylabel('Rate [Hz]', fontsize=fsize)
    plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    max_val_psth = plt.gca().get_ylim()[1]
    plt.ylim(0, max_val_psth)
    plt.yticks([0, int(max_val_psth / 2), int(max_val_psth)], fontsize=fsize)
    plt.legend(bbox_to_anchor=(1.12, 1.05), fancybox=True, shadow=True)
    for key in events.keys():
        for e_val in events[key]:
            plt.axvline(e_val, ls=ls, color='r', lw=lw, alpha=alpha)

    if 'set_xticks' in args.keys() and args['set_xticks'] == False:
        plt.xticks([])
    print 'plotting emp. and exp. coincidences rate ...'
    plt.subplot(num_row, 1, 4, sharex=ax)
    plt.plot(t_winpos + winsize / 2., Js_dict['n_emp'], label='empirical',
             lw=lw, color='c')
    plt.plot(t_winpos + winsize / 2., Js_dict['n_exp'], label='expected',
             lw=lw, color='m')
    plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    plt.ylabel('# Coinc.', fontsize=fsize)
    plt.legend(bbox_to_anchor=(1.12, 1.05), fancybox=True, shadow=True)
    YTicks = plt.ylim(0,
                      int(max(max(Js_dict['n_emp']), max(Js_dict['n_exp']))))
    plt.yticks([0, YTicks[1]], fontsize=fsize)
    for key in events.keys():
        for e_val in events[key]:
            plt.axvline(e_val, ls=ls, color='r', lw=2, alpha=alpha)
    if 'set_xticks' in args.keys() and args['set_xticks'] == False:
        plt.xticks([])

    print 'plotting Surprise ...'
    plt.subplot(num_row, 1, 5, sharex=ax)
    plt.plot(t_winpos + winsize / 2., Js_dict['Js'], lw=lw, color='k')
    plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    plt.axhline(Js_sig, ls='-', color='gray')
    plt.axhline(-Js_sig, ls='-', color='gray')
    plt.gca().text(10, Js_sig + 0.2, str(int(sig_level * 100)) + '%',
                   fontsize=fsize - 2, color='gray')
    plt.xticks(t_winpos.magnitude[::len(t_winpos) / 10])
    plt.yticks([-2, 0, 2], fontsize=fsize)
    plt.ylabel('S', fontsize=fsize)
    plt.xlabel('Time [ms]', fontsize=fsize)
    plt.ylim(S_ylim)
    for key in events.keys():
        for e_val in events[key]:
            plt.axvline(e_val, ls=ls, color='r', lw=lw, alpha=alpha)
            plt.gca().text(e_val - 10 * pq.ms, 2 * S_ylim[0], key,
                           fontsize=fsize, color='r')
    if 'set_xticks' in args.keys() and args['set_xticks'] == False:
        plt.xticks([])

    if add_epochs != []:
        plt.subplot(num_row, 1, 6, sharex=ax)
        plt.plot(coincrate, lw=lw, color='c')
        plt.plot(backgroundrate, lw=lw, color='m')
        plt.xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
        plt.ylim(plt.gca().get_ylim()[0] - 2, plt.gca().get_ylim()[1] + 2)
    if args['save_fig'] == True:
        plt.savefig(args['path_filename_format'])
        if showfig == False:
            plt.cla()
            plt.close()
            # plt.xticks(t_winpos.magnitude)

    if showfig == True:
        plt.show()
