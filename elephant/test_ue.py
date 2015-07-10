import unitary_event_analysis as ue
from test_ue_import import *
import neo
reload(misc)
reload(ue)
# Generate Data
nTrials = 100        # number of trials
T = 1000*pq.ms      # trial duration
N = 2               # number of neurons

# background rate
freq_bg = 5*pq.Hz         # oscillatory
amp_bg = 4.               # modulation depth
offset_bg = 30*pq.Hz      #  background rate

# modulatory coincidence rate
freq_coinc = 2*pq.Hz      # oscillatory
amp_coinc = 1              # modulation depth
offset_coinc = 0*pq.Hz    # constant rate offset

RateJitter = 0*pq.Hz     # inhomogeneous background across trials
reload(misc)
print 'generating data ...'
data = misc.generate_data_oscilatory(nTrials, N, T,freq_coinc, amp_coinc, offset_coinc,freq_bg, amp_bg,offset_bg,RateJitter = RateJitter)
spiketrain = data['st']
num_trial, N = numpy.shape(spiketrain)[:2]
# parameters for unitary events analysis
winsize = 100*pq.ms
binsize = 1*pq.ms
winstep = 5*pq.ms
pattern_hash = [3]
significance_level = 0.01

print 'calculating UEe ...'
UE1 = ue.jointJ_window_analysis(spiketrain, binsize, winsize, winstep, pattern_hash)
UE2 = ue.jointJ_window_analysis(spiketrain, binsize, winsize, winstep, pattern_hash, parallel=True)


# plotting parameters
Js_dict = {'events':{'':[]},
     'save_fig': False,
     'path_filename_format':'./UE.pdf',
     'showfig':True,
     'suptitle':True,
     'figsize':(10,12),
    'unit_ids':range(1,N+1,1),
    'fontsize':15,
    'linewidth':2}

# misc._plot_UE(spiketrain,UE1,significance_level,binsize,winsize,winstep, pattern_hash,N,Js_dict,data)
misc._plot_UE(spiketrain,UE1,significance_level,binsize,winsize,winstep, pattern_hash,N,Js_dict,data)
