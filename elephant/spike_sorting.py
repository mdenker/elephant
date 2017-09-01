import sys
import warnings
from functools import wraps
import atexit
import inspect
import tempfile
import numpy as np
import copy
import sklearn.cluster
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
import collections
import quantities as pq
import os.path
from os import linesep as sep
from elephant.spike_train_generation import (spike_extraction,
                                             waveform_extraction)

# Recommended spike sort version is 'dev' branch.
# The SpikeSort/src folder needs to be added to the PYTHONPATH,
# but no installation of SpikeSort is required.
try:
    import spike_sort
except ImportError:
    spike_sort = None
    warnings.warn('Could not import spike_sort. No SpikeSort '
                  '(www.spikesort.org) functionality available for spike '
                  'extraction and clustering.')

# It is not necessary to install Phy. Downloading the current
# version and adding it the PYTHONPATH is sufficient/recommended.
# However, Klustakwik2 (https://github.com/kwikteam/klustakwik2)
# needs to be installed in order to perform spike clustering
try:
    import phy.io
    import phy.session
except ImportError:
    phy = None
    warnings.warn('Could not import Phy. No Phy (www.phy.readthedocs.org) '
                  'functionality )vailable for spike extraction and '
                  'clustering.')
import neo
import elephant

parameter_templates = {
    'spikesort' : {
        'extraction_dict':{'sp_win_extract': [-0.5*pq.ms, 1.5*pq.ms],
                                    'sp_win_align': [-1*pq.ms, 1*pq.ms],
                                    'filter': [500*pq.Hz, None],
                                    'filter_order': 4,
                                    'threshold': 'auto',
                                    #'remove_doubles': 0.25*pq.ms,
                                    'edge': 'falling'},

        'sorting_dict':{   'method':'k-means-plus',
                                    'num_units': 3,
                                    'ncomps': 2}
        },

    'phy' : {
        'experiment_name' : 'dummy_experiment',
        'prb_file' :      'probe',
        'spikedetekt' : {   'filter_low':500.,  # Low pass frequency (Hz)
                            'filter_high_factor':0.95 * .5,
                            'filter_butter_order':3,  # Order of Butterworth filter.

                            'filter_lfp_low':0,  # LFP filter low-pass frequency
                            'filter_lfp_high':500,  # LFP filter high-pass frequency

                            'waveform_filter': 1,
                            'waveform_scale_factor': 1,
                            'waveform_dc_offset': 0,

                            'chunk_size_seconds':10,
                            'chunk_overlap_seconds':0.15,

                            'n_excerpts':50,
                            'excerpt_size_seconds':1,
                            'threshold_strong_std_factor':4.5,
                            'threshold_weak_std_factor':2.,
                            'use_single_threshold': 1,
                            'detect_spikes':'negative',

                            'connected_component_join_size':1,

                            'extract_s_before':16,
                            'extract_s_after':16,

                            'n_features_per_channel':3,  # Number of features per channel.
                            'pca_n_waveforms_max':10000,
                            'weight_power': 2},


        'klustakwik2' : {
                            'always_split_bimodal': False,
                             'break_fraction': 0.0,
                             'consider_cluster_deletion': True,
                             'dist_thresh': 9.2103403719761836,
                             'fast_split': False,
                             'full_step_every': 1,
                             'max_iterations': 1000,
                             'max_possible_clusters': 1000,
                             'max_quick_step_candidates': 100000000,
                             'max_quick_step_candidates_fraction': 0.4,
                             'max_split_iterations': None,
                             'mua_point': 2,
                             'noise_point': 1,
                             'num_changed_threshold': 0.05,
                             'num_cpus': 1,
                             'num_starting_clusters': 500,
                             'penalty_k': 0.0,
                             'penalty_k_log_n': 1.0,
                             'points_for_cluster_mask': 100,
                             'prior_point': 1,
                             'split_every': 40,
                             'split_first': 20,
                             'subset_break_fraction': 0.01,
                             'use_mua_cluster': True,
                             'use_noise_cluster': True,
                             'log' : True,
        }
    },
    'manual' : {
        'extraction_dict' : {
                         'filter_high':400*pq.Hz,
                         'filter_low':None,
                         'threshold':-4.5,
                         'n_pre':-10, 'n_post':10,
                         'alignment':'min'
        }
    }
}


# def get_updated_parameters(software,new_parameters):
#     # get template parameters
#     if software in parameter_templates:
#         template = copy.deepcopy(parameter_templates[software])
#     else: raise ValueError('No spike sorting software with name "%s" known. '
#                            'Available softwares include %s'%(software,parameter_templates.keys()))
#
#
#     for key, value in new_parameters.iteritems():
#         # scan if this key is available in template and should be overwritten
#         overwritten = False
#         for template_section_name, template_section in template.iteritems():
#             if hasattr(template_section,'iteritems'):
#                 for template_key, template_value in template_section.iteritems():
#                     if key == template_key:
#                         template[template_section_name][template_key] = value
#                         overwritten = True
#
#         if software == 'spikesort':
#             # translation of similar keywords (for better compatibility between sorting softwares available)
#             if key == 'filter_low':
#                 template['extraction_dict']['filter'][1] = value * pq.Hz
#                 overwritten = True
#             elif key == 'filter_high':
#                 template['extraction_dict']['filter'][0] = value * pq.Hz
#                 overwritten = True
#             elif key == 'extract_s_before':
#                 template['extraction_dict']['sp_win_extract'][0] = -1*value
#                 overwritten = True
#             elif key == 'extraction_s_after':
#                 template['extraction_dict']['sp_win_extract'][1] = value
#                 overwritten = True
#
#         elif software == 'phy':
#             # translation of similar keywords (for better compatibility between sorting softwares available)
#             if key == 'filter':
#                 if value[0] != None:
#                     template['spikedetekt']['filter_high'] = value[0].rescale('Hz').magnitude
#                 else:
#                     template['spikedetekt']['filter_high'] = 0
#                 if value[1] != None:
#                     template['spikedetekt']['filter_low'] = value[1].rescale('Hz').magnitude
#                 else:
#                     template['spikedetekt']['filter_low'] = 0
#                 overwritten = True
#             elif key == 'sp_win_extract':
#                 template['spikedetekt']['extraction_s_before'] = -1*value[0]
#                 template['spikedetekt']['extraction_s_after'] = value[0]
#                 overwritten = True
#             elif key == 'filter_order':
#                 template['spikedetekt']['filter_butter_order'] = value
#                 overwritten = True
#             elif key == 'filter_order':
#                 template['spikedetekt']['filter_butter_order'] = value
#                 overwritten = True
#             elif key == 'egde':
#                 if value == 'falling':
#                     value = 'negative'
#                 elif value == 'rising':
#                     value = 'positive'
#                 template['spikedetekt']['detect_spikes'] = value
#                 overwritten = True
#             elif key == 'threshold':
#                 warnings.warn('Assuming threshold of %s as strong_threshold_std_factor'
#                               ' for spike extraction with phy.'%value)
#                 template['spikedetekt']['threshold_strong_std_factor'] = value
#                 overwritten = True
#
#         elif software == 'manual':
#             pass
#
#         else:
#             raise ValueError('Unknown spike sorting software "%s"'%software)
#
#         if overwritten == False:
#             warnings.warn('Could not assign spike extraction parameter '
#                           'value "%s" to any parameter named "%s" or similar'%(value,key))
#
#     return template



def requires(module, msg):
    # This function is copied from Phy Cluster module
    def _decorator(func):
        def _wrapper(*args, **kwargs):
            if module is None:
                raise NotImplementedError(msg)
            else:
                return func(*args, **kwargs)
        _wrapper.__doc__ = func.__doc__
        return _wrapper
    return _decorator



# @requires(spike_sort,'SpikeSort must be available to extract spikes with this method.')
# def generate_spiketrains_from_spikesort(block, waveforms=True, sort=True, extraction_dict={}, sorting_dict={}):
#     """
#     Extracting spike times and waveforms from analogsignals in neo block object. This method uses functions
#     of the spike_sort module and has not yet been tested extensively. Use at own risk.
#     :param block: (neo block object) neo block which contains analogsignals for spike extraction
#     :param extraction_dict: (dict) additional parameters used for spike extraction
#                     Automatically used parameters are:
#                         {'sp_win_extract':[-0.5*pq.ms,1.5*pq.ms],'sp_win_align':[-1*pq.ms,1*pq.ms],
#                         'remove_doubles':0.25*pq.ms, 'filter':[500*pq.Hz,None],'filter_order':4,
#                         'threshold':'auto', 'edge':'falling'}
#     :param sorting_dict: (dict) additional parameters used for spike sorting
#                     Automatically used parameters are:
#                         {'num_units':3,'ncomps':2}
#     :return: None
#     """
#
#     def ss_wrap(anasig, contact=1):
#         return {'n_contacts': contact, 'data': np.asarray(anasig).reshape((1, -1)),
#                 'FS': anasig.sampling_rate.rescale('Hz').magnitude}
#
#     def fetPCA(sp_waves, ncomps=2):
#             """
#             Calculate principal components (PCs).
#
#             Parameters
#             ----------
#             spikes : dict
#             ncomps : int, optional
#                 number of components to retain
#
#             Returns
#             -------
#             features : dict
#             """
#
#             data = sp_waves['data']
#             n_channels = data.shape[2]
#             pcas = np.zeros((n_channels*ncomps, data.shape[1]))
#
#             for ch in range(n_channels):
#                 _, _, pcas[ch::data.shape[2], ] = spike_sort.features.PCA(data[:, :, ch], ncomps)
#
#             names = ["ch.%d:PC%d" % (j+1, i+1) for i in range(ncomps) for j in range(n_channels)]
#
#             outp = {'data': pcas.T}
#             if 'is_valid' in sp_waves:
#                 outp['is_valid'] = sp_waves['is_valid']
#             outp['time'] = sp_waves['time']
#             outp['FS'] = sp_waves['FS']
#             outp['names'] = names
#
#             return outp
#
#     for seg in block.segments:
#         for anasig in seg.analogsignals:
#             # Frequency filtering for spike detection in two steps for better filter stability
#             filtered_ana = copy.deepcopy(anasig)
#             if extraction_dict['filter'][0] is not None:
#                 filtered_ana = elephant.signal_processing.butter(filtered_ana, highpass_freq=extraction_dict['filter'][0],
#                                                                  lowpass_freq=None, order=extraction_dict['filter_order'],
#                                                                  filter_function='filtfilt', fs=1.0, axis=-1)
#             if extraction_dict['filter'][1] is not None:
#                 filtered_ana = elephant.signal_processing.butter(filtered_ana, highpass_freq=None,
#                                                                  lowpass_freq=extraction_dict['filter'][1],
#                                                                  order=extraction_dict['filter_order'],
#                                                                  filter_function='filtfilt', fs=1.0, axis=-1)
#             if any(np.isnan(filtered_ana)):
#                 raise ValueError('Parameters for filtering (%s, %s) yield non valid analogsignal'
#                                  % (extraction_dict['filter'], extraction_dict['filter_order']))
#
#             spt = spike_sort.extract.detect_spikes(ss_wrap(filtered_ana), contact=0, thresh=extraction_dict['threshold'],
#                                                    edge=extraction_dict['edge'])
#             spt = spike_sort.extract.align_spikes(ss_wrap(anasig), spt,
#                                                   [i.rescale('ms').magnitude for i in extraction_dict['sp_win_align']],
#                                                   type="min", contact=0, resample=1, remove=False)
#             if 'remove_doubles' in extraction_dict:
#                 spt = spike_sort.core.extract.remove_doubles(spt, extraction_dict['remove_doubles'])
#
#             if waveforms or sort:
#                 sp_waves = spike_sort.extract.extract_spikes(ss_wrap(anasig), spt,
#                                                              [i.rescale('ms').magnitude
#                                                              for i in extraction_dict['sp_win_extract']],
#                                                              contacts=0)
#
#                 #  align waveform in y-axis
#                 for waveform in range(sp_waves['data'].shape[1]):
#                     sp_waves['data'][:, waveform, 0] -= np.mean(sp_waves['data'][:, waveform, 0])
#
#                 if sort:
#                     if len(spt['data']) > sorting_dict['ncomps']:
#                         features = fetPCA(sp_waves, ncomps=sorting_dict['ncomps'])
#                         clust_idx = spike_sort.cluster.cluster(sorting_dict['method'], features, sorting_dict['num_units'])
#                         # clustered spike times
#                         spt_clust = spike_sort.cluster.split_cells(spt, clust_idx)
#                     else:
#                         warnings.warn('Spike sorting on electrode %i not possible due to low number of spikes.'
#                                       ' Perhaps the threshold for spike extraction is too conservative?'
#                                       % anasig.annotations['electrode_id'])
#                         spt_clust = {0: spt}
#                         clust_idx = np.array([0])
#
#                     if waveforms and len(spt['data']) > sorting_dict['ncomps']:
#                         sp_waves = dict([(cl, {'data': sp_waves['data'][:, clust_idx == cl, :]})
#                                          for cl in np.unique(clust_idx)])
#                     else:
#                         sp_waves = {0: sp_waves}
#
#
#             # Create SpikeTrain objects for each unit
#             # Unit id 0 == Mua; unit_id >0 => Sua
#             spiketrains = {i+1: j for i, j in spt_clust.iteritems()} if sort else {0: spt}
#             sp_waves = {i+1: j for i, j in sp_waves.iteritems()} if waveforms and sort else {0: sp_waves}
#             for unit_i in spiketrains:
#                 sorted = sort
#                 sorting_params = sorting_dict if sort else None
#                 spiketimes = spiketrains[unit_i]['data'] * pq.ms + anasig.t_start
#
#                 st = neo.SpikeTrain(times=spiketimes,
#                                     t_start=anasig.t_start,
#                                     t_stop=anasig.t_stop,
#                                     sampling_rate=anasig.sampling_rate,
#                                     name="Channel %i, Unit %i" % (anasig.annotations['channel_index'], unit_i),
#                                     file_origin=anasig.file_origin,
#                                     unit_id=unit_i,
#                                     channel_id=anasig.annotations['channel_index'],
#                                     electrode_id=anasig.annotations['electrode_id'],
#                                     sorted=sorted,
#                                     sorting_parameters=sorting_params,
#                                     extraction_params=extraction_dict)
#
#                 if waveforms and not any([d==0 for d in sp_waves[unit_i]['data'].shape]):
#                     if sp_waves[unit_i]['data'].shape[2] != 1:
#                         raise ValueError('Unexpected shape of waveform array.')
#                     # waveform dimensions [waveform_id,???,time]
#                     st.waveforms = np.transpose(sp_waves[unit_i]['data'][:,:,0]) * anasig.units
#                     st.waveforms = st.waveforms.reshape((st.waveforms.shape[0],1,st.waveforms.shape[1]))
#                     st.left_sweep = extraction_dict['sp_win_align'][0]
#                     # st.spike_duration = extraction_dict['sp_win_align'][1] - extraction_dict['sp_win_align'][0]
#                     # st.right_sweep = extraction_dict['sp_win_align'][1]
#                 else:
#                     st.waveforms = None
#
#                 # connecting unit, spiketrain and segment
#                 rcgs = anasig.recordingchannel.recordingchannelgroups
#                 u_annotations = {'sorted': sorted,
#                                  'parameters':{ 'sorting_params': sorting_params,
#                                                 'extraction_params': extraction_dict}}
#
#                 new_unit = None
#                 for rcg in rcgs:
#                     # checking if a similar unit already exists (eg. from sorting a different segment)
#                     rcg_units = [u for u in rcg.units if u.name == st.name and u.annotations == u_annotations]
#                     if len(rcg_units) == 1:
#                         unit = rcg_units[0]
#                     elif len(rcg_units) == 0:
#                         # Generating new unit if necessary
#                         if new_unit is None:
#                             new_unit = neo.core.Unit(name=st.name, **u_annotations)
#                         unit = new_unit
#                     else:
#                         raise ValueError('%i units of name %s and annotations %s exists.'
#                                          ' This is ambiguous.' % (len(rcg_units), st.name, u_annotations))
#                     rcg.units.append(unit)
#                     unit.spiketrains.append(st)
#                 seg.spiketrains.append(st)
#

#######################################################################################################################

# @requires(phy,'Phy must be available to extract spikes with this method.')
# def generate_spiketrains_from_phy(block, waveforms=True, sort=True, parameter_dict={}):
    #
    # original_parameters = copy.deepcopy(parameter_dict)
    #
    # session_name = block.name
    # random_id = np.random.randint(0,10**10)
    # tempdir = tempfile.gettempdir()
    # prm_file_name = os.path.join(tempdir,'temp_phy_params_%s_%i.prm'%(session_name,random_id))
    # prb_file_name = os.path.join(tempdir,'temp_phy_probe_%s_%i.prb'%(session_name,random_id))
    # dummy_data_file_name = os.path.join(tempdir,'temp_phy_dummy_data_%s_%i.dat'%(session_name,random_id))
    # kwik_file_name = os.path.join(tempdir,'temp_phy_session_%s_%i.kwik'%(session_name,random_id))
    #
    # def _remove_temp_files(temp_files):
    #     for temp_file in temp_files:
    #         if os.path.isfile(temp_file):
    #             os.remove(temp_file)
    #         elif os.path.isdir(temp_file):
    #             os.rmdir(temp_file)
    #
    #
    # # removing temporary files after program finished
    # if 'keep_temp_files' in parameter_dict:
    #     if not parameter_dict['keep_temp_files']:
    #         atexit.register(_remove_temp_files,[prm_file_name,
    #                                            prb_file_name,
    #                                            dummy_data_file_name,
    #                                            kwik_file_name,
    #                                            # also remove files generated during spikesorting
    #                                            os.path.join(tempdir,kwik_file_name.replace('.kwik','.phy')),
    #                                            os.path.join(tempdir,kwik_file_name.replace('.kwik','.kwx')),
    #                                            os.path.join(tempdir,kwik_file_name.replace('.kwik','.log')),
    #                                            os.path.join(tempdir,kwik_file_name + '.bak')])
    #     parameter_dict.pop('keep_temp_files')
    #
    #
    # def _add_traces_to_params(block):
    #     # Extracting sampling rate
    #     sampling_rate = None
    #     n_channels = None
    #     for seg in block.segments:
    #         for anasig in seg.analogsignals:
    #             if sampling_rate == None:
    #                 sampling_rate = anasig.sampling_rate
    #             elif sampling_rate != anasig.sampling_rate:
    #                 raise ValueError('Analogsignals have different sampling '
    #                                  'rates. '
    #                                  'Phy can not extract spikes from signals with varying sampling rates.')
    #         if n_channels == None:
    #             n_channels = len(seg.analogsignals)
    #         elif n_channels != len(seg.analogsignals):
    #             raise ValueError('Segments contain different numbers of analogsignals. '
    #                              'Phy can not deal with different numbers of channels in one session.')
    #
    #
    #     parameter_dict['traces'] ={'raw_data_files':dummy_data_file_name,
    #                                'voltage_gain':1.0,
    #                                'sample_rate':sampling_rate.rescale('Hz').magnitude,
    #                                'n_channels':n_channels,
    #                                'dtype':'int16'}
    #
    #
    # def _generate_prm_file(phy_params):
    #     with open(prm_file_name, 'w') as f:
    #         for key0 in phy_params.iterkeys():
    #             if isinstance(phy_params[key0],dict):
    #                 f.write('%s = dict(%s'%(key0,sep))
    #                 for key, value in phy_params[key0].iteritems():
    #                     if isinstance(value,str):
    #                         value = "'%s'"%value
    #                     f.write('\t%s = %s,%s'%(key,value,sep))
    #                 f.write(')%s'%sep)
    #             else:
    #                 value = phy_params[key0]
    #                 if isinstance(value,str):
    #                     value = "'%s'"%value
    #                 f.write('%s = %s%s'%(key0,value,sep))
    #
    # def _generate_prb_file(phy_params,probe_type='linear'):
    #     if probe_type=='linear':
    #         n_channels = phy_params['traces']['n_channels']
    #         if n_channels == 1:
    #             warnings.warn('Individual spikes on multiple contacts can not be detected'
    #                           ' if spike sorting is performed on individual contacts (n_channels=1).')
    #         with open(prb_file_name, 'w') as f:
    #             f.write('channel_groups = {%s'%sep)
    #             f.write('\t0: {%s'%sep)
    #             f.write("\t\t'channels': %s,%s"%(range(n_channels),sep))
    #             f.write("\t\t'graph': %s,%s"%([[i,i+1] for i in range(n_channels-1)],sep))
    #             f.write("\t\t'geometry': %s%s"%(dict([[i,[0.0,float(i)/10]] for i in range(n_channels)]),sep))
    #             f.write('\t}%s'%sep)
    #             f.write('}')
    #     else:
    #         raise NotImplementedError('This functionality is only implemented for linear probes.')
    #
    # def _generate_dummy_data_file():
    #     with open(dummy_data_file_name, 'w') as f:
    #         f.write('dummy data')
    #
    #
    # _add_traces_to_params(block)
    # parameter_dict['prb_file'] = prb_file_name.split(os.path.sep)[-1]
    # _generate_prm_file(parameter_dict)
    # _generate_prb_file(parameter_dict,probe_type='linear')
    # _generate_dummy_data_file()
    #
    #
    # if os.path.isfile(kwik_file_name):
    #     warnings.warn('Deleting old kwik file %s to generate new spike sorting'%kwik_file_name)
    #     os.remove(kwik_file_name)
    #
    # # creating new kwik file for phy session
    # probe = phy.io.kwik.creator.load_probe(prb_file_name)
    # phy.io.create_kwik(prm_file_name,kwik_file_name,overwrite=False,probe=probe)
    #
    # # generating phy session
    # phy_session = phy.session.Session(kwik_file_name)
    #
    # def _merge_annotations(A, B):
    #     """
    #     From neo.core.baseneo, modified
    #     Merge two sets of annotations.
    #
    #     Merging follows these rules:
    #     All keys that are in A or B, but not both, are kept.
    #     For keys that are present in both:
    #         For arrays or lists: concatenate
    #         For dicts: merge recursively
    #         For strings: concatenate with ';'
    #         Otherwise: fail if the annotations are not equal
    #     """
    #     merged = {}
    #     for name in A:
    #         if name in B:
    #             try:
    #                 merged[name] = merge_annotation(A[name], B[name])
    #             except BaseException as exc:
    #                 exc.args += ('key %s' % name,)
    #                 raise
    #         else:
    #             merged[name] = A[name]
    #     for name in B:
    #         if name not in merged:
    #             merged[name] = B[name]
    #     return merged
    #
    #
    # def merge_annotation(a, b):
    #         """
    #         From neo.core.baseneo, modified
    #         First attempt at a policy for merging annotations (intended for use with
    #         parallel computations using MPI). This policy needs to be discussed
    #         further, or we could allow the user to specify a policy.
    #
    #         Current policy:
    #             For arrays or lists: concatenate
    #             For dicts: merge recursively
    #             For strings: concatenate with ';'
    #             Otherwise: fail if the annotations are not equal
    #         """
    #
    #         if isinstance(a, list):  # concatenate b to a
    #             if isinstance(b, list):
    #                 return a + b
    #             else:
    #                 return a.append(b)
    #
    #         if type(a) != type(None) and type(b) != type(None):
    #             assert type(a) == type(b), 'type(%s) %s != type(%s) %s' % (a, type(a),
    #                                                                    b, type(b))
    #         if isinstance(a, dict):
    #             return _merge_annotations(a, b)
    #         elif isinstance(a, np.ndarray):  # concatenate b to a
    #             return np.append(a, b)
    #         elif isinstance(a, basestring):
    #             if a == b:
    #                 return a
    #             else:
    #                 return a + ";" + b
    #         else:
    #             return [a,b]
    #
    # def _hstack_signals(sig1,sig2):
    #     # This function is partially copied form neo analogsignal merge()
    #     sig1 = copy.deepcopy(sig1)
    #     sig2 = copy.deepcopy(sig2)
    #     assert sig1.sampling_rate == sig2.sampling_rate
    #     assert sig1.t_start == sig2.t_start
    #     assert len(sig1) == len(sig2)
    #     sig2.units = sig1.units
    #     # stack = np.hstack(np.array,(sig1,sig2.reshape(-1,1))) #np.hstack(map(np.array, (sig1, sig2)))
    #     kwargs = {}
    #     for name in ("name", "description", "file_origin","channel_index",'sampling_rate'):
    #         attr_sig1 = getattr(sig1, name)
    #         attr_sig2 = getattr(sig2, name)
    #         # if (not(hasattr(attr_sig1,'__iter__') or hasattr(attr_sig2,'__iter__')) \
    #         #     or ((type(attr_sig1)==pq.Quantity) and type(attr_sig2)==pq.Quantity)) \
    #         #         and attr_sig1 == attr_sig2:
    #         try:
    #             if attr_sig1 == attr_sig2:
    #                 kwargs[name] = attr_sig1
    #             else:
    #                 raise ValueError()
    #         except:
    #         # else:
    #             if type(attr_sig1) != list:
    #                 attr_sig1 = [attr_sig1]
    #             if type(attr_sig2) != list:
    #                 attr_sig2 = [attr_sig2]
    #             attr_sig1 = attr_sig1 + attr_sig2
    #             setattr(sig1,name,attr_sig1)
    #             setattr(sig2,name,attr_sig1)
    #
    #     if 'channel_index' in sig1.annotations:
    #         sig1.annotations.pop('channel_index')
    #     if 'sampling_rate' in sig1.annotations:
    #         sig1.annotations.pop('sampling_rate')
    #     if 't_start' in sig1.annotations:
    #         sig1.annotations.pop('t_start')
    #
    #     merged_annotations = merge_annotation(sig1.annotations,
    #                                            sig2.annotations)
    #
    #     sig2 = sig2.reshape((-1,1))
    #
    #     stacked = np.hstack((sig1,sig2))
    #     stacked.__dict__ = sig1.__dict__.copy()
    #     stacked.annotations = merged_annotations
    #
    #     return stacked
    #
    # def _kwik_spikes_to_neo_block(seg,traces,waveforms, sort):
    #     #read results from kwik file(s) or phy_session
    #
    #     kwikfile = phy.io.h5.File(kwik_file_name)
    #     kwikfile.open()
    #     time_samples = kwikfile.read('/channel_groups/0/spikes/time_samples')
    #     time_fractional = kwikfile.read('channel_groups/0/spikes/time_fractional')
    #     cluster_ids = np.asarray(kwikfile.read('/channel_groups/0/spikes/clusters/main'))
    #     spike_channel_masks = np.asarray([phy_session.model.masks[i] for i in range(len(time_samples))])
    #
    #     phy_session.store.is_consistent()
    #
    #     if waveforms:
    #         try:
    #             kwxfile = phy.io.h5.File(kwik_file_name.replace('.kwik','.kwx'))
    #             kwxfile.open()
    #             if kwxfile.exists('/channel_groups/0/waveforms_raw'):
    #                 waveforms_raw = kwxfile.read('/channel_groups/0/waveforms_raw')
    #             else:
    #                 waveforms_raw = phy_session.model.waveforms[range(phy_session.n_spikes)]
    #
    #             # if kwxfile.exists('/channel_groups/0/waveforms_filtered'):
    #             #     waveforms_filtered = kwxfile.read('/channel_groups/0/waveforms_filtered')
    #             # else:
    #             #     waveforms_filtered = phy_session.store.waveforms(0,'filtered')
    #             if kwxfile.exists('/channel_groups/0/features_masks'):
    #                 features_masks = kwxfile.read('/channel_groups/0/features_masks')
    #             else:
    #                 features = phy_session.store.features(0)
    #                 features_masks = phy_session.model.features_masks[range(phy_session.n_spikes)]
    #         except KeyError:
    #             warnings.warn('Could not extract wavefroms from kwik file or phy_session due to inconsistencies.')
    #             waveforms = False
    #
    #     spiketimes = (np.asarray(time_samples) / traces.sampling_rate) + t_start
    #
    #     for i,unit_id in enumerate(np.unique(cluster_ids)):
    #         unit_mask = cluster_ids == unit_id
    #
    #         for channel_id in range(phy_session.model.n_channels):
    #             print 'unit %i, channel %i'%(unit_id,channel_id)
    #             channel_id = int(channel_id)
    #             unit_channel_mask = np.where(np.logical_and(spike_channel_masks[:,channel_id] > 0, unit_mask) == True)
    #             unit_spikes = spiketimes[unit_channel_mask]
    #
    #             if len(unit_spikes) < 1:
    #                 continue
    #
    #             original_anasig = seg.analogsignals[channel_id]
    #
    #             # generating spiketrains
    #             st = neo.SpikeTrain(times=unit_spikes,
    #                                 t_start=traces.t_start,
    #                                 t_stop=traces.t_stop,
    #                                 sampling_rate=traces.sampling_rate,
    #                                 name="Channel %i, Unit %i" % (original_anasig.channel_index, unit_id),
    #                                 file_origin=traces.file_origin,
    #                                 unit_id=unit_id,
    #                                 channel_id=anasig.annotations['channel_index'],
    #                                 electrode_id=anasig.annotations['electrode_id'],
    #                                 sorted=sort,
    #                                 sorting_parameters=parameter_dict['klustakwik2'],
    #                                 extraction_params=parameter_dict['spikedetekt'],
    #                                 prb_file=parameter_dict['prb_file'],
    #                                 # channel_affiliations=spike_channel_masks[unit_channel_mask,channel_id]
    #                                 )
    #
    #             if waveforms:
    #                 # waveform dimensions [waveform_id,??,time]
    #                 st.waveforms = waveforms_raw[unit_channel_mask][:,:,channel_id] * original_anasig.units
    #                 st.waveforms = st.waveforms.reshape((st.waveforms.shape[0],1,st.waveforms.shape[1]))
    #                 st.left_sweep = -1 * parameter_dict['spikedetekt']['extract_s_before'] / anasig.sampling_rate
    #                 # st.spike_duration = (parameter_dict['spikedetekt']['extract_s_after']) / anasig.sampling_rate  -st.left_sweep
    #                 # st.right_sweep = parameter_dict['spikedetekt']['extract_s_after'] / anasig.sampling_rate
    #             else:
    #                 st.waveforms = None
    #
    #             # connecting unit, spiketrain and segment
    #             rcgs = anasig.recordingchannel.recordingchannelgroups
    #             u_annotations = {'sorted': sort,
    #                              'parameters': original_parameters}
    #
    #             new_unit = None
    #             for rcg in rcgs:
    #                 # checking if a similar unit already exists (eg. from sorting a different segment)
    #                 rcg_units = [u for u in rcg.units if u.name == st.name and u.annotations == u_annotations]
    #                 if len(rcg_units) == 1:
    #                     unit = rcg_units[0]
    #                 elif len(rcg_units) == 0:
    #                     # Generating new unit if necessary
    #                     if new_unit is None:
    #                         new_unit = neo.core.Unit(name=st.name, **u_annotations)
    #                     unit = new_unit
    #                 else:
    #                     raise ValueError('%i units of name %s and annotations %s exists.'
    #                                      ' This is ambiguous.' % (len(rcg_units), st.name, u_annotations))
    #                 rcg.units.append(unit)
    #                 unit.spiketrains.append(st)
    #             seg.spiketrains.append(st)
    #
    #
    #
    #
    #
    # # get maximal time period, where all analogsignals are present and collect signals in analogsignal
    # for seg in block.segments:
    #     traces = None
    #     for anasig in seg.analogsignals:
    #         if type(traces) == type(None):
    #             traces = anasig.reshape((-1,1))
    #
    #         else:
    #             # adjusting length of signals
    #             if anasig.t_start<traces.t_start:
    #                 anasig.timeslice(traces.t_start,None)
    #             elif anasig.t_start>traces.t_start:
    #                 traces.time_slice(anasig.t_start,None)
    #
    #             if anasig.t_stop>traces.t_stop:
    #                 anasig.timeslice(None,traces.t_stop)
    #             elif anasig.t_stop<traces.t_stop:
    #                 traces.time_slice(None,traces.t_stop)
    #
    #             # merging signals into one analogsignal
    #             traces = _hstack_signals(traces,anasig)
    #
    #     t_start, t_stop = traces.t_start, traces.t_stop
    #
    #     #detecting spikes using blank kwik file and lfp traces from neo block
    #     print 'Starting spike detection and extraction on %i (%i) anasigs.'%(len(seg.analogsignals), traces.shape[1])
    #     phy_session.detect(np.asarray(traces))
    #     phy_session.save()
    #
    #     if sort:
    #         print 'Starting spike clustering.'
    #         phy_session.cluster()
    #         phy_session.save()
    #
    #
    #     _kwik_spikes_to_neo_block(seg,traces,waveforms,sort)


# def generate_spiketrains_unsorted(block, waveforms=False, extraction_dict=None):
#
#     filter_high = extraction_dict['filter_high']
#     filter_low = extraction_dict['filter_low']
#     threshold = extraction_dict['threshold']
#     if waveforms:
#         n_pre, n_post = [extraction_dict[key] for key in ['n_pre','n_post']]
#         alignment = extraction_dict['alignment']
#
#     def get_threshold_crossing_ids(sig,threshold):
#         # normalize threshold to be positive
#         if threshold < 0:
#             threshold = -threshold
#             sig = sig*(-1)
#         # calculate ids at which signal crosses threshold value
#         crossings = (threshold - sig).magnitude
#         crossings *= (crossings>0)
#         mask_bool = crossings.astype(bool).astype(int)
#         crossing_ids = np.where(np.diff(mask_bool, axis=0)==-1)[0]
#         return crossing_ids
#
#     def check_threshold(threshold,signal):
#         if isinstance(threshold,pq.quantity.Quantity):
#             thres = threshold
#         elif isinstance(threshold,(int,float)):
#             warnings.warn('Assuming threshold is given in standard deviations '
#                           'of the signal amplitude.')
#             thres = threshold*np.std(sig)
#         else:
#             raise ValueError('Unknown threshold unit "%s"'%threshold)
#         return thres
#
#     for seg in block.segments:
#         for anasig_id, anasig in enumerate(seg.analogsignals):
#             sig = elephant.signal_processing.butter(anasig, filter_high,
#                                                     filter_low)
#
#             thres = check_threshold(threshold,sig)
#
#             ids = get_threshold_crossing_ids(sig, thres)
#
#             # remove border ids
#             ids = ids[np.logical_and(ids > -n_pre, ids < (len(sig)-n_post))]
#
#             st = neo.SpikeTrain(anasig.times[ids], unit_id=None, sorted=False,
#                 name="Channel %s, Unit %i" % (anasig.get_channel_index(), -1),
#                 t_start=anasig.t_start,t_stop=anasig.t_stop,
#                 sampling_rate=anasig.sampling_rate,
#                 electrode_id=anasig.annotations['electrode_id'],
#                 channel_index=anasig.annotations['channel_index'],
#                 left_sweep=n_pre*(-1),
#                 n_pre=n_pre,
#                 n_post=n_post)
#             seg.spiketrains.append(st)
#
#             print len(st)
#
#
#
#             if waveforms and len(ids):
#                 wfs = np.zeros((n_post - n_pre,len(ids))) * anasig.units
#                 for i, id in enumerate(ids):
#                     try:
#                         wfs[:,i] = anasig[id+n_pre:id+n_post]
#                     except:
#                         pass
#                 if alignment=='min':
#                     minima = np.min(wfs,axis=0)
#                     wfs = wfs - minima[np.newaxis,:]
#                 else:
#                     raise ValueError('Unknown aligmnment "%s"'%alignment)
#                 st.waveforms = wfs.T
#
#
#             # connecting unit and segment
#             current_chidx = anasig.channel_index
#             u_annotations = {'sorted': False,
#                              'parameters': {'extraction_dict':extraction_dict}}
#
#             channel_indexes = [anasig.channel_index]
#             new_unit = None
#             for chidx in channel_indexes:
#                 # checking if a similar unit already exists (eg. from sorting a different segment)
#                 chidx_units = [u for u in chidx.units if u.name == st.name and
#                              u.annotations == u_annotations]
#                 if len(chidx_units) == 1:
#                     unit = chidx_units[0]
#                 elif len(chidx_units) == 0:
#                     # Generating new unit if necessary
#                     if new_unit is None:
#                         new_unit = neo.core.Unit(name=st.name, **u_annotations)
#                     unit = new_unit
#                 else:
#                     raise ValueError('%i units of name %s and annotations %s exists.'
#                                      ' This is ambiguous.' % (len(chidx_units), st.name, u_annotations))
#                 chidx.units.append(unit)
#                 unit.spiketrains.append(st)
#
#
#






########################################################################################################################
# def generate_spiketrains(block, software, waveforms=True, sort=True, parameter_dict={}):
    # def SpikeTrain(func, *part_args):
    #     def wrapper(*extra_args):
    #         args = list(part_args)
    #         args.extend(extra_args)
    #         return neo.SpikeTrain(*args)
    #     return wrapper
    #
    # if software == 'phy':
    #     phy_parameters = get_updated_parameters(software=software,new_parameters=parameter_dict)
    #     generate_spiketrains_from_phy(block, waveforms=waveforms, sort=sort,parameter_dict=phy_parameters)
    #
    # elif software == 'spikesort':
    #     spikesort_parameters = get_updated_parameters(software=software,new_parameters=parameter_dict)
    #     extraction_dict = spikesort_parameters['extraction_dict']
    #     sorting_dict = spikesort_parameters['sorting_dict']
    #     generate_spiketrains_from_spikesort(block, waveforms=waveforms, sort=sort, extraction_dict=extraction_dict, sorting_dict=sorting_dict)

    # elif software == 'manual':
    #     manual_parameters = get_updated_parameters(software=software,
    #                                                new_parameters=parameter_dict)
    #     extraction_dict = manual_parameters['extraction_dict']
    #     generate_spiketrains_unsorted(block, waveforms=waveforms,
    #                                   extraction_dict=extraction_dict)


####################### Supplementory Functions ###########################
# from Brian at http://stackoverflow.com/questions/196960/can-you-list-the-keyword-arguments-a-python-function-receives
def getRequiredArgs(func):
    args, varargs, varkw, defaults = inspect.getargspec(func)
    if defaults:
        args = args[:-len(defaults)]
    return args

def getKwArgs(func):
    args, varargs, varkw, defaults = inspect.getargspec(func)
    if defaults:
        args = args[-len(defaults):]
    return dict(zip(args, defaults))

def getArgs(func):
    return inspect.getargspec(func)[0]

def missingArgs(func, argdict):
    return set(getRequiredArgs(func)).difference(argdict)

def invalidArgs(func, argdict):
    args, varargs, varkw, defaults = inspect.getargspec(func)
    if varkw: return set()  # All accepted
    return set(argdict) - set(args)

def isCallableWithArgs(func, argdict):
    return not missingArgs(func, argdict) and not invalidArgs(func, argdict)


class SpikeSorter(object):

    accepted_arguments = dict()

    def __init__(self, **parameter_dict):
        self.parameter_dict = parameter_dict
        pass

    @staticmethod
    def get_sorting_hash(parameter_dict):
        def _make_hash(o):

            """
            Makes a hash from a dictionary, list, tuple or set to any level, that contains
            only other hashable types (including any lists, tuples, sets, and
            dictionaries).
            """

            if isinstance(o, (set,tuple,list)):
                return tuple([_make_hash(e) for e in o])

            elif isinstance(o, pq.Quantity):
                return hash(str(o))

            elif isinstance(o, dict):
                new_o = copy.deepcopy(o)
                for k, v in new_o.items():
                    new_o[k] = _make_hash(v)
                ordered_obj = collections.OrderedDict(sorted(new_o.items()))
                return hash(str(ordered_obj))
            # set hash of None explicitly since differs between python
            # instances
            elif o is None:
                return -1
            else:
                return hash(o)

        return _make_hash(parameter_dict)

    # def SortedSpikeTrain(self, *args, **kwargs):
    #     kwargs['sorting_hash'] = self.sorting_hash
    #     return neo.SpikeTrain(*args,**kwargs)

    @property
    def sorting_hash(self):
        return self.get_sorting_hash(self.parameter_dict)

    def sort_analogsignal(self, anasig):
        raise NotImplementedError

    def sort_block(self, block):
        raise NotImplementedError

    def sort_segment(self, segment):
        raise NotImplementedError

    def _get_sorting_channel(self, block):
        chidx = None
        for chidx_i in block.channel_indexes:
            if ('sorting_hash' in chidx_i.annotations
                and chidx_i.annotations['sorting_hash'] == self.sorting_hash):
                chidx = chidx_i
                break

        if chidx is None:
            chidx = neo.ChannelIndex([0], #TODO: what index should be used here?
                                     name='spike sorting channel_index',
                                     sorting_hash=self.sorting_hash)
            block.channel_indexes.append(chidx)
            chidx.block = block
            block.create_relationship()

        return chidx

    def _get_channel_id(self, neo_obj):
        channel_id = None
        if hasattr(neo_obj, 'annotations'):
            if 'channel_id' in neo_obj.annotations:
                channel_id = neo_obj.annotations['channel_id']
            elif 'channel_index' in neo_obj.annotations:
                channel_id = neo_obj.annotations['channel_index']
                neo_obj.annotate(channel_id=channel_id)
            elif (hasattr(neo_obj, 'unit')
                  and self._get_channel_id(neo_obj.unit) is not None):
                channel_id = self._get_channel_id(neo_obj.unit)
        else:
            warnings.warn('Can not determine channel id of Neo object "{}".'
                          ' Using channel_id "None"'
                          ''.format(neo_obj))
        return channel_id

    def _add_unit(self, block, channel_id, unit_id):
        sorting_chidx = self._get_sorting_channel(block)
        channel_indexes = [sorting_chidx]
        # TODO: unit should also be added to channel index holding analogsignal
        # if anasig.channel_index is not None:#TODO: How should this work
        #     # if multiple chidx are attached to an analogsignal?
        #     channel_indexes.append(anasig.channel_index)
        for chidx in channel_indexes:
            # if chidx is None:
            #     continue
            for unit in chidx.units:
                if ('unit_id' in unit.annotations
                    and unit.annotations['unit_id'] == unit_id
                    and 'channel_id' in unit.annotations
                    and unit.annotations['channel_id'] == channel_id):
                    raise ValueError('Unit with id {} on channel {} exists '
                                     'already for sorting {}'
                                     ''.format(unit_id, channel_id,
                                               self.sorting_hash))

        unit = neo.Unit(channel_id=channel_id, unit_id=unit_id,
                        sorting_hash=self.sorting_hash)
        unit.channel_index = sorting_chidx
        for chidx in channel_indexes:
            chidx.units.append(unit)
        return unit

    def _add_to_segment(self, anasig, spiketrains):
        if not isinstance(anasig, list):
            anasig.segment.spiketrains.extend(spiketrains)
        else:
            anasig.segment.spiketrains.append(spiketrains)

    def sort_analogsignal(self, anasig):
        raise ValueError('{} is not implemented/applicable for {}.'.format(
                         sys._getframe().f_code.co_name, self.__class__))

    def sort_segment(self, segment):
        raise ValueError('{} is not implemented/applicable for {}.'.format(
                         sys._getframe().f_code.co_name, self.__class__))

    def sort_block(self, block):
        raise ValueError('{} is not implemented/applicable for {}.'.format(
                         sys._getframe().f_code.co_name, self.__class__))

    def sort_spiketrain(self, spiketrain):
        raise ValueError('{} is not implemented/applicable for {}.'.format(
                         sys._getframe().f_code.co_name, self.__class__))

    def AnnotatedSpiketrain(self, *args, **kwargs):
        st = neo.SpikeTrain(*args, **kwargs)
        self._annotate_with_hash(st)
        return st

    def _annotate_with_hash(self, neo_object):
        if not isinstance(neo_object, list):
            neo_object = [neo_object]
        for neo_obj in neo_object:
            if 'sorting_hash' not in neo_obj.annotations:
                neo_obj.annotate(sorting_hash=self.sorting_hash)
                neo_obj.annotate(sorter=self.__class__.__name__)
            else:
                # listify if multiple sorters (eg extractor + sorter) have
                # been used
                neo_obj.annotations['sorting_hash'] = \
                    [neo_obj.annotations['sorting_hash'], self.sorting_hash]
                neo_obj.annotations['sorter'] = \
                    [neo_obj.annotations['sorter'], self.__class__.__name__]


    # def annotate_hash(self,neo_class):
    #     @wraps(neo_class)
    #     def wrapper(*args, **kwargs):
    #         kwargs['sorting_hash'] = self.sorting_hash
    #         return neo_class(*args, **kwargs)
    #     return wrapper


class SpikeExtractor(SpikeSorter):

    # required_arguments = getRequiredArgs(spike_extraction)
    arguments = getKwArgs(spike_extraction)
    if 'signal' in arguments:
        arguments.pop('signal')
    accepted_arguments = arguments

    def __init__(self, **parameter_dict):
        super(self.__class__, self).__init__(**parameter_dict)
        pass

    def sort_analogsignal(self, anasig):
        parameters = copy.deepcopy(self.parameter_dict)
        sig = anasig
        if ('filter_high' in parameters
            and 'filter_low' in parameters):
            low = parameters.pop('filter_low')
            high = parameters.pop('filter_high')
            if (low is not None) or (high is not None):
                sig = elephant.signal_processing.butter(anasig, high, low)
                sig.segment = anasig.segment
                sig.channel_index = anasig.channel_index
        spiketrains = spike_extraction(sig, **parameters)
        self._annotate_with_hash(spiketrains)

        self._add_to_segment(anasig, spiketrains)

        channel_id = self._get_channel_id(anasig)
        unit = self._add_unit(anasig.segment.block, channel_id, unit_id=0)
        unit.annotate(sorted=False,
                      unit_type='mua')
        # the sorting_hash could also be annotated using the mock module in
        # python 3 to adjust the python-neo spiketrain class

        unit.spiketrains = spiketrains

        unit.create_relationship()
        return spiketrains

    def sort_segment(self, segment):
        for anasig in segment.analogsignals:
            self.sort_analogsignal(anasig)

    def sort_block(self, block):
        for seg in block.segments:
            self.sort_segment(seg)


class SpikeTrainGenerator(SpikeSorter):
    arguments = {'filter_high': None, 'filter_low': None}
    arguments.update(getKwArgs(waveform_extraction))
    for kw in ['signal', 'spiketrains']:
        if kw in arguments:
            arguments.pop(kw)
    accepted_arguments = arguments


    def __init__(self, **parameter_dict):
        super(self.__class__, self).__init__(**parameter_dict)
        pass

    def generate_spiketrains(self, times, anasig=None,
                             filter_high=None, filter_low=None):
        parameters = copy.deepcopy(self.parameter_dict)
        sig = anasig
        if filter_high and filter_low:
            low = filter_low
            high = filter_high
            if (low is not None) or (high is not None):
                sig = elephant.signal_processing.butter(anasig, high, low)
                sig.segment = anasig.segment
                sig.channel_index = anasig.channel_index

        # match dimensions of times to analogsignal
        if not isinstance(times, pq.Quantity) and len(times)==1:
            times_list = [times]*anasig.shape[-1]
        else:
            times_list = times

        sts = []
        for asig_id in range(len(anasig.shape[-1])):
            t = times_list[asig_id]
            if len(t):
                t_start, t_stop = t[0], t[-1]
            spiketrain = neo.SpikeTrain(t, t_start=t_start, t_stop=t_stop,
                                        sampling_rate=anasig.sampling_rate)
            sts.append(spiketrain)

            waveform_extraction(anasig, spiketrain,
                                extr_interval=parameters['extr_interval'])
            self._annotate_with_hash(spiketrain)

            self._add_to_segment(anasig, spiketrain)

            channel_id = self._get_channel_id(anasig)
            unit = self._add_unit(anasig.segment.block, channel_id, unit_id=-1)
            unit.annotate(sorted=False,
                          unit_type='mua',
                          signal_type='stimulation_times')
            # the sorting_hash could also be annotated using the mock module in
            # python 3 to adjust the python-neo spiketrain class

            unit.spiketrains.append(spiketrain)
            unit.create_relationship()
        return sts


@requires(sklearn, 'PCA spike sorting requires sklearn')
class KMeansSorter(SpikeSorter):
    accepted_arguments = getKwArgs(sklearn.cluster.KMeans.__init__)

    def __init__(self, **parameter_dict):
        super(self.__class__, self).__init__(**parameter_dict)

    def sort_spiketrain(self, spiketrain):
        # put id axis first to calculate KMeans across this axis
        # waveforms = np.swapaxes(spiketrain.waveforms, 0, -1)
        waveforms = spiketrain.waveforms
        if len(waveforms.shape) == 3:
            assert waveforms.shape[1] in [0, 1]
            waveforms = waveforms.reshape((waveforms.shape[0],
                                           waveforms.shape[2]))

        if waveforms.shape[0] == 0:
            print('No waveforms present to be sorted.')
            channel_id = self._get_channel_id(spiketrain)
            unit = self._add_unit(spiketrain.segment.block, channel_id,
                                  unit_id=None)
            unit.annotate(sorted=True, unit_type='sua')

            unit.spiketrains.append(spiketrain)
            spiketrain.unit = unit
            return None

        # ica = FastICA(n_components=3)
        # S_ = ica.fit_transform(waveforms.T)
        # A_ = ica.mixing_

        # pca = PCA(n_components=3).fit(waveforms)
        m = sklearn.cluster.KMeans(**self.parameter_dict).fit(waveforms)
        # spiketrain.annotate(cluster_id=m.labels_)
        labels = m.labels_

        channel_id = self._get_channel_id(spiketrain)

        unique, counts = np.unique(labels, return_counts=True)
        cluster_ids = dict(zip(unique, counts))
        # sorting by size of clusters
        cluster_ids = sorted(cluster_ids, key=cluster_ids.get, reverse=True)

        for unit_id, cluster_id in enumerate(cluster_ids):
            mask = np.where(labels == cluster_id)[0]
            new_times = spiketrain.times[mask]
            new_waveforms = spiketrain.waveforms[mask,:,:]
            sorted_st = spiketrain.duplicate_with_new_data(new_times,
                                                           spiketrain.t_start,
                                                           spiketrain.t_stop,
                                                           new_waveforms)
            self._annotate_with_hash(sorted_st)

            sorted_st.segment = spiketrain.segment
            spiketrain.segment.spiketrains.append(sorted_st)

            unit = self._add_unit(spiketrain.segment.block, channel_id,
                                  unit_id=unit_id)
            unit.annotate(sorted=True, unit_type='sua')

            unit.spiketrains.append(sorted_st)
            sorted_st.unit = unit

        spiketrain.segment.block.create_relationship()

    def sort_segment(self, segment):
        for st in copy.copy(segment.spiketrains):
            self.sort_spiketrain(st)

    def sort_block(self, block):
        for seg in copy.copy(block.segments):
            self.sort_segment(seg)


msg = 'SpikeSort must be available to extract spikes using the SpikeSortSorter.'
@requires(spike_sort, msg)
class SpikeSortSorter(SpikeSorter):
    accepted_arguments = {
        'extraction_dict': {'sp_win_extract': [-0.5 * pq.ms, 1.5 * pq.ms],
                            'sp_win_align': [-1 * pq.ms, 1 * pq.ms],
                            'filter': [500 * pq.Hz, None],
                            'filter_order': 4,
                            'threshold': 'auto',
                            # 'remove_doubles': 0.25*pq.ms,
                            'edge': 'falling'},

        'sorting_dict': {'method': 'k-means-plus',
                         'num_units': 3,
                         'ncomps': 2}}

    def __init__(self, **parameter_dict):
        super(self.__class__, self).__init__(**parameter_dict)

    def _ss_wrap(anasig, contact=1):
        return {'n_contacts': contact, 'data': np.asarray(anasig).reshape((1, -1)),
                'FS': anasig.sampling_rate.rescale('Hz').magnitude}

    def _fet_pca(sp_waves, ncomps=2):
            """
            Calculate principal components (PCs).

            Parameters
            ----------
            spikes : dict
            ncomps : int, optional
                number of components to retain

            Returns
            -------
            features : dict
            """

            data = sp_waves['data']
            n_channels = data.shape[2]
            pcas = np.zeros((n_channels*ncomps, data.shape[1]))

            for ch in range(n_channels):
                _, _, pcas[ch::data.shape[2], ] = spike_sort.features.PCA(data[:, :, ch], ncomps)

            names = ["ch.%d:PC%d" % (j+1, i+1) for i in range(ncomps) for j in range(n_channels)]

            outp = {'data': pcas.T}
            if 'is_valid' in sp_waves:
                outp['is_valid'] = sp_waves['is_valid']
            outp['time'] = sp_waves['time']
            outp['FS'] = sp_waves['FS']
            outp['names'] = names

            return outp

    def sort_analogsignal(self, anasig, waveforms, sort):
        extraction_dict = self.parameter_dict['extraction_dict']
        sorting_dict = self.parameter_dict['sorting_dict']
        # Frequency filtering for spike detection in two steps for better filter stability
        filtered_ana = copy.deepcopy(anasig)
        if extraction_dict['filter'][0] is not None:
            filtered_ana = elephant.signal_processing.butter(filtered_ana, highpass_freq=extraction_dict['filter'][0],
                                                             lowpass_freq=None, order=extraction_dict['filter_order'],
                                                             filter_function='filtfilt', fs=1.0, axis=-1)
        if extraction_dict['filter'][1] is not None:
            filtered_ana = elephant.signal_processing.butter(filtered_ana, highpass_freq=None,
                                                             lowpass_freq=extraction_dict['filter'][1],
                                                             order=extraction_dict['filter_order'],
                                                             filter_function='filtfilt', fs=1.0, axis=-1)
        if any(np.isnan(filtered_ana)):
            raise ValueError('Parameters for filtering (%s, %s) yield non valid analogsignal'
                             % (extraction_dict['filter'], extraction_dict['filter_order']))

        spt = spike_sort.extract.detect_spikes(self._ss_wrap(filtered_ana), contact=0, thresh=extraction_dict['threshold'],
                                               edge=extraction_dict['edge'])
        spt = spike_sort.extract.align_spikes(self._ss_wrap(anasig), spt,
                                              [i.rescale('ms').magnitude for i in extraction_dict['sp_win_align']],
                                              type="min", contact=0, resample=1, remove=False)
        if 'remove_doubles' in extraction_dict:
            spt = spike_sort.core.extract.remove_doubles(spt, extraction_dict['remove_doubles'])

        if waveforms or sort:
            sp_waves = spike_sort.extract.extract_spikes(self._ss_wrap(anasig), spt,
                                                         [i.rescale('ms').magnitude
                                                         for i in extraction_dict['sp_win_extract']],
                                                         contacts=0)

            #  align waveform in y-axis
            for waveform in range(sp_waves['data'].shape[1]):
                sp_waves['data'][:, waveform, 0] -= np.mean(sp_waves['data'][:, waveform, 0])

            if sort:
                if len(spt['data']) > sorting_dict['ncomps']:
                    features = self._fet_pca(sp_waves, ncomps=sorting_dict['ncomps'])
                    clust_idx = spike_sort.cluster.cluster(sorting_dict['method'], features, sorting_dict['num_units'])
                    # clustered spike times
                    spt_clust = spike_sort.cluster.split_cells(spt, clust_idx)
                else:
                    warnings.warn('Spike sorting on electrode %i not possible due to low number of spikes.'
                                  ' Perhaps the threshold for spike extraction is too conservative?'
                                  % anasig.annotations['electrode_id'])
                    spt_clust = {0: spt}
                    clust_idx = np.array([0])

                if waveforms and len(spt['data']) > sorting_dict['ncomps']:
                    sp_waves = dict([(cl, {'data': sp_waves['data'][:, clust_idx == cl, :]})
                                     for cl in np.unique(clust_idx)])
                else:
                    sp_waves = {0: sp_waves}


        # Create SpikeTrain objects for each unit
        # Unit id 0 == Mua; unit_id >0 => Sua
        spiketrains = {i+1: j for i, j in spt_clust.iteritems()} if sort else {0: spt}
        sp_waves = {i+1: j for i, j in sp_waves.iteritems()} if waveforms and sort else {0: sp_waves}
        for unit_i in spiketrains:
            sorted = sort
            sorting_params = sorting_dict if sort else None
            spiketimes = spiketrains[unit_i]['data'] * pq.ms + anasig.t_start

            st = neo.SpikeTrain(times=spiketimes,
                                t_start=anasig.t_start,
                                t_stop=anasig.t_stop,
                                sampling_rate=anasig.sampling_rate,
                                name="Channel %i, Unit %i" % (anasig.annotations['channel_index'], unit_i),
                                file_origin=anasig.file_origin,
                                unit_id=unit_i,
                                channel_id=anasig.annotations['channel_index'],
                                electrode_id=anasig.annotations['electrode_id'],
                                sorted=sorted,
                                sorting_parameters=sorting_params,
                                extraction_params=extraction_dict)

            if waveforms and not any([d==0 for d in sp_waves[unit_i]['data'].shape]):
                if sp_waves[unit_i]['data'].shape[2] != 1:
                    raise ValueError('Unexpected shape of waveform array.')
                # waveform dimensions [waveform_id,???,time]
                st.waveforms = np.transpose(sp_waves[unit_i]['data'][:,:,0]) * anasig.units
                st.waveforms = st.waveforms.reshape((st.waveforms.shape[0],1,st.waveforms.shape[1]))
                st.left_sweep = extraction_dict['sp_win_align'][0]
                # st.spike_duration = extraction_dict['sp_win_align'][1] - extraction_dict['sp_win_align'][0]
                # st.right_sweep = extraction_dict['sp_win_align'][1]
            else:
                st.waveforms = None

            # connecting unit, spiketrain and segment
            rcgs = anasig.recordingchannel.recordingchannelgroups
            u_annotations = {'sorted': sorted,
                             'parameters':{ 'sorting_params': sorting_params,
                                            'extraction_params': extraction_dict}}

            new_unit = None
            for rcg in rcgs:
                # checking if a similar unit already exists (eg. from sorting a different segment)
                rcg_units = [u for u in rcg.units if u.name == st.name and u.annotations == u_annotations]
                if len(rcg_units) == 1:
                    unit = rcg_units[0]
                elif len(rcg_units) == 0:
                    # Generating new unit if necessary
                    if new_unit is None:
                        new_unit = neo.core.Unit(name=st.name, **u_annotations)
                    unit = new_unit
                else:
                    raise ValueError('%i units of name %s and annotations %s exists.'
                                     ' This is ambiguous.' % (len(rcg_units), st.name, u_annotations))
                rcg.units.append(unit)
                unit.spiketrains.append(st)
            seg.spiketrains.append(st)

msg = 'Phy must be available to extract spikes using the PhySorter.'
@requires(phy, msg)
class PhySorter(SpikeSorter):
    # accepted_arguments = getKwArgs(sklearn.cluster.KMeans.__init__)

    def __init__(self, **parameter_dict):
        super(self.__class__, self).__init__(**parameter_dict)
    #
    # original_parameters = copy.deepcopy(parameter_dict)
    #
    # session_name = block.name
    # random_id = np.random.randint(0,10**10)
    # tempdir = tempfile.gettempdir()
    # prm_file_name = os.path.join(tempdir,'temp_phy_params_%s_%i.prm'%(session_name,random_id))
    # prb_file_name = os.path.join(tempdir,'temp_phy_probe_%s_%i.prb'%(session_name,random_id))
    # dummy_data_file_name = os.path.join(tempdir,'temp_phy_dummy_data_%s_%i.dat'%(session_name,random_id))
    # kwik_file_name = os.path.join(tempdir,'temp_phy_session_%s_%i.kwik'%(session_name,random_id))
    #
    # def _remove_temp_files(temp_files):
    #     for temp_file in temp_files:
    #         if os.path.isfile(temp_file):
    #             os.remove(temp_file)
    #         elif os.path.isdir(temp_file):
    #             os.rmdir(temp_file)
    #
    #
    # # removing temporary files after program finished
    # if 'keep_temp_files' in parameter_dict:
    #     if not parameter_dict['keep_temp_files']:
    #         atexit.register(_remove_temp_files,[prm_file_name,
    #                                            prb_file_name,
    #                                            dummy_data_file_name,
    #                                            kwik_file_name,
    #                                            # also remove files generated during spikesorting
    #                                            os.path.join(tempdir,kwik_file_name.replace('.kwik','.phy')),
    #                                            os.path.join(tempdir,kwik_file_name.replace('.kwik','.kwx')),
    #                                            os.path.join(tempdir,kwik_file_name.replace('.kwik','.log')),
    #                                            os.path.join(tempdir,kwik_file_name + '.bak')])
    #     parameter_dict.pop('keep_temp_files')
    #
    #
    # def _add_traces_to_params(block):
    #     # Extracting sampling rate
    #     sampling_rate = None
    #     n_channels = None
    #     for seg in block.segments:
    #         for anasig in seg.analogsignals:
    #             if sampling_rate == None:
    #                 sampling_rate = anasig.sampling_rate
    #             elif sampling_rate != anasig.sampling_rate:
    #                 raise ValueError('Analogsignals have different sampling '
    #                                  'rates. '
    #                                  'Phy can not extract spikes from signals with varying sampling rates.')
    #         if n_channels == None:
    #             n_channels = len(seg.analogsignals)
    #         elif n_channels != len(seg.analogsignals):
    #             raise ValueError('Segments contain different numbers of analogsignals. '
    #                              'Phy can not deal with different numbers of channels in one session.')
    #
    #
    #     parameter_dict['traces'] ={'raw_data_files':dummy_data_file_name,
    #                                'voltage_gain':1.0,
    #                                'sample_rate':sampling_rate.rescale('Hz').magnitude,
    #                                'n_channels':n_channels,
    #                                'dtype':'int16'}
    #
    #
    # def _generate_prm_file(phy_params):
    #     with open(prm_file_name, 'w') as f:
    #         for key0 in phy_params.iterkeys():
    #             if isinstance(phy_params[key0],dict):
    #                 f.write('%s = dict(%s'%(key0,sep))
    #                 for key, value in phy_params[key0].iteritems():
    #                     if isinstance(value,str):
    #                         value = "'%s'"%value
    #                     f.write('\t%s = %s,%s'%(key,value,sep))
    #                 f.write(')%s'%sep)
    #             else:
    #                 value = phy_params[key0]
    #                 if isinstance(value,str):
    #                     value = "'%s'"%value
    #                 f.write('%s = %s%s'%(key0,value,sep))
    #
    # def _generate_prb_file(phy_params,probe_type='linear'):
    #     if probe_type=='linear':
    #         n_channels = phy_params['traces']['n_channels']
    #         if n_channels == 1:
    #             warnings.warn('Individual spikes on multiple contacts can not be detected'
    #                           ' if spike sorting is performed on individual contacts (n_channels=1).')
    #         with open(prb_file_name, 'w') as f:
    #             f.write('channel_groups = {%s'%sep)
    #             f.write('\t0: {%s'%sep)
    #             f.write("\t\t'channels': %s,%s"%(range(n_channels),sep))
    #             f.write("\t\t'graph': %s,%s"%([[i,i+1] for i in range(n_channels-1)],sep))
    #             f.write("\t\t'geometry': %s%s"%(dict([[i,[0.0,float(i)/10]] for i in range(n_channels)]),sep))
    #             f.write('\t}%s'%sep)
    #             f.write('}')
    #     else:
    #         raise NotImplementedError('This functionality is only implemented for linear probes.')
    #
    # def _generate_dummy_data_file():
    #     with open(dummy_data_file_name, 'w') as f:
    #         f.write('dummy data')
    #
    #
    # _add_traces_to_params(block)
    # parameter_dict['prb_file'] = prb_file_name.split(os.path.sep)[-1]
    # _generate_prm_file(parameter_dict)
    # _generate_prb_file(parameter_dict,probe_type='linear')
    # _generate_dummy_data_file()
    #
    #
    # if os.path.isfile(kwik_file_name):
    #     warnings.warn('Deleting old kwik file %s to generate new spike sorting'%kwik_file_name)
    #     os.remove(kwik_file_name)
    #
    # # creating new kwik file for phy session
    # probe = phy.io.kwik.creator.load_probe(prb_file_name)
    # phy.io.create_kwik(prm_file_name,kwik_file_name,overwrite=False,probe=probe)
    #
    # # generating phy session
    # phy_session = phy.session.Session(kwik_file_name)
    #
    # def _merge_annotations(A, B):
    #     """
    #     From neo.core.baseneo, modified
    #     Merge two sets of annotations.
    #
    #     Merging follows these rules:
    #     All keys that are in A or B, but not both, are kept.
    #     For keys that are present in both:
    #         For arrays or lists: concatenate
    #         For dicts: merge recursively
    #         For strings: concatenate with ';'
    #         Otherwise: fail if the annotations are not equal
    #     """
    #     merged = {}
    #     for name in A:
    #         if name in B:
    #             try:
    #                 merged[name] = merge_annotation(A[name], B[name])
    #             except BaseException as exc:
    #                 exc.args += ('key %s' % name,)
    #                 raise
    #         else:
    #             merged[name] = A[name]
    #     for name in B:
    #         if name not in merged:
    #             merged[name] = B[name]
    #     return merged
    #
    # # TODO: This function belongs to a more general 'neo utility' module...
    # def merge_annotation(a, b):
    #         """
    #         From neo.core.baseneo, modified
    #         First attempt at a policy for merging annotations (intended for use with
    #         parallel computations using MPI). This policy needs to be discussed
    #         further, or we could allow the user to specify a policy.
    #
    #         Current policy:
    #             For arrays or lists: concatenate
    #             For dicts: merge recursively
    #             For strings: concatenate with ';'
    #             Otherwise: fail if the annotations are not equal
    #         """
    #
    #         if isinstance(a, list):  # concatenate b to a
    #             if isinstance(b, list):
    #                 return a + b
    #             else:
    #                 return a.append(b)
    #
    #         if type(a) != type(None) and type(b) != type(None):
    #             assert type(a) == type(b), 'type(%s) %s != type(%s) %s' % (a, type(a),
    #                                                                    b, type(b))
    #         if isinstance(a, dict):
    #             return _merge_annotations(a, b)
    #         elif isinstance(a, np.ndarray):  # concatenate b to a
    #             return np.append(a, b)
    #         elif isinstance(a, basestring):
    #             if a == b:
    #                 return a
    #             else:
    #                 return a + ";" + b
    #         else:
    #             return [a,b]
    #
    # def _hstack_signals(sig1,sig2):
    #     # This function is partially copied form neo analogsignal merge()
    #     sig1 = copy.deepcopy(sig1)
    #     sig2 = copy.deepcopy(sig2)
    #     assert sig1.sampling_rate == sig2.sampling_rate
    #     assert sig1.t_start == sig2.t_start
    #     assert len(sig1) == len(sig2)
    #     sig2.units = sig1.units
    #     # stack = np.hstack(np.array,(sig1,sig2.reshape(-1,1))) #np.hstack(map(np.array, (sig1, sig2)))
    #     kwargs = {}
    #     for name in ("name", "description", "file_origin","channel_index",'sampling_rate'):
    #         attr_sig1 = getattr(sig1, name)
    #         attr_sig2 = getattr(sig2, name)
    #         # if (not(hasattr(attr_sig1,'__iter__') or hasattr(attr_sig2,'__iter__')) \
    #         #     or ((type(attr_sig1)==pq.Quantity) and type(attr_sig2)==pq.Quantity)) \
    #         #         and attr_sig1 == attr_sig2:
    #         try:
    #             if attr_sig1 == attr_sig2:
    #                 kwargs[name] = attr_sig1
    #             else:
    #                 raise ValueError()
    #         except:
    #         # else:
    #             if type(attr_sig1) != list:
    #                 attr_sig1 = [attr_sig1]
    #             if type(attr_sig2) != list:
    #                 attr_sig2 = [attr_sig2]
    #             attr_sig1 = attr_sig1 + attr_sig2
    #             setattr(sig1,name,attr_sig1)
    #             setattr(sig2,name,attr_sig1)
    #
    #     if 'channel_index' in sig1.annotations:
    #         sig1.annotations.pop('channel_index')
    #     if 'sampling_rate' in sig1.annotations:
    #         sig1.annotations.pop('sampling_rate')
    #     if 't_start' in sig1.annotations:
    #         sig1.annotations.pop('t_start')
    #
    #     merged_annotations = merge_annotation(sig1.annotations,
    #                                            sig2.annotations)
    #
    #     sig2 = sig2.reshape((-1,1))
    #
    #     stacked = np.hstack((sig1,sig2))
    #     stacked.__dict__ = sig1.__dict__.copy()
    #     stacked.annotations = merged_annotations
    #
    #     return stacked
    #
    # def _kwik_spikes_to_neo_block(seg,traces,waveforms, sort):
    #     #read results from kwik file(s) or phy_session
    #
    #     kwikfile = phy.io.h5.File(kwik_file_name)
    #     kwikfile.open()
    #     time_samples = kwikfile.read('/channel_groups/0/spikes/time_samples')
    #     time_fractional = kwikfile.read('channel_groups/0/spikes/time_fractional')
    #     cluster_ids = np.asarray(kwikfile.read('/channel_groups/0/spikes/clusters/main'))
    #     spike_channel_masks = np.asarray([phy_session.model.masks[i] for i in range(len(time_samples))])
    #
    #     phy_session.store.is_consistent()
    #
    #     if waveforms:
    #         try:
    #             kwxfile = phy.io.h5.File(kwik_file_name.replace('.kwik','.kwx'))
    #             kwxfile.open()
    #             if kwxfile.exists('/channel_groups/0/waveforms_raw'):
    #                 waveforms_raw = kwxfile.read('/channel_groups/0/waveforms_raw')
    #             else:
    #                 waveforms_raw = phy_session.model.waveforms[range(phy_session.n_spikes)]
    #
    #             # if kwxfile.exists('/channel_groups/0/waveforms_filtered'):
    #             #     waveforms_filtered = kwxfile.read('/channel_groups/0/waveforms_filtered')
    #             # else:
    #             #     waveforms_filtered = phy_session.store.waveforms(0,'filtered')
    #             if kwxfile.exists('/channel_groups/0/features_masks'):
    #                 features_masks = kwxfile.read('/channel_groups/0/features_masks')
    #             else:
    #                 features = phy_session.store.features(0)
    #                 features_masks = phy_session.model.features_masks[range(phy_session.n_spikes)]
    #         except KeyError:
    #             warnings.warn('Could not extract wavefroms from kwik file or phy_session due to inconsistencies.')
    #             waveforms = False
    #
    #     spiketimes = (np.asarray(time_samples) / traces.sampling_rate) + t_start
    #
    #     for i,unit_id in enumerate(np.unique(cluster_ids)):
    #         unit_mask = cluster_ids == unit_id
    #
    #         for channel_id in range(phy_session.model.n_channels):
    #             print 'unit %i, channel %i'%(unit_id,channel_id)
    #             channel_id = int(channel_id)
    #             unit_channel_mask = np.where(np.logical_and(spike_channel_masks[:,channel_id] > 0, unit_mask) == True)
    #             unit_spikes = spiketimes[unit_channel_mask]
    #
    #             if len(unit_spikes) < 1:
    #                 continue
    #
    #             original_anasig = seg.analogsignals[channel_id]
    #
    #             # generating spiketrains
    #             st = neo.SpikeTrain(times=unit_spikes,
    #                                 t_start=traces.t_start,
    #                                 t_stop=traces.t_stop,
    #                                 sampling_rate=traces.sampling_rate,
    #                                 name="Channel %i, Unit %i" % (original_anasig.channel_index, unit_id),
    #                                 file_origin=traces.file_origin,
    #                                 unit_id=unit_id,
    #                                 channel_id=anasig.annotations['channel_index'],
    #                                 electrode_id=anasig.annotations['electrode_id'],
    #                                 sorted=sort,
    #                                 sorting_parameters=parameter_dict['klustakwik2'],
    #                                 extraction_params=parameter_dict['spikedetekt'],
    #                                 prb_file=parameter_dict['prb_file'],
    #                                 # channel_affiliations=spike_channel_masks[unit_channel_mask,channel_id]
    #                                 )
    #
    #             if waveforms:
    #                 # waveform dimensions [waveform_id,??,time]
    #                 st.waveforms = waveforms_raw[unit_channel_mask][:,:,channel_id] * original_anasig.units
    #                 st.waveforms = st.waveforms.reshape((st.waveforms.shape[0],1,st.waveforms.shape[1]))
    #                 st.left_sweep = -1 * parameter_dict['spikedetekt']['extract_s_before'] / anasig.sampling_rate
    #                 # st.spike_duration = (parameter_dict['spikedetekt']['extract_s_after']) / anasig.sampling_rate  -st.left_sweep
    #                 # st.right_sweep = parameter_dict['spikedetekt']['extract_s_after'] / anasig.sampling_rate
    #             else:
    #                 st.waveforms = None
    #
    #             # connecting unit, spiketrain and segment
    #             rcgs = anasig.recordingchannel.recordingchannelgroups
    #             u_annotations = {'sorted': sort,
    #                              'parameters': original_parameters}
    #
    #             new_unit = None
    #             for rcg in rcgs:
    #                 # checking if a similar unit already exists (eg. from sorting a different segment)
    #                 rcg_units = [u for u in rcg.units if u.name == st.name and u.annotations == u_annotations]
    #                 if len(rcg_units) == 1:
    #                     unit = rcg_units[0]
    #                 elif len(rcg_units) == 0:
    #                     # Generating new unit if necessary
    #                     if new_unit is None:
    #                         new_unit = neo.core.Unit(name=st.name, **u_annotations)
    #                     unit = new_unit
    #                 else:
    #                     raise ValueError('%i units of name %s and annotations %s exists.'
    #                                      ' This is ambiguous.' % (len(rcg_units), st.name, u_annotations))
    #                 rcg.units.append(unit)
    #                 unit.spiketrains.append(st)
    #             seg.spiketrains.append(st)
    #
    #
    #
    #
    #
    # # get maximal time period, where all analogsignals are present and collect signals in analogsignal
    # for seg in block.segments:
    #     traces = None
    #     for anasig in seg.analogsignals:
    #         if type(traces) == type(None):
    #             traces = anasig.reshape((-1,1))
    #
    #         else:
    #             # adjusting length of signals
    #             if anasig.t_start<traces.t_start:
    #                 anasig.timeslice(traces.t_start,None)
    #             elif anasig.t_start>traces.t_start:
    #                 traces.time_slice(anasig.t_start,None)
    #
    #             if anasig.t_stop>traces.t_stop:
    #                 anasig.timeslice(None,traces.t_stop)
    #             elif anasig.t_stop<traces.t_stop:
    #                 traces.time_slice(None,traces.t_stop)
    #
    #             # merging signals into one analogsignal
    #             traces = _hstack_signals(traces,anasig)
    #
    #     t_start, t_stop = traces.t_start, traces.t_stop
    #
    #     #detecting spikes using blank kwik file and lfp traces from neo block
    #     print 'Starting spike detection and extraction on %i (%i) anasigs.'%(len(seg.analogsignals), traces.shape[1])
    #     phy_session.detect(np.asarray(traces))
    #     phy_session.save()
    #
    #     if sort:
    #         print 'Starting spike clustering.'
    #         phy_session.cluster()
    #         phy_session.save()
    #
    #
    #     _kwik_spikes_to_neo_block(seg,traces,waveforms,sort)

