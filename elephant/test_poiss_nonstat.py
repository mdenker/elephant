import elephant.spike_train_generation as stg
import elephant.statistics as stat
import numpy as np
import neo
import matplotlib.pyplot as plt
import quantities as pq
import time

sampling_period = 0.001*pq.s
samples = 10000
cosine = np.cos(np.arange(5,10,0.001))+2
rate = neo.AnalogSignal([2]*len(cosine)*pq.Hz, sampling_period=sampling_period)
# t_0_thin = time.time()
# sts_thin = stg.poisson_nonstat_thinning(rate_signal=rate, n=samples)
# t_thin = time.time() - t_0_thin
# rate_thin = stat.time_histogram(
#     sts_thin,sampling_period)/(sampling_period.magnitude*np.array(samples))
t_0_warp = time.time()
sts_warp = stg.poisson_nonstat_time_rescale(rate_signal=rate,n=samples)
t_warp = time.time() - t_0_warp
rate_warp = stat.time_histogram(
    sts_warp,sampling_period*10)/(sampling_period.magnitude*10*np.array(samples))
# print(rate[-3:])
# print(rate[:3])
cosine = np.cos(np.arange(5,10,0.01))+2
rate = neo.AnalogSignal([5]*len(cosine)*pq.Hz, sampling_period=sampling_period*10)
print('rate_warp')
print(rate_warp[:4]-rate[:4].magnitude)
print(rate_warp[-4:]-rate[-4:].magnitude)

# print(rate_thin[-3:])
plt.plot(rate.times, rate)
# plt.plot(rate.times, rate_thin, 'r')
plt.plot(rate.times, rate_warp,'g')

