import sys
import numpy
import quantities as pq
sys.path.append('./elephant/')
sys.path.append('./python-neo/')
sys.path.append('./elephant_dev/')
import spike_train_correlation as corr
import elephant.spike_train_surrogates as surr
import elephant.statistics as stats
import elephant.conversion as conv
import stocmod as stocmod
import cubic as cubic
import misc
import matplotlib.pyplot as plt
import neo
# %matplotlib inline


