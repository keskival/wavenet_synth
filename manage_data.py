#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import pylab

import tensorflow as tf
import numpy as np
import scipy.fftpack
import numpy.fft

import random
import json
import itertools

# Returns one sequence of n_steps.
def getNextTrainingBatch(data, n_steps):
    # A random displacement to take the batch from.
    data_length = data.shape[0]
    disp = random.randint(0, len(data[:]) - n_steps - 1) # 200000 + random.randint(0, 3) * 100000 #
    return data[disp:disp + n_steps]

def getNextTrainingBatchSequence(data, sample_size):
    result = []
    sequence = getNextTrainingBatch(data, sample_size)
    x = np.asarray(sequence)
    return x
