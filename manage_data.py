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
def getNextTrainingBatch(data, n_steps, iter):
    # A random displacement to take the batch from.
    data_length = data.shape[0]
    if iter:
        disp = iter * n_steps
        if (disp > len(data[:]) - n_steps - 1):
            disp = 0
    else:
        disp = random.randint(0, len(data[:]) - n_steps - 1)
    return data[disp:disp + n_steps]

def getNextTrainingBatchSequence(data, sample_size, iter = None):
    result = []
    sequence = getNextTrainingBatch(data, sample_size, iter)
    x = np.asarray(sequence)
    return x
