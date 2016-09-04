#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import pylab

import tensorflow as tf
import numpy as np

import random
import json
import itertools

# Returns one sequence of n_steps.
def getNextTrainingBatch(data, n_steps):
    # A random displacement to take the batch from.
    data_length = data.shape[0]
    disp = random.randint(0, len(data[:]) - n_steps - 1)
    return data[disp:disp + n_steps]

def getNextTrainingBatchSequences(data, batch_size, half_sample):
    result = []
    for batch in range(batch_size):
        sequence = getNextTrainingBatch(data, half_sample * 3)
        result.append(sequence)
    x = np.asarray(result)
    return x
