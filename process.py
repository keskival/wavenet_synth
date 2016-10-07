#!/usr/bin/python
import traceback

from scipy.io import wavfile

import tensorflow as tf
import numpy as np

import random
import json
import itertools
import math
import time
import matplotlib.pyplot as plt

import params
import model
import train

print params.parameters

sample_freq, snd = wavfile.read('red.wav')
sample_freq_test, test = wavfile.read('test.wav')
snd = snd / (2.**15)
snd_test = test / (2.**15)
training_data = snd[:,0]
testing_data = snd_test[:,0]
print "Length of training data: ", len(training_data)
print "Length of test data: ", len(testing_data)
parameters = params.parameters

model = model.create(params.parameters)

train.train(params.parameters, model, training_data, testing_data, minutes=60 * 24 * 7, loss_improved_limit=10000)
#train.train(params.parameters, model, training_data, testing_data, starting_model="sound-model", minutes=60 * 24 * 7, loss_improved_limit=10000)
