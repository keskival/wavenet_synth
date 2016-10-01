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

sampleFreq, snd = wavfile.read('red.wav')
snd = snd / (2.**15)
channel1 = snd[:,0]
parameters = params.parameters

model = model.create(params.parameters)

# Dividing the data from the given audio file to training (3/4) and test sets (1/4).
# Validation set is not used here.
training_data = channel1[0:len(channel1)/4*3]
testing_data = channel1[len(channel1)/4*3:len(channel1)]

train.train(params.parameters, model, training_data, testing_data, minutes=60 * 24, loss_improved_limit=10000)
