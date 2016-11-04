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
import argparse

print params.parameters

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None,
                    help='The name of the saved checkpoint to start training from. For example "sound-model". ' +
                    'The default value is empty and that means starting training from scratch. ' +
                    'Note that despite the name given here, the trained model will always be saved in ' +
                    'a file named "sound-model"')
args = parser.parse_args()

sample_freq, snd = wavfile.read('corpus.wav')
sample_freq_test, test = wavfile.read('test.wav')
snd = snd / (2.**15)
snd_test = test / (2.**15)
training_data = snd[:,0]
testing_data = snd_test[:,0]
print "Length of training data: ", len(training_data)
print "Length of test data: ", len(testing_data)
parameters = params.parameters

trainable_model = None

# Note that this model is too large to run on a normal GPU. Trying to run it on GPU only gives you
# grey hairs prematurely. If you want a model that can be run on a very high-end GPU, try the ibab Wavenet
# implementation from GitHub.
with tf.device("/cpu:0"):
    trainable_model = model.create(params.parameters)

train.train(params.parameters, trainable_model, training_data, testing_data, starting_model=args.model,
            minutes=60 * 24 * 7, loss_improved_limit=10000)
