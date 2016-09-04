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
import pyaudio

import params
import model
import train

sampleFreq, snd = wavfile.read('red.wav')
print snd.dtype
snd = snd / (2.**15)
print snd.shape
print sampleFreq
print snd.shape[0] / sampleFreq

channel1 = snd[:,0]

p = pyaudio.PyAudio()

model = model.create(params.parameters)

training_data = channel1[0:len(channel1)/3*2]
testing_data = channel1[len(channel1)/3*2:len(channel1)]

train.train(params.parameters, model, training_data, testing_data, None, 30)

wav = np.asarray(map(int, channel1 * (2.**15)), dtype=np.int16)

#stream = p.open(format=p.get_format_from_width(2),
#                channels=1,
#                rate=sampleFreq,
#                output=True)

#for chunk in np.array_split(wav, 100):
#    stream.write(chunk, np.size(chunk, 0))

#stream.stop_stream()
#stream.close()

#p.terminate()
