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
import matplotlib.pyplot as plt

import params
import model
import train
import export_to_octave

def de_mu_law(y, mu):
    scaled = 2 * (y / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) ** abs(scaled) - 1)
    return np.sign(scaled) * magnitude

parameters = params.parameters
print parameters

signal = np.append(np.zeros(parameters['sample_length']), random.randrange(-1,1))

p = pyaudio.PyAudio()

generative_model = model.create_generative_model(parameters)

init = tf.initialize_all_variables()
saver = tf.train.Saver(tf.all_variables())
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80, allocator_type = 'BFC')
config = tf.ConfigProto(gpu_options=gpu_options)

image = []

with tf.Session(config=config) as sess:
    saver.restore(sess, 'sound-model-best-')
    # Creating a 10 second sample
    for i in range(44100 * 10):
        print i
        [probabilities] = sess.run([generative_model['generated_output']], feed_dict = {
                generative_model['x']: signal
            })
        image.append(probabilities)
        next_val = np.random.choice(np.arange(parameters['quantization_channels']), p=probabilities)
        value = de_mu_law(next_val, float(parameters['quantization_channels'] - 1))
        signal = np.append(signal, value)
        export_to_octave.save('image.mat', 'i', image)
        wav = np.asarray(map(int, signal * (2.**15)), dtype=np.int16)
        export_to_octave.save('sound.mat', 's', wav)
    stream = p.open(format=p.get_format_from_width(2),
                channels=1,
                rate=sampleFreq,
                output=True)

    for chunk in np.array_split(wav, 100):
        stream.write(chunk, np.size(chunk, 0))

    stream.stop_stream()
    stream.close()

    p.terminate()
