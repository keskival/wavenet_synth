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
import scipy.io as sio

import params
import model
import train
import export_to_octave
import operations
parameters = params.parameters
print parameters

# 440 Hz
#t = np.asarray(range(parameters['sample_length'])) / 48000.0 * 2.0 * np.pi * 440
#signal = np.sin(t)
# Zeros
#t = np.zeros(parameters['sample_length'])
#(_, a1) = sio.wavfile.read("seeds/a1.wav")
#signal = a1 / (2.**15)
(_, i) = sio.wavfile.read("input.wav")
signal = i / (2.**15)

output_signal = np.copy(np.asarray(signal))
signal = np.asarray(signal[len(signal) - parameters['sample_length'] : len(signal)])
p = pyaudio.PyAudio()

generative_model = model.create_generative_model(parameters)

init = tf.initialize_all_variables()
saver = tf.train.Saver(tf.all_variables())
gpu_options = tf.GPUOptions()
config = tf.ConfigProto(gpu_options=gpu_options)

image = []

with tf.Session(config=config) as sess:
    saver.restore(sess, 'sound-model-best')
    # Creating a 100 second sample
    next_val = 0.0
    for i in range(48000 * 100):
        print "Step: ", i
        [probabilities] = sess.run([generative_model['generated_output']], feed_dict = {
                generative_model['input']: signal
            })
        #image.append(probabilities)

        def choose_value(sample, prev_value):
            sample = np.asarray(sample)
            sample /= sample.sum()
            sampled = np.random.choice(np.arange(parameters['quantization_channels']), p=sample)
            probability_selected = sample[sampled]
            # Decreasing the weight of small probability selections.
            #factor = min(1.0, probability_selected / 0.03)
            new_value = operations.de_mu_law(sampled, float(parameters['quantization_channels'] - 1))
            #final_sample = new_value * factor + prev_value * (1-factor)
            print "Sampled, new_value, probability_selected: ", sampled, new_value, probability_selected
            return new_value
        
        next_val = choose_value(probabilities, next_val)
        signal = np.append(signal, next_val)[1:]
        output_signal = np.append(output_signal, next_val)
        #export_to_octave.save('image.mat', 'i', image)
        wav = np.asarray(map(int, output_signal * (2.**15)), dtype=np.int16)
        wav2 = np.asarray(map(int, signal * (2.**15)), dtype=np.int16)
        export_to_octave.save('sound.mat', 's', wav)
        export_to_octave.save('sound2.mat', 's', wav2)
    stream = p.open(format=p.get_format_from_width(2),
                channels=1,
                rate=sampleFreq,
                output=True)

    for chunk in np.array_split(wav, 100):
        stream.write(chunk, np.size(chunk, 0))

    stream.stop_stream()
    stream.close()

    p.terminate()
