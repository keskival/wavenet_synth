#!/usr/bin/python

import traceback

import matplotlib
matplotlib.use('Agg')
import pylab

import tensorflow as tf
import numpy as np

import random
import json
import itertools
import math

import manage_data
import export_to_octave
import operations

import time
import sys
import scipy.fftpack
import numpy.fft

def make_x_and_y(x, noise, amplitude_plusminus_factor):
    # Length of shifted x and y
    length = np.size(x, 0) - 1
    # Removing the first item of y.
    y = np.copy(np.asarray(x[1:length + 1]))
    # Removing the last item x to be the last item to predict.
    new_x = np.clip(np.copy(np.asarray(x[0:length])) * random.uniform(1.0 - amplitude_plusminus_factor, 1.0 + amplitude_plusminus_factor), -1.0, 1.0)
    # Adding salt and pepper noise to x.
    number_of_corruptions = int(noise * length)
    for i in range(number_of_corruptions):
        index = random.randint(0, length - 1)
        new_x[index] = random.uniform(-1.0, 1.0)
    return (x, new_x, y)

def train(parameters, model, trainingData, testingData, startingModel=None, minutes=60 * 24, name="", loss_improved_limit=50):
    print('Launching training.')

    init = tf.initialize_all_variables()
    saver = tf.train.Saver(tf.all_variables())
    # Launch the graph
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)
    start_time = time.time()
    iters_since_loss_improved = 0
    with tf.Session(config=config) as sess:
        if startingModel:
            saver.restore(sess, startingModel)
        else:
            sess.run(init)
        
        writer = tf.train.SummaryWriter("logs", sess.graph)
    
        iter = 1
        train_error_trend = []
        test_error_trend = []
        now = time.time()
        # Training for a specific number of minutes
        last_losses = []
        last_loss = None
        best_loss = 1e20
        training_length = parameters['training_length']
        while now - start_time < 60 * minutes and iters_since_loss_improved < loss_improved_limit:

            # Note: Taking sample length times 2 to have a good amount of full input window examples.
            x = manage_data.getNextTrainingBatchSequence(trainingData, training_length)
            (original_x, x, y) = make_x_and_y(x, parameters['input_salt_and_pepper_noise'],
                                              parameters['amplitude_plusminus_factor'])

            # Fit training using batch data

            (_, cost) = sess.run([model['optimizer'], model['cost']], feed_dict = {
                model['input']: x,
                model['schedule_step']: iter,
                model['noise']: parameters['noise'],
                model['input_noise']: parameters['input_noise'],
                model['target_output']: y
            })
            print "Time elapsed: ", now - start_time, ", iter: ", iter, \
                ", training cost: ", cost
                
            if iter % parameters['display_step'] == 0:
                if last_loss:
                    print "Time elapsed: ", now - start_time, ", last median testing loss: ", last_loss, \
                        ", best median testing loss: ", best_loss, \
                        ", iters_since_loss_improved: ", iters_since_loss_improved
                saver.save(sess, 'sound-model')
                
                [error, output] = sess.run([tf.stop_gradient(model['cost']), model['output']], feed_dict = {
                    model['input']: x,
                    model['schedule_step']: iter,
                    model['noise']: parameters['noise'],
                    model['input_noise']: parameters['input_noise'],
                    model['target_output']: y
                })
                train_error_trend.append(error)
                if (len(train_error_trend) > 10000):
                    train_error_trend.pop(0)
                #export_to_octave.save('output.mat', 'output', output)
                export_to_octave.save('input.mat', 'input', original_x)
                export_to_octave.save('corrupted.mat', 'input', x)

                def choose_value(sample):
                    sample = np.asarray(sample)
                    sample /= sample.sum()
                    sampled = np.random.choice(np.arange(parameters['quantization_channels']), p=sample)
                    return operations.de_mu_law(sampled, float(parameters['quantization_channels'] - 1))

                realization = np.asarray(map(choose_value, output.tolist()))
                export_to_octave.save('realization.mat', 'realization', realization)

                print "Iter {}".format(iter) + ", Testing Loss={}".format(error)

                test_x = manage_data.getNextTrainingBatchSequence(testingData, training_length)
                (original_test_x, test_x, test_y) = make_x_and_y(test_x, 0.0, 0.0)

                [test_error, test_output] = sess.run([tf.stop_gradient(model['cost']), tf.stop_gradient(model['output'])],
                    feed_dict={
                               model['input']: test_x,
                               model['schedule_step']: iter,
                               model['noise']: 0.0,
                               model['input_noise']: 0.0,
                               model['target_output']: test_y
                    })
                test_realization = np.asarray(map(choose_value, test_output.tolist()))
                export_to_octave.save('test_input.mat', 'test_input', original_test_x)
                export_to_octave.save('test_realization.mat', 'test_realization', test_realization)

                test_error_trend.append(test_error)
                if (len(test_error_trend) > 10000):
                    test_error_trend.pop(0)

                last_losses.append(test_error)
                # Taking the median of the 15 last testing losses.
                if (len(last_losses) > 15):
                    last_losses.pop(0)
                last_loss = np.median(last_losses)
                if (last_loss and last_loss < best_loss):
                    best_loss = last_loss
                    iters_since_loss_improved = 0
                    if name:
                        saver.save(sess, 'sound-model-best-' + name)
                    else:
                        saver.save(sess, 'sound-model-best')
                else:
                    iters_since_loss_improved = iters_since_loss_improved + 1
                print "Testing Error:", test_error
                print "Last loss:", last_loss
                if name:
                    export_to_octave.save('train_error_' + name + '.mat', 'train_error', train_error_trend)
                    export_to_octave.save('test_error_' + name + '.mat', 'test_error', test_error_trend)
                else:
                    export_to_octave.save('train_error.mat', 'train_error', train_error_trend)
                    export_to_octave.save('test_error.mat', 'test_error', test_error_trend)
                sys.stdout.flush()
            iter += 1
            now = time.time()
        if name:
            saver.save(sess, 'sound-model-final-' + name, global_step=iter)
        else:
            saver.save(sess, 'sound-model-final', global_step=iter)
        print "Optimization Finished!"
        sys.stdout.flush()

        # Returning the last loss value for hyper parameter search
        return last_loss
    