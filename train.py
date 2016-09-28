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

import time
import sys
import scipy.fftpack
import numpy.fft

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
        
        step = 1
        trainErrorTrend = []
        testErrorTrend = []
        now = time.time()
        # Training for a specific number of minutes
        last_losses = []
        last_loss = None
        best_loss = 1e20
        sample_length = parameters['sample_length']
        while now - start_time < 60 * minutes and iters_since_loss_improved < loss_improved_limit:
            if last_loss:
                print "Time elapsed: ", now - start_time, ", last_loss: ", last_loss, \
                      ", best_loss: ", best_loss, \
                      ", iters_since_loss_improved: ", iters_since_loss_improved

            batch_xs = manage_data.getNextTrainingBatchSequence(trainingData, sample_length)
            
            # Fit training using batch data

            sess.run([model['optimizer']], feed_dict = {
                model['x']: batch_xs
            })
            if step % parameters['display_step'] == 0:
                saver.save(sess, 'sound-model')
                
                [error, prediction] = sess.run([tf.stop_gradient(model['cost']), tf.stop_gradient(model['output'])], feed_dict = {
                    model['x']: batch_xs
                })
                trainErrorTrend.append(error)

                print "Iter {}".format(iter) + ", Loss={}".format(error)

                test_x = manage_data.getNextTrainingBatchSequence(testingData, sample_length)

                [testError] = sess.run([tf.stop_gradient(model['cost'])],
                    feed_dict={model['x']: test_x})
                testErrorTrend.append(testError)
                last_losses.append(testError)
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
                print "Testing Error:", testError
                print "Last loss:", last_loss
                if name:
                    export_to_octave.save('train_error_' + name + '.mat', 'train_error', trainErrorTrend)
                    export_to_octave.save('test_error_' + name + '.mat', 'test_error', testErrorTrend)
                else:
                    export_to_octave.save('train_error.mat', 'train_error', trainErrorTrend)
                    export_to_octave.save('test_error.mat', 'test_error', testErrorTrend)

                print "prediction: ", prediction
                sys.stdout.flush()
            iter += 1
            step += 1
            now = time.time()
        if name:
            saver.save(sess, 'sound-model-final-' + name, global_step=iter)
        else:
            saver.save(sess, 'sound-model-final', global_step=iter)
        print "Optimization Finished!"
        sys.stdout.flush()

        # Returning the last loss value for hyper parameter search
        return last_loss
    