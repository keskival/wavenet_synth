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
import pyaudio
import scipy.fftpack
import numpy.fft

p = pyaudio.PyAudio()

def train(parameters, model, trainingData, testingData, start, minutes, name="", loss_improved_limit=50):
    print('Launching training.')

#for chunk in np.array_split(wav, 100):

#stream.stop_stream()
#stream.close()

#p.terminate()
#    accuracy_summary = tf.scalar_summary("cost", model["cost"])
#    merged = tf.merge_all_summaries()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver(tf.all_variables())
    # Launch the graph
    # config=tf.ConfigProto(log_device_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80, allocator_type = 'BFC')
    config = tf.ConfigProto(gpu_options=gpu_options)
    start_time = time.time()
    iters_since_loss_improved = 0
    with tf.Session(config=config) as sess:
        if start:
            saver.restore(sess, start)
        else:
            sess.run(init)
        
#        writer = tf.train.SummaryWriter("logs", sess.graph)
    
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
        #while step / parameters['display_step'] <= 300:
        while now - start_time < 60 * minutes and iters_since_loss_improved < loss_improved_limit:
            if last_loss:
                print "Time elapsed: ", now - start_time, ", last_loss: ", last_loss, \
                      ", best_loss: ", best_loss, \
                      ", iters_since_loss_improved: ", iters_since_loss_improved
            #export_to_octave.save('training_data_d.mat', 'trainingData', trainingData)
        
            # parameters['learning_rate'] = parameters['learning_rate'] * parameters['decay']
            batch_xs = manage_data.getNextTrainingBatchSequence(trainingData, sample_length)
            
            # Fit training using batch data

            sess.run([model['optimizer']], feed_dict = {
                model['x']: batch_xs
            })
            if step % parameters['display_step'] == 0:
                saver.save(sess, 'sound-model')
                
                # For debugging, exporting a couple of arrays to Octave.
                #export_to_octave.save('batch_xs.mat', 'batch_xs', batch_xs)
                
                #batch_xs = manage_data.getNextTrainingBatchSequences(trainingData,
                #    parameters['batch_size'], parameters['half_sample_length'])
                # Calculate batch error as mean distance
                [error, prediction] = sess.run([tf.stop_gradient(model['cost']), tf.stop_gradient(model['output'])], feed_dict = {
                    model['x']: batch_xs
                })
                trainErrorTrend.append(error)

                # Calculate batch loss
                print "Iter {}".format(iter) + ", Loss={}".format(error)

                test_x = manage_data.getNextTrainingBatchSequence(testingData, sample_length)

                [testError] = sess.run([tf.stop_gradient(model['cost'])],
                    feed_dict={model['x']: test_x})
                ##writer.add_summary(summary_str, iter)
                testErrorTrend.append(testError)
                last_losses.append(testError)
                # Taking the median of the 10 last testing losses.
                if (len(last_losses) > 10):
                    last_losses.pop(0)
                last_loss = np.median(last_losses)
                if (last_loss and last_loss < best_loss):
                    best_loss = last_loss
                    iters_since_loss_improved = 0
                    saver.save(sess, 'sound-model-best-' + name)
                else:
                    iters_since_loss_improved = iters_since_loss_improved + 1
                print "Testing Error:", testError
                print "Last loss:", last_loss
                export_to_octave.save('train_error_' + name + '.mat', 'train_error', trainErrorTrend)
                export_to_octave.save('test_error_' + name + '.mat', 'test_error', testErrorTrend)

                print "prediction: ", prediction
                #wav = np.int16(prediction[0] * (2.**15))
                #stream = p.open(format=p.get_format_from_width(2),
                #    channels=1,
                #    rate=44100,
                #    output=True)
                #stream.write(wav)
                #stream.stop_stream()
                #stream.close()
                ## Random activations from 0.0-1.0.                
                ##random_act = np.random.uniform(size=64)
                # Random one hot.
                sys.stdout.flush()
            iter += 1
            step += 1
            now = time.time()
        saver.save(sess, 'sound-model-' + name, global_step=iter)
        print "Optimization Finished!"
        sys.stdout.flush()

        p.terminate()
        # Returning the last loss value for hyper parameter search
        return last_loss
    