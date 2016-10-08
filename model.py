#!/usr/bin/python

import matplotlib
from boto.gs.acl import SCOPE
from atk import Layer
matplotlib.use('Agg')
import pylab
import math

import tensorflow as tf
from tensorflow.python.ops.math_ops import real
import numpy as np

import random
import json
import itertools
import sys

import ops

def xavier_halfrange(n_in):
    ''' Returns the variance to use with Xavier initialization for sigmoid/tanh outputs '''
    return math.sqrt(12.0 * (1.0 / n_in)) / 2.0

def xavier_halfrange_rect(n_in):
    ''' Returns the variance to use with Xavier initialization for rectified outputs '''
    return math.sqrt(12.0 * (2.0 / n_in)) / 2.0

def mu_law(x, mu, noise = 0):
    ml = tf.sign(x) * tf.log(mu * tf.abs(x) + 1.0) / tf.log(mu + 1.0)
    # Additive noise to prevent overlearning. Overlearning makes it difficult to retrieve patterns,
    # so it is very problematic.
    #noise_vec = tf.random_normal(tf.shape(x), 0.0, noise)
    #noise_vec = tf.Print(noise_vec, [noise_vec, ml], "Noise and mu law: ")
    #ml = tf.add_n([ml, noise_vec])
    # Clamping between -1 and 1.
    #ml = tf.clip_by_value(ml, -1.0, 1.0)
    # Scaling between 0 and quantization_channels-1 integers.
    return tf.cast((ml + 1.0) / 2.0 * mu + 0.5, tf.int32)

# value shape is [width, quantization_channels]
# filters shape is [filter_width, quantization_channels, dilation_channels]
# In some implementations dilation_channels is 256.
def causal_atrous_conv1d(value, filters, rate, padding):
    # Using height in 2-D as the 1-D. Adding the batch dimension also.
    # Note that for filters using 'SAME' padding, padding zeros are added to the end of the input.
    # This means that for causal convolutions, we must shift the output right.
    # add zeros to the start and remove the future values from the end.

    value_with_batch = tf.expand_dims(value, 0)
    # Normally we would use this, but in practice CuDNN does not have implementations for the strided convolutions
    # so this only works for CPU.
    # value_2d = tf.expand_dims(value_with_batch, 2)
    # filters_2d = tf.expand_dims(filters, 1)
    # atrous_conv = tf.nn.atrous_conv2d(value_2d, filters_2d, rate, padding)
    # # Squeezing out the width and the batch dimensions.
    # atr_conv_1d = tf.squeeze(atrous_conv, [0, 2])
    # width = tf.shape(value)[0]
    # filter_shape = tf.shape(filters)
    # filter_width = filter_shape[0]
    # filter_width_up = filter_width + (filter_width - 1) * (rate - 1)
    # pad_width = filter_width_up - 1
    # pad_left = pad_width // 2
    # pad_right = pad_width - pad_left
    # # We want to shift the result so that acausal values are removed.
    # # Any value in the output that makes use of right padding values are acausal.
    # # So, we remove pad_right elements from the end, and add as many zeros to the beginning.
    # dilation_channels = tf.shape(atr_conv_1d)[1]
    # causal = tf.pad(tf.slice(atr_conv_1d, [0, 0], [width - pad_right, dilation_channels]),
    #                 [[pad_right, 0], [0, 0]])
    # return causal

    # Instead we use this implementation from Igor Babuschkin:
    atr_conv_1d_with_batch = ops.causal_conv(value_with_batch, filters, rate)
    atr_conv_1d = tf.squeeze(atr_conv_1d_with_batch, [0])
    # atr_conv_1d shape is [width, dilation_channels]

    return atr_conv_1d

# Returns a tuple of output to the next layer and skip output.
# The shape of x is [width, dense_channels]
def gated_unit(x, dilation, parameters, layer_index, noise):
    tf.histogram_summary('{}_x'.format(layer_index), x)
    
    filter_width = parameters['filter_width']
    dense_channels = parameters['dense_channels']
    dilation_channels = parameters['dilation_channels']
    quantization_channels = parameters['quantization_channels']

    #noise_vec = tf.random_normal(tf.shape(x), 0.0, noise)
    #x = x + noise_vec

    w1 = tf.Variable(tf.random_uniform([filter_width, dense_channels, dilation_channels], -xavier_halfrange(dense_channels), xavier_halfrange(dense_channels)),
            dtype=tf.float32, name='w1')
    w2 = tf.Variable(tf.random_uniform([filter_width, dense_channels, dilation_channels], -xavier_halfrange(dense_channels), xavier_halfrange(dense_channels)),
            dtype=tf.float32, name='w2')
    cw = tf.Variable(tf.random_uniform([1, dilation_channels, dense_channels], -xavier_halfrange_rect(dilation_channels), xavier_halfrange_rect(dilation_channels)),
            dtype=tf.float32, name='cw')

    tf.histogram_summary('{}_w1'.format(layer_index), w1)
    tf.histogram_summary('{}_w2'.format(layer_index), w2)
    tf.histogram_summary('{}_cw'.format(layer_index), cw)
    
    with tf.name_scope('causal_atrous_convolution'):
        dilated1 = causal_atrous_conv1d(x, w1, dilation, 'SAME')
        dilated2 = causal_atrous_conv1d(x, w2, dilation, 'SAME')
    with tf.name_scope('gated_unit'):
        z = tf.mul(tf.tanh(dilated1), tf.sigmoid(dilated2))
    # dilated1, dilated2, z shapes are [width, dilation_channels]
    skip = tf.squeeze(tf.nn.conv1d(tf.expand_dims(z, 0), cw, 1, 'SAME'), [0])
    tf.histogram_summary('{}_skip'.format(layer_index), skip)
    output = skip + x
    tf.histogram_summary('{}_output'.format(layer_index), output)
    # combined and output shapes are [width, dense_channels]
    return (output, skip)

# Returns a tuple of (output, non-softmaxed-logits output)
# The non-softmaxed output is used for the loss calculation.
# The shape of x is [width, quantization_channels]
# The shape of output is [width, quantization_channels]
# Dilations is an array of [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, ..., 512]
def layers(x, parameters, noise):
    dilations = parameters['dilations']
    quantization_channels = parameters['quantization_channels']
    dense_channels = parameters['dense_channels']
    width = tf.shape(x)[0]

    co_dense = tf.Variable(tf.random_uniform([1, quantization_channels, dense_channels],
                                             -xavier_halfrange(quantization_channels), xavier_halfrange(quantization_channels)),
            dtype=tf.float32, name='dense_w')
    
    next_input = tf.squeeze(tf.nn.conv1d(tf.expand_dims(x, 0), co_dense, 1, 'SAME'), [0]) # , use_cudnn_on_gpu=False not supported...
    skip_connections = []
    for (i, dilation) in enumerate(dilations):
        with tf.name_scope('layer_{}'.format(i)):
            print "Creating layer {}".format(i)
            (output, skip) = gated_unit(next_input, dilation, parameters, i, noise)
            # output and skip shapes are [width, dense_channels]
            next_input = output
            skip_connections.append(skip)
            sys.stdout.flush()
    skips_tensor = tf.nn.relu(tf.pack(skip_connections, 2))

    co1 = tf.Variable(tf.random_uniform([1, 1, len(dilations), 1],
                                        -xavier_halfrange_rect(len(dilations)), xavier_halfrange_rect(len(dilations))),
            dtype=tf.float32, name='co1')
    
    weighted_skips = tf.squeeze(tf.nn.conv2d(tf.expand_dims(skips_tensor, 0), co1, [1, 1, 1, 1], padding = 'SAME'), [0, 3])

    # weighted_skips shape is [width, dense_channels]
    relu1 = tf.nn.relu(weighted_skips)
    
    co2 = tf.Variable(tf.random_uniform([1, dense_channels, 256], -xavier_halfrange(dense_channels), xavier_halfrange(dense_channels)),
            dtype=tf.float32, name='co2')
    
    raw_output = tf.squeeze(tf.nn.conv1d(tf.expand_dims(relu1, 0), co2, 1, 'SAME'), [0])
    # raw_output shape is [width, quantization_channels]
    
    # Too large tensor for the softmax crashes my GPU in CUDA_ERROR_ILLEGAL_ADDRESS error,
    # so we can do it one by one instead of just this:
    output = tf.nn.softmax(raw_output)
    # Note that the GPU seems to be exactly as fast as CPU.
    #sm_outputs = []
    #stride = 128
    #for i in range(width // stride):
    #    length = min(stride, width - i * stride)
    #    sm_outputs.append(tf.nn.softmax(tf.slice(raw_output, [i * stride, 0], [length, quantization_channels])))
    #output = tf.concat(0, sm_outputs)
    return (output, raw_output)

def create(parameters):
    quantization_channels = parameters['quantization_channels']
    training_length = parameters['training_length']
    input = tf.placeholder(tf.float32, name='input')
    target_output = tf.placeholder(tf.float32, name='target_output')
    schedule_step = tf.placeholder(tf.float32, name='schedule_step')
    noise = tf.placeholder(tf.float32, name="noise")
    input_noise = tf.placeholder(tf.float32, name="input_noise")
    mu_lawd = mu_law(input, float(quantization_channels - 1), input_noise)
    mu_law_x = tf.one_hot(mu_lawd, quantization_channels)
    
    classes_y = mu_law(target_output, quantization_channels - 1, 0)
    (output, raw_output) = layers(mu_law_x, parameters, noise)
    
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(raw_output, classes_y, name='cost')
    
    tvars = tf.trainable_variables()
    gradients = tf.gradients(cost, tvars)
    grads, _ = tf.clip_by_global_norm(gradients, parameters['clip_gradients'])
    optimizer = tf.train.AdamOptimizer(learning_rate = parameters['learning_rate'])

    train_op = optimizer.apply_gradients(zip(grads, tvars))
    tf.add_check_numerics_ops()

    model = {
        'output': output,
        'optimizer': train_op,
        'input': input,
        'target_output': target_output,
        'cost': cost,
        'schedule_step': schedule_step,
        'input_noise': input_noise,
        'noise': noise
    }
    return model

def create_generative_model(parameters):
    quantization_channels = parameters['quantization_channels']
    input = tf.placeholder(tf.float32, name='input')
    mu_law_input = tf.one_hot(mu_law(input, float(quantization_channels - 1), 0), quantization_channels)
    
    (full_generated_output, _) = layers(mu_law_input, parameters, 0)
    # Generated output is only the last predicted distribution
    generated_output = tf.squeeze(tf.slice(full_generated_output, [tf.shape(full_generated_output)[0] - 1, 0], [1, -1]), [0])

    model = {
        'generated_output': generated_output,
        'input': input
    }
    return model
