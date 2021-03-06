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

def mu_law(x, mu):
    ml = tf.sign(x) * tf.log(mu * tf.abs(x) + 1.0) / tf.log(mu + 1.0)
    # Scaling between -128 and 128 integers.
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
    width = tf.shape(value)[0]
    dilation_channels = tf.shape(filters)[2]
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

    #return atr_conv_1d
    return tf.zeros([width, dilation_channels])

# Returns a tuple of output to the next layer and skip output.
# The shape of x is [width, dense_channels]
def gated_unit(x, dilation, parameters, layer_index):
    #tf.histogram_summary('{}_x'.format(layer_index), x)
    
    filter_width = parameters['filter_width']
    dense_channels = parameters['dense_channels']
    dilation_channels = parameters['dilation_channels']
    quantization_channels = parameters['quantization_channels']

    w1 = tf.Variable(tf.random_normal([filter_width, dense_channels, dilation_channels], stddev=0.05),
            dtype=tf.float32, name='w1')
    w2 = tf.Variable(tf.random_normal([filter_width, dense_channels, dilation_channels], stddev=0.05),
            dtype=tf.float32, name='w2')
    cw = tf.Variable(tf.random_normal([1, dilation_channels, dense_channels], mean=1.0, stddev=0.05),
            dtype=tf.float32, name='cw')

    #tf.histogram_summary('{}_w1'.format(layer_index), w1)
    #tf.histogram_summary('{}_w2'.format(layer_index), w2)
    #tf.histogram_summary('{}_cw'.format(layer_index), cw)
    
    with tf.name_scope('causal_atrous_convolution'):
        dilated1 = causal_atrous_conv1d(x, w1, dilation, 'SAME')
        dilated2 = causal_atrous_conv1d(x, w2, dilation, 'SAME')
    with tf.name_scope('gated_unit'):
        z = tf.multiply(tf.tanh(dilated1), tf.sigmoid(dilated2))
    # dilated1, dilated2, z shapes are [width, dilation_channels]
    skip = tf.squeeze(tf.nn.conv1d(tf.expand_dims(z, 0), cw, 1, 'SAME'), [0])
    #tf.histogram_summary('{}_skip'.format(layer_index), skip)
    output = skip + x
    #tf.histogram_summary('{}_output'.format(layer_index), output)
    # combined and output shapes are [width, dense_channels]
    return (output, skip)

# Returns a tuple of (output, non-softmaxed-logits output)
# The non-softmaxed output is used for the loss calculation.
# The shape of x is [width, quantization_channels]
# The shape of output is [width, quantization_channels]
# Dilations is an array of [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, ..., 512]
def layers(x, parameters):
    dilations = parameters['dilations']
    quantization_channels = parameters['quantization_channels']
    dense_channels = parameters['dense_channels']
    width = parameters['sample_length']

    co_dense = tf.Variable(tf.random_normal([1, quantization_channels, dense_channels], mean=1.0, stddev=0.05),
            dtype=tf.float32, name='dense_w')
    
    next_input = tf.squeeze(tf.nn.conv1d(tf.expand_dims(x, 0), co_dense, 1, 'SAME'), [0]) # , use_cudnn_on_gpu=False not supported...
    skip_connections = []
    for (i, dilation) in enumerate(dilations):
        with tf.name_scope('layer_{}'.format(i)):
            print "Creating layer {}".format(i)
            #(output, skip) = gated_unit(next_input, dilation, parameters, i)
            output = tf.zeros([width, dense_channels])
            skip = tf.zeros([width, dense_channels])
            # output and skip shapes are [width, dense_channels]
            next_input = output
            skip_connections.append(skip)
            sys.stdout.flush()
    #skips_tensor = tf.nn.relu(tf.pack(skip_connections, 2))

    #co1 = tf.Variable(tf.random_normal([1, 1, len(dilations), 1], mean=1.0, stddev=0.05),
    #        dtype=tf.float32, name='co1')
    
    #weighted_skips = tf.squeeze(tf.nn.conv2d(tf.expand_dims(skips_tensor, 0), co1, [1, 1, 1, 1], padding = 'SAME'), [0, 3])
    weighted_skips = tf.zeros([width, dense_channels])

    # weighted_skips shape is [width, dense_channels]
    #relu1 = tf.nn.relu(weighted_skips)
    
    #co2 = tf.Variable(tf.random_normal([1, dense_channels, 256], mean=1.0, stddev=0.05),
    #        dtype=tf.float32, name='co2')
    
    #raw_output = tf.squeeze(tf.nn.conv1d(tf.expand_dims(relu1, 0), co2, 1, 'SAME'), [0])
    raw_output = tf.zeros([width, quantization_channels])
    # raw_output shape is [width, quantization_channels]
    #output = tf.nn.softmax(raw_output)
    sm_outputs = []
    for i in range(width):
        sm_outputs.append(tf.nn.softmax(tf.slice(raw_output, [i, 0], [1, -1])))
    output = tf.pack(sm_outputs, 0) 
    #output = tf.zeros([width, quantization_channels])
    return (output, raw_output)

def create(parameters):
    quantization_channels = parameters['quantization_channels']
    sample_length = parameters['sample_length']
    input = tf.placeholder(tf.float32, shape=(sample_length), name='input')
    y = input
    x = tf.pad(tf.slice(input, [0], [tf.shape(input)[0] - 1]), [[1, 0]])
    width = tf.shape(x)[0]
    # x is shifted right by one and padded by zero.
    mu_lawd = mu_law(x, float(quantization_channels - 1))
    shifted_mu_law_x = tf.one_hot(mu_lawd, quantization_channels)
    
    classes_y = mu_law(y, quantization_channels - 1)
    (output, raw_output) = layers(shifted_mu_law_x, parameters)
    #output = tf.zeros([width,quantization_channels])
    #raw_output = tf.zeros([width,quantization_channels])
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(raw_output, classes_y, name='cost')
    
    tvars = tf.trainable_variables()
    #gradients = tf.gradients(cost, tvars)
    # grads, _ = tf.clip_by_global_norm(gradients, parameters['clip_gradients'])
    optimizer = tf.train.AdamOptimizer(learning_rate = parameters['learning_rate'])

    #train_op = optimizer.apply_gradients(zip(gradients, tvars))
    train_op = x
    tf.add_check_numerics_ops()

    model = {
        'output': output,
        'optimizer': train_op,
        'x': input,
        'cost': cost
    }
    return model

def create_generative_model(parameters):
    quantization_channels = parameters['quantization_channels']
    input = tf.placeholder(tf.float32, name='input')
    mu_law_input = tf.one_hot(mu_law(input, float(quantization_channels - 1)), quantization_channels)
    
    (full_generated_output, _) = layers(mu_law_input, parameters)
    # Generated output is only the last predicted distribution
    generated_output = tf.squeeze(tf.slice(full_generated_output, [tf.shape(full_generated_output)[0] - 1, 0], [1, -1]), [0])

    model = {
        'generated_output': generated_output,
        'x': input
    }
    return model
