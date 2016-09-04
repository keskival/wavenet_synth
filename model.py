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

# The model is a deep self-associative bottleneck network with RELU activations,
# with input and output being the pure waveform. The loss is calculated in a
# convolutional fashion combining two adjanced outputs with triangular mixing.
# Additionally the loss includes a regularization term for the difference of the two
# bottleneck layer activations.

def loss(prediction, y):
    return tf.reduce_mean(tf.nn.l2_loss(tf.sub(prediction, y)))

def generative_module(x, layers, model):
    next_layer_input = x
    layer_activations = []
    min_layer_size = 99999
    bottleneck = None
    min_layer_index = 0
    for (i, layer) in enumerate(layers):
        if (layer < min_layer_size):
            min_layer_size = layer
            min_layer_index = i
    # Only processing the layers after the bottleneck here.
    for i in range(min_layer_index + 1, len(layers)):
        w = model['weights'][i]
        b = model['biases'][i]
        next_layer_input = tf.nn.relu(tf.add(tf.matmul(next_layer_input, w), b))
    return next_layer_input

def module(x, layers, model):
    next_layer_input = x
    layer_activations = []
    min_layer_size = 99999
    bottleneck = None
    for (layer, w, b) in zip(layers, model['weights'], model['biases']):
        next_layer_input = tf.nn.relu(tf.add(tf.matmul(next_layer_input, w), b))
        layer_activations.append(next_layer_input)
        # The bottleneck state is important.
        if layer < min_layer_size:
            min_layer_size = Layer
            bottleneck = next_layer_input
    return (next_layer_input, bottleneck)

def create(parameters):
    print('Creating the neural network model.')
    tf.reset_default_graph()
    # tf Graph input
    # The input consists of two samples, overlapping with half a sample width.
    x = tf.placeholder(tf.float32, shape=(None, parameters['half_sample_length'] * 3), name='training_input')
    # The expected output is the one half-sample at the overlap.
    y = tf.slice(x, [0, parameters['half_sample_length']], [parameters['batch_size'], parameters['half_sample_length']], name='y')
    # The generative input is forcing the activation of the bottleneck layer.
    generative_input = tf.placeholder(tf.float32, shape=(None, parameters['generative_input_size']), name='generative_input')

    weights = [parameters['half_sample_length'] * 2] + parameters['layers']
    layers = parameters['layers'] + [parameters['half_sample_length'] * 2]
    model = {
        'weights': [tf.Variable(tf.random_normal(
            [weight_length, layer_size]), name='weights_' + str(i)) for (i, (weight_length, layer_size)) in enumerate(zip(weights, layers))],
        'biases': [tf.Variable(tf.random_normal(
            [layer_size]), name='biases_' + str(i)) for (i, layer_size) in enumerate(layers)],
        'x': x,
        'y': y,
        'generative_input': generative_input
    }

    # Define loss and optimizer
    batch_size = parameters['batch_size']

    # Collecting the predictions for both sides.
    (left_prediction, left_bottleneck) = module(tf.slice(x, [0, 0], [parameters['batch_size'], parameters['half_sample_length'] * 2]), layers, model)
    left_half = tf.slice(left_prediction, [0, parameters['half_sample_length']], [parameters['batch_size'], parameters['half_sample_length']])

    (right_prediction, right_bottleneck) = module(tf.slice(x, [0, parameters['half_sample_length']], [parameters['batch_size'], parameters['half_sample_length'] * 2]), layers, model)
    right_half = tf.slice(right_prediction, [0, 0], [parameters['batch_size'], parameters['half_sample_length']])

    model['generative_output'] = generative_module(generative_input, layers, model)
    # half_sample_length sized linear slope from 0.0 to 1.0
    right_slope = np.asfarray(range(0, parameters['half_sample_length']), dtype=np.float32) / (parameters['half_sample_length'] - 1)
    left_slope = 1 - right_slope
    prediction = tf.mul(left_slope, left_half) + tf.mul(right_slope, right_half, name='prediction')
    cost = loss(prediction, y) + tf.nn.l2_loss(tf.sub(left_bottleneck, right_bottleneck))

    tvars = tf.trainable_variables()
    gradients = map(tf.to_float, tf.gradients(cost, tvars))
    optimizer = tf.train.AdamOptimizer(learning_rate = parameters['learning_rate'])

    train_op = optimizer.apply_gradients(zip(gradients, tvars))
    
    model['prediction'] = prediction
    model['cost'] = cost
    model['optimizer'] = train_op
    
    return model
