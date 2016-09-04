#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import pylab

import numpy as np
import scipy.io as sio

import random
import json
import itertools

# Saves .mat files for Octave
def save(file_name, variable_name, value):
    sio.savemat(file_name, {variable_name:value})
