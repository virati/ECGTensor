#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:22:58 2019

@author: virati
Basic display of 3D ECG vector in space, with animations
"""


import numpy as np
import mayavi
import scipy
import scipy.signal as sig
import scipy.io

import matplotlib.pyplot as plt

import wfdb

ecg = wfdb.rdrecord('sample-data/a103l')