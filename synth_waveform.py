#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:01:28 2019

@author: virati
Synthetic waveform generator
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import mayavi

t = np.linspace(-10,10,1000)
phi = np.pi/3
x = np.random.normal(5,0.1,t.shape) * sig.gausspulse(t,fc=10,tpr=30) # np.random.normal(5,0.1,t.shape) * np.sin(2 * np.pi * 10 * t)
y = np.random.normal(5,0.1,t.shape) * sig.gausspulse(t,fc=10) #np.random.normal(5,0.1,t.shape) * np.sin(2 * np.pi * 10 * t + phi)

x = np.random.normal(5,0.5,t.shape) * np.sinc(2 * np.pi * 10 * t)
y = np.random.normal(5,0.5,t.shape) * np.sinc(2 * np.pi * 10 * t + phi)

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,x)
plt.subplot(2,1,2)
plt.plot(t,y)

plt.figure()
plt.plot(t,x)
plt.plot(t,y)

plt.figure()
plt.scatter(x,y)
plt.plot(x,x)


#mayavi time
