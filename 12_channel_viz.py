#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 19:51:37 2019

@author: virati
ECG viz for 12 channel
"""


import numpy as np
import mayavi
import scipy
import scipy.signal as sig
import scipy.io

import matplotlib.pyplot as plt

import wfdb
from scipy import interpolate
from scipy.interpolate import interp1d

import mayavi.mlab
from mayavi.mlab import quiver3d

import numpy as np
from mayavi import mlab


ecg = wfdb.rdrecord('sample-data/p10143')

## Plot the raw
plt.plot(ecg.p_signal)
#%%
filt_sig = []
for ii in range(3):
    filt_sig.append(sig.detrend(ecg.p_signal[:,ii]))
filt_sig = np.array(filt_sig)
#%%
for ii in range(3):
    plt.plot(filt_sig[ii,:])
#%%
#plt.figure()
#plt.plot(filt_sig[0,:],filt_sig[1,:])

vec = sig.decimate(filt_sig,q=10).T
orig_len = len(vec[:,0])
ti = np.linspace(2,orig_len+1, 10 * orig_len)

for ii in [1]:
    for jj in [2]:
        
        x_orig = (sig.detrend(vec[:,ii],type='linear'))
        y_orig = (sig.detrend(vec[:,jj],type='linear'))
        
        x = np.concatenate((x_orig[-3:-1],x_orig,x_orig[1:3]))
        y = np.concatenate((y_orig[-3:-1],y_orig,y_orig[1:3]))
        t = np.arange(len(x))
        xi = interp1d(t,x,kind='cubic')(ti)
        yi = interp1d(t,y,kind='cubic')(ti)
        
        tstart = 0
        tend=-1
        
        plt.figure()
        plt.subplot(211)
        plt.scatter(x,y,alpha=0.2,cmap='jet')
        plt.scatter(x[tstart],y[tstart],marker='o',s=400,color='blue')
        plt.scatter(x[tend],y[tend],marker='X',s=400,color='red')
        plt.plot(xi,yi,alpha=0.3)
        
        plt.subplot(212)
        plt.plot(x)
        plt.plot(y)
        plt.xlim(0,500)
#%%
# Try to do our true vector inference on three channels
        
e_tilda = np.array([[1,0,0],[0.5,-0.5,0.1],[-0.5,-0.5,0.1]])
e_pinv = np.linalg.pinv(np.dot(e_tilda.T,e_tilda)) * e_tilda.T
x_hat = np.zeros_like(vec.T)
for ii in range(8250):
    imed = np.dot(e_pinv,vec[ii,:].reshape(-1,1))
    x_hat[:,ii] = imed.squeeze()
        
#%%
#

def animate_signals(filt_sig):
    vec = filt_sig.T
    s = quiver3d(vec[0,0],vec[0,1],vec[0,2],line_width=10)
    
    
    @mlab.animate(delay=10)
    def anim():
        for ii in np.arange(1000):
            s.mlab_source.u = vec[ii,0]
            s.mlab_source.v = vec[ii,1]
            s.mlab_source.w = vec[ii,2]
            yield
    anim()
    mlab.show()
animate_signals(filt_sig)