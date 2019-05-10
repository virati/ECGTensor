#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:26:09 2019

@author: virati
"""

import numpy as np
import mayavi
import scipy
import scipy.signal as sig
import scipy.io

from collections import defaultdict

import pdb
import matplotlib.pyplot as plt

import wfdb
from scipy import interpolate
from scipy.interpolate import interp1d

import mayavi.mlab
from mayavi.mlab import quiver3d, plot3d

import numpy as np
from mayavi import mlab

from sklearn.decomposition import PCA

class ecg_tensor:
    def __init__(self,ch=9):
        #No previous MI
        self.ecg = wfdb.rdrecord('focus_data/007c')
        #Previous MI:
        #self.ecg = wfdb.rdrecord('focus_data/013c')
        self.ch=ch
        
        #self.detrend_ecg()
        self.vect_estimate = defaultdict()
        
    def plot_raw(self,filtered=False):
        
        ## Plot the raw
        if not filtered:
            plt.plot(self.ecg.p_signal)
        else:
            
            for ii in range(self.ch):
                plt.plot(self.filt_sig[ii,:])


    def calculate_augmenteds(self):
        self.augs = {'aVF':[],'aVR':[],'aVL':[]}
        self.augs['aVF'] = self.filt_sig[1,:] - 0.5 * self.filt_sig[0,:]
        self.augs['aVR'] = -0.5*(self.filt_sig[0,:] + self.filt_sig[1,:])
        self.augs['aVL'] = self.filt_sig[0,:] - 0.5*self.filt_sig[1,:]
        
        
        aug_matrix = np.array((self.augs['aVF'],self.augs['aVL'],self.augs['aVR']))
        #pdb.set_trace()
        self.filt_sig = np.vstack((self.filt_sig,aug_matrix))

    def detrend_ecg(self):
        filt_sig = []
        for ii in range(self.ch):
            filt_sig.append(sig.detrend(self.ecg.p_signal[:,ii]))
        self.filt_sig = np.array(filt_sig)
        
        self.calculate_augmenteds()
        
    def resample(self):
        vec = sig.decimate(self.filt_sig,q=10).T
        ti = np.linspace(0,vec.shape[0]/1000,vec.shape[0])
        self.ds_ecg = (ti,vec)

    def plot_cycle(self,ch=1):
        plt.figure()
        plt.plot(self.ds_ecg[1][0:100,:])
    
    def interpolated_states(self,c1=0,c2=1,c3=2):
        vec = sig.decimate(self.filt_sig,q=10).T
        x_orig = (sig.detrend(vec[:,c1],type='linear'))
        y_orig = (sig.detrend(vec[:,c2],type='linear'))
        z_orig = (sig.detrend(vec[:,c3],type='linear'))
        
        #Concatenate for the cubic splines
        x = np.concatenate((x_orig[-3:-1],x_orig,x_orig[1:3]))
        y = np.concatenate((y_orig[-3:-1],y_orig,y_orig[1:3]))
        z = np.concatenate((z_orig[-3:-1],z_orig,z_orig[1:3]))
        
        t = np.arange(len(x))
        orig_len = len(vec[:,0])
        
        ti = np.linspace(2,orig_len+1, 10 * orig_len)
        
        xi = interp1d(t,x,kind='cubic')(ti)
        yi = interp1d(t,y,kind='cubic')(ti)
        zi = interp1d(t,z,kind='cubic')(ti)
        
        return x,xi,y,yi,z,zi
    
    def plot_phase(self,c1=0,c2=1,c3=2):
        x,xi,y,yi,_,_ = self.interpolated_states(c1=c1,c2=c2,c3=c3)
        
        tstart = 0
        tend=400
        
        #Maybe add a peak-threshold finder and label all the q waves?
        
        plt.figure()
        plt.subplot(211)
        plt.scatter(x[tstart:tend],y[tstart:tend],alpha=0.2,cmap='jet')
        plt.scatter(x[tstart],y[tstart],marker='o',s=400,color='blue')
        plt.scatter(x[tend],y[tend],marker='X',s=400,color='red')
        plt.plot(xi[tstart*10:tend*10],yi[tstart*10:tend*10],alpha=0.3)
        
        plt.subplot(212)
        plt.plot(x[tstart:tend])
        plt.plot(y[tstart:tend])
        plt.xlim(0,500)
    
    def plot_3d_phase(self):
        x,xi,y,yi,z,zi = self.interpolated_states(c1=0,c2=1,c3=2)
        
        tstart = 0
        tend=200
        
        plot3d(xi[0:1000],yi[0:1000],zi[0:1000],line_width=0.1)
        quiver3d(1,0,0)
        quiver3d(0,1,0)
        quiver3d(0,0,1)
        
    def animate_signals(self):
        #vec = self.filt_sig.T
        #if you want to do the raw data with interpolations
        #_,x,_,y,_,z = self.interpolated_states()
        
        #if you want the PCAd version
        x = self.vect_estimate['PCA'][:,0]
        y = self.vect_estimate['PCA'][:,1]
        z = self.vect_estimate['PCA'][:,2]
        
        s = quiver3d(x[0],y[0],z[0],line_width=10)
        
        
        @mlab.animate(delay=10)
        def anim():
            for ii in np.arange(1000):
                s.mlab_source.u = x[ii]
                s.mlab_source.v = y[ii]
                s.mlab_source.w = z[ii]
                yield
        anim()
        mlab.show()
    
    def vect_find(self):
        pca = PCA(n_components=3)
        #_,x,_,y,_,z = self.interpolated_states()
        
        #data = np.vstack((x,y,z))
        
        data = self.filt_sig.T
        self.pca_data = data
        self.vect_estimate['PCA'] = pca.fit_transform(data)

ecg_data = ecg_tensor()
ecg_data.detrend_ecg()
ecg_data.resample()
ecg_data.vect_find()

#%%
ecg_data.plot_3d_phase()
#%%
ecg_data.plot_cycle()
ecg_data.plot_phase(c1=1,c2=-2,c3=-3)
#%%
ecg_data.animate_signals()
        
#%%
# take the PCA way to learn a rereferencing

#%%
# Try to do our true vector inference on three channels
# This requires us to know the basis vectors
e_tilda = np.array([[1,0,0],[0.5,-0.5,0.1],[-0.5,-0.5,0.1]])
e_pinv = np.linalg.pinv(np.dot(e_tilda.T,e_tilda)) * e_tilda.T
x_hat = np.zeros_like(vec.T)
for ii in range(8250):
    imed = np.dot(e_pinv,vec[ii,:].reshape(-1,1))
    x_hat[:,ii] = imed.squeeze()
        
#%%
#


animate_signals(filt_sig)