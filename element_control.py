#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:48:32 2019

@author: virati
Per-element topology 
"""

import numpy as np
from mayavi.mlab import *
from mayavi import mlab

from scipy.ndimage.filters import gaussian_filter#, quiver3d
from scipy.spatial import Delaunay

import time

# Create data with x and y random in the [-2, 2] segment, and z a
# Gaussian function of x and y.

def f(x, y):
    return np.exp(-(x ** 2 + y ** 2)/np.sqrt(0.1))

def g(x,y):
    return np.exp(-((x-1) ** 2 + (y-1) ** 2)/np.sqrt(0.1))

def h(x,y,beta=0.5):
    base_w = [2,2]
    swig = 0#gaussian_filter(np.random.normal(0.0,1.0,size=x.shape),sigma=10.0)

    return beta*np.sin((base_w[0]+swig)*x) + (1-beta)*np.sin((base_w[-1]+swig)*y)

class heart_surf:
    def __init__(self,samples=1000):
        np.random.seed(12345)
        self.n = samples
        self.x = 7 * (np.random.random(samples) - 0.5)
        self.y = 7 * (np.random.random(samples) - 0.5)

        self.z = h(self.x, self.y)

    def update_points(self,frame):
        self.x += np.random.normal(0.0,0.01,size=self.x.shape)
        self.y += np.random.normal(0.0,0.01,size=self.x.shape)#+np.sinc(frame-200)
        self.z = h(self.x,self.y) #np.random.normal(0.0,0.01,size=self.x.shape)
        
        if frame > 100:
            self.x += 0.5*np.sinc(2*np.pi*(frame-200)/2*self.x)
            self.y += 0.5*np.sinc(2*np.pi*(frame-200)/2*self.y)
        
    def plot_surf(self):
        mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
        p2d = np.vstack([self.x,self.y]).T
        d2d = Delaunay(p2d)

        # Visualize the points
        #self.pts = mlab.points3d(self.x, self.y, self.z, self.z, scale_mode='none', scale_factor=0.1,opacity=0.7,color=(1.0,0.0,0.0))
        self.tmesh = mlab.triangular_mesh(self.x, self.y, self.z, d2d.vertices,
                             scalars=self.z, colormap='jet')
        
        @mlab.animate(delay=10)
        def anim():
            mplt = self.tmesh.mlab_source
            for ii in np.arange(1000):
                self.update_points(frame=ii)
                print(ii)
                p2d = np.vstack([self.x,self.y]).T
                d2d = Delaunay(p2d)
                
                mplt.set(x=self.x,y=self.y,z=self.z,triangles=d2d.vertices)
                #self.pts.mlab_source.set(x=self.x,y=self.y,z=self.z)
                
                yield
                
        anim()
        #mlab.show()

heart = heart_surf()
heart.plot_surf()

