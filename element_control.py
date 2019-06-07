#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:48:32 2019

@author: virati
Per-element topology 
"""

import numpy as np
from mayavi.mlab import *
from scipy.ndimage.filters import gaussian_filter#, quiver3d
from mayavi import mlab
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
    def __init__(self,samples=100):
        np.random.seed(12345)
        self.n = samples
        self.x = 7 * (np.random.random(samples) - 0.5)
        self.y = 7 * (np.random.random(samples) - 0.5)

        self.z = h(self.x, self.y)

    def plot_surf(self):
        mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
        
        # Visualize the points
        self.pts = mlab.points3d(self.x, self.y, self.z, self.z, scale_mode='none', scale_factor=0.1,opacity=0.7,color=(1.0,0.0,0.0))
        
        # Create and visualize the mesh
        self.mesh = mlab.pipeline.delaunay2d(self.pts)
        self.interp_surf = mlab.pipeline.surface(parent=self.mesh,opacity=0.4)
    
        
        #self.interp_surf = mlab.surf(self.mesh)

        @mlab.animate(delay=10)
        def anim():
            mplt = self.pts.mlab_source
            for ii in np.arange(10000):
                
                new_x = (1+ii/1000)*self.x#*np.random.normal(0.0,0.01,size=self.x.shape)
                new_y = (1+ii/1000) * self.y#*np.random.normal(0.0,0.01,size=self.x.shape)
                new_z = (1+ii/1000) * self.z #np.random.normal(0.0,0.01,size=self.x.shape)
                
                #self.pts.mlab_source.x = new_x
                #self.pts.mlab_source.y = new_y
                #self.pts.mlab_source.z = new_z
                mplt.set(x=new_x,y=new_y,z=new_z)
                
                self.mesh = mlab.pipeline.delaunay2d(self.pts)
                self.interp_surf.mlab_source.parent = self.mesh
                
                yield
        anim()
        mlab.show()

heart = heart_surf()
heart.plot_surf()