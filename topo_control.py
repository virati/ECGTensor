#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:45:51 2019

@author: virati
Topology control project
"""

import numpy as np
import scipy.signal as sig
import networkx as nx
import mayavi
from mayavi.mlab import *

from scipy.ndimage.filters import gaussian_filter#, quiver3d

#setup our base topology
#x = np.linspace(-10,10,100)
#y = np.linspace(-10,10,100)

#x,y = np.meshgrid(x,y)
x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]

np.random.seed(12346788)

base_w = [0.5,0.5]
swig = gaussian_filter(np.random.normal(0.0,20.0,size=x.shape),sigma=10.0)

beta = 0.2
z = beta*np.sin((base_w[0]+swig)*x) + (1-beta)*np.sin((base_w[-1]+swig)*y)
#%%
s = surf(x,y,z,colormap='gist_earth')#color=(1.0,1.0,1.0))

dz = -np.array(np.gradient(z))
z_offset = 0.2
s = quiver3d(x,y,z+z_offset,dz[0],dz[1],0*dz[1],line_width=3)
