#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:14:00 2019

@author: virati
"""

from mayavi import mlab
from numpy import array, cos, sin, cos

x_coord = array([0.0, 1.0, 0.0, -1.0])
y_coord = array([1.0, 0.0, -1.0, 0.0])
z_coord = array([0.2, -0.2, 0.2, -0.2])

plt = mlab.points3d(x_coord, y_coord, z_coord)

msplt = plt.mlab_source
@mlab.animate(delay=100)
def anim():
    angle = 0.0
    while True:
        x_coord = array([sin(angle), cos(angle), -sin(angle), -cos(angle)])
        y_coord = array([cos(angle), -sin(angle), -cos(angle), sin(angle)])
        msplt.set(x=x_coord, y=y_coord)
        yield
        angle += 0.1

#anim()
#mlab.show()

import numpy as np
from mayavi.mlab import *

def h(x,y,beta=0.5):
    base_w = [2,2]
    swig = 0#gaussian_filter(np.random.normal(0.0,1.0,size=x.shape),sigma=10.0)

    return beta*np.sin((base_w[0]+swig)*x) + (1-beta)*np.sin((base_w[-1]+swig)*y)


import numpy as np
from scipy.spatial import Delaunay
from mayavi import mlab

X2 = np.array([0, 0, 1, 1])
Y2 = np.array([0.5, 0.45, 1, 0.5])
Z2 = np.array([0, 1, 0.5,0])

n=100
x = 7 * (np.random.random(n) - 0.5)
y = 7 * (np.random.random(n) - 0.5)
z = h(x, y)


# use scipy for delaunay:
p2d = np.vstack([x,y]).T
d2d = Delaunay(p2d)

fig = mlab.figure(1, bgcolor=(1, 0.7, 1), fgcolor=(0.5, 0.5, 0.5))

# Generate triangular Mesh:
tmesh = mlab.triangular_mesh(x, y, z, d2d.vertices,
                             scalars=y, colormap='jet')

# Simple plot.
mlab.outline(extent=(0,1,0,1,0,1))
mlab.axes(extent=(0,1,0,1,0,1))
mlab.show()

def test_triangular_mesh():
    """An example of a cone, ie a non-regular mesh defined by its
        triangles.
    """
    n = 100
    t = np.linspace(-np.pi, np.pi, n)
    z = np.exp(1j * t)
    x = z.real.copy()
    y = z.imag.copy()
    z = np.zeros_like(x)

    triangles = [(0, i, i + 1) for i in range(1, n)]
    np.random.seed(12345)
    #n = 100
    x = 7 * (np.random.random(n) - 0.5)
    y = 7 * (np.random.random(n) - 0.5)
    
    z = h(x, y)

    #x = np.r_[0, x]
    #y = np.r_[0, y]
    #z = np.r_[1, z]
    t = np.r_[0, t]

    return triangular_mesh(x, y, z, triangles, scalars=t)