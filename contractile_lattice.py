#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 12:44:04 2019

@author: virati
Base class for a simple grid and contractility propogation.
"""

import networkx as nx
import numpy as np
from mayavi.mlab import *
from mayavi import mlab
from scipy.spatial import Delaunay
    
class c_sync:
    def __init__(self,num_nodes=10):
        self.network = nx.triangular_lattice_graph(num_nodes,num_nodes)
        self.get_pos()
        self.L = nx.laplacian_matrix(self.network).todense()
        
    def plot_graph(self):
        mlab.figure(1, bgcolor=(0, 0, 0))
        mlab.clf()
        
        self.viz_nodes = mlab.points3d(self.pos[:,0], self.pos[:,1], self.pos[:,2],
                            self.colors,
                            scale_factor=0.1,
                            scale_mode='none',
                            colormap='Blues',
                            resolution=20)
        
        self.viz_nodes.mlab_source.dataset.lines = np.array(self.G.edges())
        tube = mlab.pipeline.tube(self.viz_nodes, tube_radius=0.01)
        self.viz_edges = mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8),opacity=0.1)
        
        #mlab.savefig('mayavi2_spring.png')
        #mlab.show() # interactive window

    def get_pos(self):
        self.G=nx.convert_node_labels_to_integers(self.network)
        # 3d spring layout
        network_pos=nx.spring_layout(self.G,dim=3)
        
        # numpy array of x,y,z positions in sorted node order
        self.pos = np.array([network_pos[v] for v in sorted(self.G)])
        #self.pos = np.hstack((self.pos,np.zeros((self.pos.shape[0],1))))
        
        # scalar colors
        self.colors=np.array(self.G.nodes())+5
        self.scalars = np.zeros_like(self.G.nodes(),dtype='float64').reshape(-1,1)
        
    def update_pos(self):
        self.pos[10,1] +=0.01
        #self.pos += np.random.normal(0.0,0.01,size=self.pos.shape)
        
    def run(self):
        init_condit = np.zeros_like(self.scalars)
        init_condit[2] = 1.0
        
        self.update_manifold()
        
        self.scalars += init_condit
        
        #mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
        # Initialization of the triangular mesh
        self.tmesh = mlab.triangular_mesh(self.pos[:,0], self.pos[:,1], self.pos[:,2], self.d2d.vertices,
                             scalars=self.pos[:,2], colormap='jet')
        
        @mlab.animate(delay=10)
        def anim():
            mplt = self.tmesh.mlab_source
            for ii in np.arange(100):
                self.update_pos()
                self.update_manifold()
                self.blit_manifold()
                self.blit_network()
                
                yield
                
        anim()
        #mlab.show() # This seems to freeze the visualization after we're done
        
    def update_manifold(self):
        p2d = np.vstack(self.pos[:,0:2])
        self.d2d = Delaunay(p2d)
                
    def blit_network(self):
        self.viz_nodes.mlab_source.set(x=self.pos[:,0],y=self.pos[:,1],z=self.pos[:,2])
        
    def blit_manifold(self):
        self.tmesh.mlab_source.set(x=self.pos[:,0],y=self.pos[:,1],z=self.pos[:,2],scalars=self.pos[:,2],triangles=self.d2d.vertices)

heart = c_sync()
heart.plot_graph()
heart.run()