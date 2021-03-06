#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:26:09 2019

@author: virati
"""

from ecg_tensor import ecg_tensor

ecg_data = ecg_tensor()
ecg_data.filt_ecg()
ecg_data.resample()
ecg_data.vect_find()

#%%

#%%
ecg_data.plot_cycle()
ecg_data.plot_phase(c1=0,c2=1,c3=2)
#%%
ecg_data.plot_3d_phase()
ecg_data.animate_signals(reduced=False)
