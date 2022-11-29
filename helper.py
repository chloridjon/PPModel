#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:45:29 2022

@author: root
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import normal
import pyinform as pin
import pandas as pd


def mov_avg(array, interval_size = 50):
    sma = pd.Series(array).rolling(interval_size).sum()/interval_size
    return np.array(sma)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])
    if angle > np.pi:
        angle = angle - 2*np.pi
    return angle

    
def get_avg_series(phi_series):
    avg_series = np.zeros(len(phi_series[0]))
    n = len(phi_series)
    for series in phi_series:
        avg_series += np.array(series)/n
    
    return avg_series

def get_dphi_series(phi, sub):
    dphi = np.zeros(len(phi))
    for i in range(len(dphi)-sub):
        if phi[i] > phi[i+sub]:
            dphi[i] = 1
            
    return dphi