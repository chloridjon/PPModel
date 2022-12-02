#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:26:55 2022

@author: root
"""

from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import normal
import pyinform as pin
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import curve_fit
import entropy_estimators3 as ee
import math
import helper
from importlib import reload
helper = reload(helper)
import matplotlib.style as mplstyle



class timeseries():
    """
    gets created by the model
    """
    def __init__(self, dataframe, n_agents, n_pred, mode = "binary"):
        self.values = dataframe
        self.disc = mode
        self.n_agents = n_agents
        self.n_pred = n_pred

    def extract_preys(self, sub = 1):
        x = np.zeros((self.n_agents, len(self.values.index[::sub])))
        y = np.zeros((self.n_agents, len(self.values.index[::sub])))
        u = np.zeros((self.n_agents, len(self.values.index[::sub])))
        v = np.zeros((self.n_agents, len(self.values.index[::sub])))
        count = 0

        for column in self.values:
            if column[0] == "x" and not column[0:2] == "xp":
                    x[count] = np.array(self.values.iloc[::sub][column])
            elif column[0] == "y" and not column[0:2] == "yp":
                    y[count] = np.array(self.values.iloc[::sub][column])
            elif column[0:3] == "phi" and not column[0:4] == "phip":
                    u[count] = np.cos(np.array(self.values.iloc[::sub][column]))
                    v[count] = np.sin(np.array(self.values.iloc[::sub][column]))
                    count += 1
        
        preys = np.concatenate(([x],[y],[u],[v]))
        preys = np.transpose(preys, axes = (1,0,2))
        return preys

    def extract_predators(self, sub = 1):
        x_head = np.zeros((self.n_pred, len(self.values.index[::sub])))
        y_head = np.zeros((self.n_pred, len(self.values.index[::sub])))
        x_tail = np.zeros((self.n_pred, len(self.values.index[::sub])))
        y_tail = np.zeros((self.n_pred, len(self.values.index[::sub])))
        u_pred = np.zeros((self.n_pred, len(self.values.index[::sub])))
        v_pred = np.zeros((self.n_pred, len(self.values.index[::sub])))
        count = 0

        for column in self.values:
            if column[0:2] == "xp" and column[-4:] == "head":
                    x_head[count] = np.array(self.values.iloc[::sub][column])
            elif column[0:2] == "yp" and column[-4:] == "head":
                    y_head[count] = np.array(self.values.iloc[::sub][column])
            elif column[0:2] == "xp" and column[-4:] == "tail":
                    x_tail[count] = np.array(self.values.iloc[::sub][column])
            elif column[0:2] == "yp" and column[-4:] == "tail":
                    y_tail[count] = np.array(self.values.iloc[::sub][column])
            elif column[0:4] == "phip":
                    u_pred[count] = np.cos(np.array(self.values.iloc[::sub][column]))
                    v_pred[count] = np.sin(np.array(self.values.iloc[::sub][column]))
                    count += 1
        
        preds = np.concatenate(([x_head],[y_head], [x_tail], [y_tail], [u_pred],[v_pred]))
        preds = np.transpose(preds, axes = (1,0,2))
        
        return preds

    def fast_animation(self, path = "Animations/fastanimation.mp4", sub = 10):
        mplstyle.use('fast')
        fig, ax = plt.subplots()

        preys = self.extract_preys(sub = sub)
        preds = self.extract_predators(sub = sub)
        
        x_min = np.array(preys[:,0,:]).min() - 10
        x_max = np.array(preys[:,0,:]).max() + 10
        y_min = np.array(preys[:,1,:]).min() - 10
        y_max = np.array(preys[:,1,:]).max() + 10

        def update(num):
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
            ax.quiver(preys[:,0,num], preys[:,1,num], preys[:,2,num], preys[:,3,num])
            ax.quiver(preds[:,0, num], preds[:,1, num], preds[:,4, num], preds[:,5, num], color = "red")
            ax.plot(preds[:,2, num], preds[:,3, num], "ro")
                
        Q_ani = animation.FuncAnimation(fig, update, interval = 1, frames = len(preys[0][0]))
        plt.show()  

        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=int(np.around(60/sub)))
        Q_ani.save(path, writer=writer, dpi=75)


    def animate(self, path = "Animations/sloweranimation.mp4"):
        fig = plt.figure()
        ax = plt.axes()

        #iterate over every prey
        x = np.zeros((self.n_agents, len(self.values.index)))
        y = np.zeros((self.n_agents, len(self.values.index)))
        u = np.zeros((self.n_agents, len(self.values.index)))
        v = np.zeros((self.n_agents, len(self.values.index)))
        count = 0
        for column in self.values:
            if column[0] == "x":
                    x[count] = np.array(self.values.iloc[:][column])
            elif column[0] == "y":
                    y[count] = np.array(self.values.iloc[:][column])
            elif column[0:3] == "phi":
                    u[count] = np.cos(np.array(self.values.iloc[:][column]))
                    v[count] = np.sin(np.array(self.values.iloc[:][column]))
                    count += 1
        
        preys = []
        for i in range(len(x)):
            preys.append(np.array([x[i],y[i],u[i],v[i]]))
        
        
        #iterate over every predator
        x_head = np.zeros((self.n_pred, len(self.values.index)))
        y_head = np.zeros((self.n_pred, len(self.values.index)))
        x_tail = np.zeros((self.n_pred, len(self.values.index)))
        y_tail = np.zeros((self.n_pred, len(self.values.index)))
        u_pred = np.zeros((self.n_pred, len(self.values.index)))
        v_pred = np.zeros((self.n_pred, len(self.values.index)))
        count = 0
        for column in self.values:
            if column[0:6] == "head_x":
                    x_head[count] = np.array(self.values.iloc[:][column])
            elif column[0:6] == "head_y":
                    y_head[count] = np.array(self.values.iloc[:][column])
            elif column[0:6] == "tail_x":
                    x_tail[count] = np.array(self.values.iloc[:][column])
            elif column[0:6] == "tail_y":
                    y_tail[count] = np.array(self.values.iloc[:][column])
            elif column[0:8] == "pred_phi":
                    u_pred[count] = np.cos(np.array(self.values.iloc[:][column]))
                    v_pred[count] = np.sin(np.array(self.values.iloc[:][column]))
                    count += 1
        
        preds = []
        for i in range(len(x_head)):
            preds.append(np.array([x_head[i],y_head[i],x_tail[i],y_tail[i],u_pred[i],v_pred[i]]))

        def update(num):
            ax.clear()
            ax.set_aspect("equal")
            for agent in preys:
                ax.quiver(agent[0, num], agent[1, num], agent[2, num], agent[3, num])
            
            for agent in preds:
                ax.quiver(agent[0, num], agent[1, num], agent[4, num], agent[5, num], color = "red")
                ax.plot(agent[2, num], agent[3, num], "ro")
                
        Q_ani = animation.FuncAnimation(fig, update, interval = 10, frames = len(x[0]))
        plt.show()  

        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='animation of predator and prey')
        writer = FFMpegWriter(fps=60, metadata=metadata)
        Q_ani.save(path, writer=writer,dpi=100)
    
    def subsample_dphi(self, sub = 3):
        dphi = []
        for column in self.values:
            if column[0:3] == "phi":
                phi = self.values.iloc[:][column]
                dphi = helper.get_dphi_series(phi, sub)
            if column[0:4] == "dphi":
                self.values[column] = dphi
                
                
        
    def transfer_entropy(self, mode = "global", history_length = 2):
        dphi = []
        for column in self.values:
            if column[0:4] == "dphi":
                dphi.append(np.array(self.values.iloc[:][column]))
        n_fish = len(dphi)
        time = len(dphi[0]) 
        
        if mode == "global":
            TE = np.zeros((n_fish, n_fish))
            if self.disc != "continuos":
                for i in range(n_fish):
                    for j in range(n_fish):
                        if i != j:
                            TE[i][j] = pin.transfer_entropy(dphi[i], dphi[j], k=history_length)
            elif self.disc == "continous":
                for i in range(n_fish):
                    for j in range(n_fish):
                        if i != j:
                            k = int(history_length/2)
                            x = dphi[i][k:]
                            y = dphi[j][:-k]
                            z = np.zeros(len(x))
                            z[1:] = x[:-1]
                            x=np.reshape(x,(-1,1))
                            y=np.reshape(y,(-1,1))
                            z=np.reshape(z,(-1,1))
                            TE[i][j] = ee.cmi(x,y,z)
                            
        elif mode == "local":
            TE = np.zeros((n_fish, n_fish, time - history_length))
            if self.disc != "continuos":
                for i in range(n_fish):
                    for j in range(n_fish):
                        if i != j:
                            TE[i][j] = pin.transfer_entropy(dphi[i], dphi[j], k=history_length, local=True)[0]
            elif self.disc == "continuos":
                print("Not possible with continuous values (now)")
                
        return TE
    
    def active_information(self, mode = "global", sub = 1, history_length = 2):
        dphi = []
        for column in self.values:
            if column[0:4] == "dphi":
                dphi.append(np.array(self.values.iloc[:][column]))
        n_fish = len(dphi)
        time = len(dphi[0]) 
        
        if mode == "global":
            AIS = np.zeros(n_fish)
            if self.disc != "continuous":
                for i in range(n_fish):
                    AIS[i] = pin.active_info(dphi[i][::sub], k=history_length) 
            elif self.disc == "continuous":
                for i in range(n_fish):
                    k = int(history_length/2)
                    x = dphi[i][1:]
                    z = np.zeros(len(x))
                    z[1:] = x[:-1]
                    x=np.reshape(x,(-1,1))
                    z=np.reshape(z,(-1,1))
                    AIS[i] = ee.mi(x,z)
                    
        elif mode == "local":
            AIS = np.zeros((n_fish, time - history_length))
            if self.disc != "continuos":
                for i in range(n_fish):
                    AIS[i] = pin.active_info(dphi[i], k=history_length, local=True)[0]
            else:
                print("Not possible with continuous values (now)")
        return AIS

    def polarization(self):
        time = len(self.values)
        P_ges = 0
        for i in range(time):
            current = self.values.iloc[[i]]
            ag_counter = 0
            avg_v = 0
            count = 0
            for column in current:
                if column[0:3] == "phi":
                    v = np.array([np.cos(current[column]), np.sin(current[column])])
                    avg_v += v
                    ag_counter += 1
                count+=1
            avg_v = np.linalg.norm(avg_v)
            P = avg_v/ag_counter
            P_ges += P/time
        
        return P_ges
    
    def average_position(self):
        time = len(self.values)
        Q_ges = np.zeros(self.n_agents)
        for i in range(time):
            current = self.values.iloc[[i]]
            ag_count = 0
            for column in current:
                if column[0:1] == "Q":
                    Q_ges[ag_count] += current[column]
                    ag_count += 1
                    
        Q_ges = Q_ges/time
        return Q_ges
    
    def distance_to_predator(self):
        time = len(self.values)
        D = np.zeros((time, self.n_pred, self.n_agents))
        for i in range(time):
            current = self.values.iloc[[i]]
            x = np.zeros(self.n_agents)
            y = np.zeros(self.n_agents)
            head_x = np.zeros(self.n_pred)
            head_y = np.zeros(self.n_pred)
            ag_count = 0
            pred_count = 0
            for column in current:
                if column[0:1] == "x":
                    x[ag_count] = current[column]
                elif column[0:1] == "y":
                    y[ag_count] = current[column]
                    ag_count += 1
                elif column[0:6] == "head_x":
                    head_x[pred_count] = current[column]
                elif column[0:6] == "head_y":
                    head_y[pred_count] = current[column]
                    pred_count += 1
            for j in range(self.n_pred):
                x = x - head_x[j]
                y = y - head_y[j]
                D[i][j] = np.sqrt(x**2 + y**2)
        
        return D
                    
    def average_distance(self):
        time = len(self.values)
        D_ges = np.zeros((self.n_agents, self.n_agents))
        for i in range(time):
            current = self.values.iloc[[i]]
            x_count = 0
            y_count = 0
            x_c = np.zeros(self.n_agents)
            y_c = np.zeros(self.n_agents)
            for column in current:
                if column[0:1] == "x":
                    x_c[x_count] += current[column]
                    x_count += 1
                elif column[0:1] == "y":
                    y_c[y_count] += current[column]
                    y_count += 1
            for i in range(len(x_c)):
                for j in range(len(y_c)):
                    D_ges[i][j] += np.sqrt((x_c[i]-x_c[j])**2 + (y_c[i]-y_c[j])**2)
                    
        D_ges = D_ges/time
        return D_ges
    
    def simple_interaction_matrix(self, interaction_threshold = 50):
        D = self.average_distance()
        int_matrix = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if D[i][j] < 50:
                    int_matrix[i][j] = 1
        return int_matrix
   
    def more_transfer_entropy(self, mode = "avg_neighbors", history_length = 2, neighbor_threshold = 50, sub = 1):
        """
        mode : avg_neighbors, n_neighbors, background
        
        """
    
        if mode == "avg_neighbors":
            TE = np.zeros(self.n_agents)
            phi = []
            for column in self.values:
                if column[0:3] == "phi":
                    phi.append(np.array(self.values.iloc[:][column]))
            new_phi = []
            D = self.average_distance()
            for i in range(len(phi)):
                for j in range(len(phi)):
                    if 0 < D[i][j] < neighbor_threshold:
                        new_phi.append(phi[j])
                if len(new_phi) > 0:
                    avg_phi = helper.get_avg_series(new_phi)
                    dphi = helper.get_dphi_series(phi[i], sub)
                    dphi_avg = helper.get_dphi_series(avg_phi, sub)
                    TE[i] = pin.transfer_entropy(dphi_avg, dphi, k=history_length)
        if mode == "background":
            TE = np.zeros((self.n_agents, self.n_agents))
            dphi = []
            for column in self.values:
                if column[0:4] == "dphi":
                    dphi.append(np.array(self.values.iloc[:][column])) 
            D = self.average_distance()        
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i != j:
                        del_list = [i,j]
                        for k in range(self.n_agents):
                            if D[i][k] > neighbor_threshold and k !=i and k != j:
                                del_list.append(k)
                        background = np.array(dphi)
                        ws = np.delete(background, del_list, 0)
                        if len(ws) > 1:
                            TE[i][j] = pin.transfer_entropy(dphi[i], dphi[j], k=history_length, condition = ws)
                        else:
                            TE[i][j] = pin.transfer_entropy(dphi[i], dphi[j], k=history_length)
                        
        return TE 
    