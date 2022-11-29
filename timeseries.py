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
    
class model():
    """
    collection of agents
    """
    def __init__(self, function, predfunction, a = [-0.5,-0.5,-0.5], r0 = [100,25,50], sigma = 0.02, angle = 0):
        """
        
        function : function for calculating next position
        a : steepness of tanh functions
        r0 : turnover point in tanh function for different zones
        sigma : noise factor
        
        """
        self.function = function
        self.steepness = a
        self.r0 = r0
        self.sigma = sigma
        self.predfunction = predfunction
        self.angle = angle #angle for pred diverging
                            #preys
    def initialize_agents(self, n_agents, mus, mus_pred = [30,20] , positions = 0, angles = 0, velocity = 10, r_max = 20, mode = "lead_follow",
                          leader_type = "single", leader_mus = [0,10,0], ratio = 0.5, function = "linear", mu_min = 0,
                          mu_max = 10, break_rank = 3, steepness = -0.5, alpha = 1, s_0 = 5,
                            #preds
                          n_pred = 0, pred_angles = 0, pred_positions = 0,
                          r_max_pred = 100, s_pred = 13, pred_length = 5, pred_mu = 5, r_pred = [10,20]):
        
        """
        
        n_agents : Number of agents, integer or 2d-vector for lead-follow mode
        mus : [mu_attraction, mu_repulsion, mu_alignment], if all should have different mu values nx3-array
        positions : nx2-array with x,y values, if positions = False, random positions between, 0 and r_max
        angles: nx1-array with theta orientation values, if angles = False, random positions between 0 and 2*np.pi
        velocity : constant velocities of agents, float
        r_max : maximum radius for random generation of starting positions
        mode :  "allsame" - all agents have same mu values
                "lead_follow" - one leader group and one follower group
                        leader_type : "single", "continuous"
                "continuous" - continuous spectrum of all interactions according to function type
                        function : "linear", "sigmoid"
        
        """
        self.n = np.sum(n_agents)
        self.n_pred = n_pred
        self.prey_agents = []
        self.pred_agents = []
        
        #initial positions of prey
        if type(angles) == int:
            theta = np.random.uniform(0,2*np.pi, np.sum(n_agents))
        else:
            theta = angles
        if type(positions) == int:
            rand_r = np.random.uniform(0, r_max, np.sum(n_agents))
            rand_phi = np.random.uniform(0, 2*np.pi, np.sum(n_agents))
            r = rand_r * np.array([np.cos(rand_phi), np.sin(rand_phi)])
            r = np.transpose(r)
        else:
            r = positions
            
        #prey agent creation  
        if mode == "allsame":
            for i in range(self.n):
                self.prey_agents.append(prey_agent(i, r[i], theta[i], velocity, mus, mus_pred, r_pred, alpha = alpha, s_0 = s_0))
        
        if mode == "lead_follow":
            if leader_type == "single":
                counter = 0
                for _ in range(n_agents[0]): #leaders
                    self.prey_agents.append(prey_agent(counter, r[counter], theta[counter], velocity, leader_mus, mus_pred, r_pred, alpha = alpha, s_0 = s_0))
                    counter += 1
                for _ in range(n_agents[1]): #followers
                    self.prey_agents.append(prey_agent(counter, r[counter], theta[counter], velocity, mus, mus_pred, r_pred, alpha = alpha, s_0 = s_0))
                    counter += 1
            elif leader_type == "continuous":
                counter = 0
                for _ in range(n_agents[0]): #leaders
                    lead_mus = [ratio*mus[0], mus[1], ratio*mus[2]]
                    self.prey_agents.append(prey_agent(counter, r[counter], theta[counter], velocity, lead_mus, mus_pred, r_pred, alpha = alpha, s_0 = s_0))
                    counter += 1
                for _ in range(n_agents[1]): #followers
                    self.prey_agents.append(prey_agent(counter, r[counter], theta[counter], velocity, mus, mus_pred, r_pred, alpha = alpha, s_0 = s_0))
                    counter += 1
        
        if mode == "continuous":
            if function == "linear":
                for i in range(self.n):
                    mu_linear = mu_min + i*(mu_max-mu_min)/(self.n-1)
                    print(mu_linear)
                    self.prey_agents.append(prey_agent(i+1, r[i], theta[i], velocity, [mus[0], mus[1], mu_linear], mus_pred, r_pred, alpha = alpha, s_0 = s_0))
            if function == "sigmoid":
                for i in range(self.n):
                    mu_sigmoid = mu_min + mu_max*0.5*(np.tanh(steepness*(i-break_rank))+1)
                    print(mu_sigmoid)
                    self.prey_agents.append(prey_agent(i+1, r[i], theta[i], velocity, [mus[0], mus[1], mu_sigmoid], mus_pred, r_pred, alpha = alpha, s_0 = s_0))
        
        #initial positions of predators
        if type(pred_angles) == int:
            theta = np.random.uniform(0,2*np.pi, n_pred)
        else:
            theta = pred_angles
        if type(pred_positions) == int:
            rand_r = np.random.uniform(0, r_max_pred, n_pred)
            r = rand_r * np.array([np.cos(theta), np.sin(theta)])
            r = np.transpose(r)
        else:
            r = pred_positions        
        for i in range(n_pred):
            self.pred_agents.append(pred_agent(i, r[i], theta[i], s_pred, pred_length, pred_mu))
             
    def update(self, t_step):
        """

        updates all agents by 1 timestep

        """
        new_preds = self.predfunction(self.prey_agents, self.pred_agents, t_step, angle = self.angle)
        self.pred_agents = new_preds
        new_agents = self.function(self.prey_agents, self.pred_agents, t_step, a = self.steepness, r = self.r0, sigma = self.sigma)
        self.prey_agents = new_agents

    def create_timeseries(self, t_range = 20, t_step = 0.1, mode="binary", mva_interval = 2, fwd_range = 1, sub = 1):
        """
        
        t_range : time range
        t_step : time step
        mode :  "binary" - right 0, left 1
                "binary_mva" - as above, moving average over timesteps -> filters out noise
                "tertiary" - right 0, forward 1, left 2 (if rotationvalue < X, it is considered forward)
                "tertiary_mva" - as above, moving average over timesteps -> filters out noise
                "n_states" - values are cut up into n chunks and discretized accordingly
                "continous" - continuos values for dphi
        mva_interval : how many timesteps for mva calculation
        fwd_range : threshhold in which it is considered to be a forward motion
        n_states : n states for n_states mode
        
        """
        time = int(t_range/t_step)
        timeseries_phi = np.zeros((self.n, time))
        timeseries_dphi = np.zeros((self.n, time))
        timeseries_x = np.zeros((self.n, time))
        timeseries_y = np.zeros((self.n, time))
        timeseries_Q = np.zeros((self.n, time))
        timeseries_v = np.zeros((self.n, time))
        timeseries_dv = np.zeros((self.n, time))
        
        timeseries_pred_phi = np.zeros((self.n, time))
        timeseries_pred_dphi = np.zeros((self.n, time))
        timeseries_pred_xhead = np.zeros((self.n, time))
        timeseries_pred_yhead = np.zeros((self.n, time))
        timeseries_pred_xtail = np.zeros((self.n, time))
        timeseries_pred_ytail = np.zeros((self.n, time))
        for i in range(time):
            r_com = np.array([0,0])
            v_com = np.array([0,0])
            for j in range(self.n):
                timeseries_phi[j][i] = self.prey_agents[j].angle
                timeseries_x[j][i] = self.prey_agents[j].position[0]
                timeseries_y[j][i] = self.prey_agents[j].position[1]
                timeseries_v[j][i] = self.prey_agents[j].s
                r_com = r_com + np.array([self.prey_agents[j].position[0], self.prey_agents[j].position[1]])/self.n
                v_com = v_com + np.array([np.cos(self.prey_agents[j].angle), np.sin(self.prey_agents[j].angle)])/self.n
            
            for j in range(self.n):
                r_comj = np.array([self.prey_agents[j].position[0],self.prey_agents[j].position[1]]) - r_com
                timeseries_Q[j][i] = np.dot(v_com, r_comj)/(np.linalg.norm(v_com)*np.linalg.norm(r_comj))
            
            #predator
            for j in range(self.n_pred):
                timeseries_pred_phi[j][i] = self.pred_agents[j].angle
                timeseries_pred_xhead[j][i] = self.pred_agents[j].head_position[0]
                timeseries_pred_yhead[j][i] = self.pred_agents[j].head_position[1]
                timeseries_pred_xtail[j][i] = self.pred_agents[j].tail_position[0]
                timeseries_pred_ytail[j][i] = self.pred_agents[j].tail_position[1]
                
            self.update(t_step) 
        
        #assignment of dphi and ds values
        if mode == "binary":
            #preys
            for j in range(self.n):
                phi = timeseries_phi[j]
                for i in range(len(phi)-sub):
                    if phi[i] > phi[i+sub]:
                        timeseries_dphi[j][i] = 0
                    else:
                        timeseries_dphi[j][i] = 1
                v = timeseries_v[j]
                for i in range(len(v)-sub):
                    if v[i] > v[i+sub]:
                        timeseries_dv[j][i] = 0
                    else:
                        timeseries_dv[j][i] = 1
                
            #preds
            for j in range(self.n_pred):
                phi = timeseries_phi[j]
                timeseries_pred_dphi[j] = helper.get_dphi_series(phi, sub)
                
        elif mode == "tertiary":
            for j in range(self.n):
                phi = timeseries_pred_phi[j][::sub]
                for i in range(len(phi)-1):
                    if (phi[i] - phi[i+1]) > fwd_range:
                        timeseries_dphi[j][i] = 0
                    elif (phi[i] - phi[i+1]) <  (fwd_range*-1):
                        timeseries_dphi[j][i] = 2
                    else:
                        timeseries_dphi[j][i] = 1
                    
        elif mode == "continuous":
            for j in range(self.n):
                timeseries_dphi[j][i] = (timeseries_phi[j][i] - self.agents[j].angle)/t_step
                    
        if mode == "binary_mva":
            for j in range(self.n):
               phi = helper.mov_avg(timeseries_phi[j][::sub], interval_size = mva_interval)
               for i in range(len(phi)-1):
                   if phi[i] > phi[i+1]:
                       timeseries_dphi[j][i] = 0
                   else:
                       timeseries_dphi[j][i] = 1
         
                        
                        
                        
        df = pd.DataFrame()
        for i in range(self.n):
            df["x"+str(i+1)] = timeseries_x[i][:-sub]
            df["y"+str(i+1)] = timeseries_y[i][:-sub]
            df["phi"+str(i+1)] = timeseries_phi[i][:-sub]
            df["dphi"+str(i+1)] = timeseries_dphi[i][:-sub]
            df["Q" + str(i+1)] = timeseries_Q[i][:-sub]
            df["v"+str(i+1)] = timeseries_v[i][:-sub]
            df["dv"+str(i+1)] = timeseries_dv[i][:-sub]
        for i in range(self.n_pred):
            df["head_x"+str(i+1)] = timeseries_pred_xhead[i][:-sub]
            df["head_y"+str(i+1)] = timeseries_pred_yhead[i][:-sub]
            df["tail_x"+str(i+1)] = timeseries_pred_xtail[i][:-sub]
            df["tail_y"+str(i+1)] = timeseries_pred_ytail[i][:-sub]
            df["pred_phi"+str(i+1)] = timeseries_pred_phi[i][:-sub]
            df["pred_dphi"+str(i+1)] = timeseries_pred_dphi[i][:-sub]
        ts = timeseries(df, n_agents = self.n, n_pred = self.n_pred, mode = mode)
      
        return ts
        
class prey_agent():
    """
    moving agents
    """
    def __init__(self,index, r, angle, s, mus, mu_pred = [6,5], r_pred = [10,20], s_0 = 5, alpha = 1):
        self.position = r # [x,y]
        self.angle = angle 
        self.s = s
        self.mus = mus
        self.index = index
        self.mu_pred = mu_pred
        self.r_head= r_pred[0]
        self.r_tail = r_pred[1]
        self.s_0 = s_0
        self.alpha = alpha 
        
    def update(self, r_t, phi_t):
        self.position = r_t
        self.angle = phi_t

class pred_agent():
    def __init__(self, index, r_head, angle, s, length, mu, s_0 = 16, alpha = 0.01):
        self.head_position = r_head
        self.angle = angle
        self.length = length
        self.tail_position = np.array([self.head_position[0] - self.length*np.cos(self.angle),
                              self.head_position[1] - self.length*np.sin(self.angle)])
        self.s = s
        self.index = index
        self.mu = mu
        self.s_0 = s_0
        self.alpha = alpha
    
    def update(self, r_t, phi_t):
        self.head_position = r_t
        self.angle = phi_t
        self.tail_position = np.array([self.head_position[0] - self.length*np.cos(self.angle),
                              self.head_position[1] - self.length*np.sin(self.angle)])
    
