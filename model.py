from numba import jit
import numpy as np
from numpy.random import normal
import pyinform as pin
import pandas as pd
import helper
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib.animation as animation
reload(helper)
import matplotlib.style as mplstyle
import calculations as calc
reload(calc)
import timeseries as ts
reload(ts)
import multiprocessing as mp
import concurrent.futures as cf
from scipy.spatial import Voronoi

class model():
    """
    
    """
    def __init__(self):
        self.agents = []
        self.prey = []
        self.pred = []
        self.n_pred = 0
        self.n_prey = 0
    
    def list_preys(self):
        self.update_agents()
        preys = pd.DataFrame()
        preys["index"] = [ag.index for ag in self.prey]
        display(preys)

    def list_preds(self):
        self.update_agents()
        preds = pd.DataFrame()
        preds["index"] = [ag.index for ag in self.pred]
        display(preds)

    def add_agents(self, n = 1, type = "prey",
                   r = [0,0], phi = 0, s = 8,
                   # parameters for preys
                   r_max = 100, phi_int = [0,2*np.pi], #random position
                   mu_con = [0.02,3,0.5], mu_pred = [10,5], #interaction
                   a_con = -0.5, a_pred = -0.15,
                   r_con = [60,15,30], r_pred = [40,60],
                   alpha = 0.1, s_pref = None,
                   interaction_con = "voronoi", interaction_pred = "all",
                   #parameters for preds
                   length = 10,
                   r_max_pred = 500, phi_int_pred = [0,2*np.pi], #random position
                   mu_con_pred = [0,0,0], mu_prey = 10, #interaction
                   a_con_pred = -0.5, a_prey = -0.15,
                   r_con_pred = [100,25,50], r_prey = 1000,
                   alpha_pred = 0.1, s_pref_pred = None, attack_angle = 0):
        """
        n : number of agents to be added of the type

        type : pred or prey
        
        r : positions (None for random in specified radius, 
        n*R^n array for specified position)

        phi : orientations (None for random orientation in specified interval,
        n-array for specific)
        
        s : (starting) velocities (None for random orientation in specified interval,
        n-array for specific)
        
        """
        if n == 1:
            if type == "prey":
                ind = str(self.n_prey + 1)
                self.agents.append(prey(index = ind, position = r, phi = phi, s = s, interaction_con = interaction_con, interaction_pred = interaction_pred, mu_con = mu_con))
            elif type == "pred":
                ind = "p" + str(self.n_pred + 1)
                self.agents.append(pred(index = ind, position = r, phi = phi, s = s, length = length, attack_angle = attack_angle))
        else:
            for i in range(n):
                if type == "prey":
                    ind = str(self.n_prey + i + 1)
                    self.agents.append(prey(index = ind, position = r[i], phi = phi[i], s = s[i], interaction_con = interaction_con, interaction_pred = interaction_pred, mu_con = mu_con))
                elif type == "pred":
                    ind = "p" + str(self.n_pred + i + 1)
                    self.agents.append(pred(index = ind, position = r[i], phi = phi[i], s = s[i], length = length[i], attack_angle = attack_angle))
        self.update_agents()

    def kill_agents(self, indices):
        """
        delete agents by index
        """
        pass
    
    def update_agents(self):
        """
        updates lists of different agents and numbers
        """
        self.prey = [ag for ag in self.agents if ag.type == "prey"]
        self.pred = [ag for ag in self.agents if ag.type == "pred"]
        self.n_prey = len(self.prey)
        self.n_pred = len(self.pred)
    
    def modify_agents(self, index):
        """
        index : single index or list of indices
        to implement:
        - how to define what it is to be modified
        - modification
        """
        pass
    
    def move_agents(self, t_step):
        """
        moves all agents one timestep forward

        #method 1
        procs = []
        for ag in self.agents:
            p = mp.Process(target = ag.move, args = [self.prey, self.pred, t_step])
            p.start()
            procs.append(p)
        
        for proc in procs:
            proc.join()
        
        
        # method 2
        with cf.ThreadPoolExecutor(max_workers=None) as executor:
            results = [executor.submit(ag.move, self.prey, self.pred, t_step) for ag in self.agents]
        """
        if self.n_prey > 5:
            P = np.array([self.prey[i].position for i in range(self.n_prey)])
            V = Voronoi(P)
        else:
            V = None
        for ag in self.agents:
            ag.move(self.prey, self.pred, t_step, voronoi_object = V)
        self.update_agents()

    def create_timeseries(self, time, t_step = 0.01):
        """
        simulate all timesteps over time
        creates timeseries object
        """
        length = int(time/t_step)
        preys = np.zeros((self.n_prey,length,4)) #x,y,phi,v
        preds = np.zeros((self.n_pred,length,6)) #x_head, y_head, x_tail, y_tail, phi, v
        for t in range(length):
            self.move_agents(t_step)
            #track all prey
            count1 = 0
            for prey in self.prey:
                preys[count1,t] = np.array([prey.position[0], prey.position[1], prey.phi, prey.s])
                count1 += 1
            #track all pred
            count2 = 0
            for pred in self.pred:
                preds[count2,t] = np.array([pred.position[0], pred.position[1], pred.tail_position[0], pred.tail_position[1], pred.phi, pred.s])
                count2 += 1
        #convert data
        data_preys = np.array(np.split(preys, self.n_prey, axis = 0))[:,0]
        if self.n_pred > 0:
            data_preds = np.array(np.split(preds, self.n_pred, axis = 0))[:,0]
            data = np.concatenate((np.concatenate(data_preys, axis = 1),
                                np.concatenate(data_preds, axis = 1)), axis = 1)
        else:
            data = np.concatenate(data_preys, axis = 1)
        #define column names
        names = []
        for prey in self.prey:
            names.append("x" + str(prey.index))
            names.append("y" + str(prey.index))
            names.append("phi" + str(prey.index))
            names.append("v" + str(prey.index))
        for pred in self.pred:
            names.append("x" + str(pred.index) + "_head")
            names.append("y" + str(pred.index) + "_head")
            names.append("x" + str(pred.index) + "_tail")
            names.append("y" + str(pred.index) + "_tail")
            names.append("phi" + str(pred.index))
            names.append("v" + str(pred.index))
        df = pd.DataFrame(data, columns = names)
        return ts.timeseries(df, n_agents = self.n_prey, n_pred = self.n_pred)

    def live_simulation(self, time, t_step = 0.01, sub = 15):
        """
        creates a live animation of a simulation run
        """
        #plt.style.use('fivethirtyeight')
        fig = plt.figure()
        ax = plt.axes()
        mplstyle.use('fast')
        def animate(i):
            plt.cla()
            for prey in self.prey:
                if prey.pred_interaction == True:
                    ax.quiver(prey.position[0], prey.position[1], np.cos(prey.phi), np.sin(prey.phi), color = "blue")
                else:
                    ax.quiver(prey.position[0], prey.position[1], np.cos(prey.phi), np.sin(prey.phi))
            for pred in self.pred:
                ax.plot(pred.tail_position[0], pred.tail_position[1], "ro")
                ax.quiver(pred.position[0], pred.position[1], np.cos(pred.phi), np.sin(pred.phi), color = "red")
            ax.set_aspect("equal")
            for _ in range(sub):
                self.move_agents(t_step)
        ani = animation.FuncAnimation(fig, animate, interval = 1, frames = int(time/(sub*t_step)))
        plt.show()

class agent():
    """
    agent in the model
    """
    def __init__(self, index, position, phi, s):
        self.index = index
        self.position = position
        self.phi = phi
        self.s = s


class prey(agent):
    """
    prey agent
    """
    def __init__(self, index, position, phi, s, alpha = 0.1, s0 = 5, sigma = 0.02,
                 con_function = calc.preyprey_force, mu_con = [0.2,5,1.5], a_con = [-0.15,-0.15,-0.15], r_con = [160,20,40], interaction_con = "voronoi",
                 pred_function = calc.predprey_force, mu_pred = [10,7], a_pred = [-0.15,-0.15], r_pred = [50,70], interaction_pred = "all",):
        super().__init__(index, position, phi, s)
        self.type = "prey"

        #inidvidual parameters #function for calculation of force between preys
        self.alpha = alpha #stubborness
        self.s0 = s0 #prefered speed
        self.sigma = sigma

        #conspecific interaction parameters
        self.con_function = con_function
        self.mu_con = mu_con
        if type(a_con) == "int" or type(a_con) == "float":
            self.a_con = np.full(3, a_con)
        else:
            self.a_con = a_con
        self.r_con = r_con
        self.interaction_type_con = interaction_con 

        #predator interaction parameters
        self.pred_function = pred_function
        self.mu_pred = mu_pred
        if type(a_pred) == "int" or type(a_pred) == "float":
            self.a_pred = np.full(3, a_pred)
        else:
            self.a_pred = a_pred
        self.r_pred = r_pred
        self.interaction_type_pred = interaction_pred
        self.pred_interaction = False 

    def move(self, preys, preds, t_step, voronoi_object = 0):
        """
        move agent one timestep
        """
        force = self.con_function(self, preys, interaction_type = self.interaction_type_con, voronoi_object = voronoi_object) + self.pred_function(self, preys, preds, interaction_type = self.interaction_type_pred)
        new_r, new_phi, new_s = calc.move_prey(self, t_step, force)
        self.phi = new_phi
        self.s = new_s
        self.position = new_r
    
    def mark_for_interaction(self, marking):
        self.pred_interaction = marking

class pred(agent):
    """
    predator agent
    """
    def __init__(self, index, position, phi, s, length, sigma = 0.02,
                 con_function = calc.predpred_force,
                 prey_function = calc.preypred_force, mu_prey = 0.35, a_prey = -0.15, r_prey = 1000, attack_angle = 0):
        super().__init__(index, position, phi, s)

        #individual parameters
        self.type = "pred"
        self.length = length
        self.tail_position = np.array([self.position[0] - self.length*np.cos(self.phi),
                                       self.position[1] - self.length*np.sin(self.phi)])
        self.sigma = sigma
    
        #conspecific interactions
        self.con_function = con_function

        #prey interactions
        self.prey_function = prey_function
        self.mu_prey = mu_prey
        self.a_prey = a_prey
        self.r_prey = r_prey
        self.attack_angle = attack_angle

    def move(self, preys, preds, t_step, voronoi_object = 0):
        force = self.prey_function(self, preys, angle = self.attack_angle) + self.con_function(self, preds)
        new_r, new_phi, new_s = calc.move_pred(self, t_step, force)
        self.phi = new_phi
        self.s = new_s
        self.position = new_r
        self.tail_position = np.array([new_r[0] - self.length*np.cos(new_phi),
                                       new_r[1] - self.length*np.sin(new_phi)])