import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from numba import jit
from numpy.random import normal

def mov_avg(array, interval_size = 50):
    sma = pd.Series(array).rolling(interval_size).sum()/interval_size
    return np.array(sma)

@jit
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

@jit
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

@jit    
def get_avg_series(phi_series):
    avg_series = np.zeros(len(phi_series[0]))
    n = len(phi_series)
    for series in phi_series:
        avg_series += np.array(series)/n
    
    return avg_series

@jit
def get_dphi_series(phi, sub):
    dphi = np.zeros(len(phi))
    for i in range(len(dphi)-sub):
        if phi[i] > phi[i+sub]:
            dphi[i] = 1
            
    return dphi

@jit(nopython = True, fastmath = True)
def sigmoid(alpha, d, r):
    S = 0.5 * (np.tanh(alpha*(d - r)) + 1)
    return S

@jit(nopython = True, fastmath = True)
def length(v):
    L = np.linalg.norm(v)
    return L

def interaction_indices_con(agent, agents, type = "nnn", N_nearest = 5, ran = 50, voronoi_object = 0):
    """
    agent : agent for which interactions are determinded
    agents : all agents of the model
    type : type of interaction selection
    (all, nnn (n-nearest-neighbors), range, voronoi)

    """
    n = len(agents) - 1
    ag_indices = np.array([i for i in range(n+1) if agents[i].index != agent.index])
    for i in range(len(agents)):
        if agent.index == agents[i].index:
            ag_index = i

    if type == "all":
        indices = ag_indices
    elif type == "nnn":
        if N_nearest < n+1:
            r_i = agent.position
            D = np.zeros(n)
            for i in range(n):
                r = np.abs(r_i - agents[i].position)
                D[i] = 0.96*max(r) + 0.4*min(r) #faster approximation of distance (4% Error)
            Idx = np.argsort(D)[:N_nearest]
            indices = [ag_indices[i] for i in Idx]
        else:
            indices = ag_indices

    elif type == "range":
        r_i = agent.position
        D = np.zeros(n)
        counter = 0
        for i in range(n):
            r = np.abs(r_i - agents[i].position)
            D[i] = 0.96*max(r) + 0.4*min(r) #faster approximation of distance (4% Error)
            if D[i] > ran:
                D[i] = 0
                counter += 1
        Idx = np.argsort(D)[counter:]
        indices = [ag_indices[i] for i in Idx]
    elif type == "voronoi":
        Idx = []
        if voronoi_object == 0:
            P = np.zeros((n+1, 2))
            for i in range(n+1):
                P[i] = np.array(agents[i].position)
            V = Voronoi(P)
            agent_vertices = V.regions[V.point_region[ag_index]]
            for i in range(n):
                if i != ag_index:
                    vertices_i = V.regions[V.point_region[i]]
                    if not set(agent_vertices).isdisjoint(vertices_i):
                        Idx.append(i)
        else:
            V = voronoi_object
            agent_vertices = V.regions[V.point_region[ag_index]]
            for i in range(n):
                if i != ag_index:
                    vertices_i = V.regions[V.point_region[i]]
                    if not set(agent_vertices).isdisjoint(vertices_i):
                        Idx.append(i)
        indices = [ag_indices[i] for i in Idx]

    return indices

def interaction_indices_pred(agent, preys, preds, type = "nnn", N_nearest = 5, ran = 50):
    """
    agent : agent for which interactions are determinded
    agents : all agents of the model
    type : type of interaction selection
    (all, nnn (n-nearest-neighbors), range, voronoi)

    """
    n = len(preds)
    pred_indices = np.array([i for i in range(n)])

    if type == "all":
        indices = pred_indices
    elif type == "range":
        r_i = agent.position
        D = np.zeros(n)
        D_tail = np.zeros(n)
        counter = 0
        for i in range(n):
            r_head = np.abs(r_i - preds[i].position)
            r_tail = np.abs(r_i - preds[i].tail_position)
            D[i] = 0.96*max(r_head) + 0.4*min(r_head) #faster approximation of distance (4% Error)
            D_tail[i] = 0.96*max(r_tail) + 0.4*min(r_tail)
            if D[i] > 50 and D_tail[i] > 70:
                D[i] = 0
                counter += 1
        Idx = np.argsort(D)[counter:]
        indices = [pred_indices[i] for i in Idx]

    return indices

def preyprey_force(agent, preys, interaction_type = "voronoi", ran = 70, voronoi_object = 0):
    """
    forces of prey on prey
    """
    #with which conspecifics does the agent interact
    interactions = interaction_indices_con(agent, preys, type = interaction_type, ran = ran, voronoi_object=voronoi_object)
    
    #set up forces
    F_att = np.array([0.,0.])
    F_rep = np.array([0.,0.])
    F_alg = np.array([0.,0.])

    for j in interactions:
        #vectors between agents
        r_ij = preys[j].position - agent.position
        v_ij = np.array([np.cos(preys[j].phi), np.sin(preys[j].phi)]) - np.array([np.cos(agent.phi), np.sin(agent.phi)])
        dis = length(r_ij)

        #calc forces
        F_att += agent.mu_con[0] * sigmoid(agent.a_con[0], dis, agent.r_con[0]) * r_ij/dis
        F_rep -= agent.mu_con[1] * sigmoid(agent.a_con[1], dis, agent.r_con[1]) * r_ij/dis
        F_alg += agent.mu_con[2] * sigmoid(agent.a_con[2], dis, agent.r_con[2]) * v_ij
    
    F_i = F_att + F_rep + F_alg

    return F_i

def predprey_force(agent, preys, preds, interaction_type = "range"):
    """
    forces of pred on prey
    """
    interactions = interaction_indices_pred(agent, preys,  preds, type = interaction_type)
    F_att = np.array([0.,0.])
    F_rep = np.array([0.,0.])

    if len(interactions) > 0:
        agent.mark_for_interaction(True)
        for j in interactions:
            #vectors between head and tail and agent
            r_head = preds[j].position - agent.position
            r_tail = preds[j].tail_position - agent.position
            dis_head = length(r_head)
            dis_tail = length(r_tail)

            #calcforces
            F_rep -= agent.mu_pred[0] * sigmoid(agent.a_pred[0], dis_head, agent.r_pred[0]) * r_head/dis_head
            F_att += agent.mu_pred[1] * sigmoid(agent.a_pred[1], dis_tail, agent.r_pred[1]) * r_tail/dis_tail
    else:
        agent.mark_for_interaction(False)

    F_i = F_att + F_rep
    return F_i


def preypred_force(agent, preys, angle = 0):
    r_p = agent.position
    r_com = np.array([0,0])
    d_closest = 100000000
    for ag in preys:
        r_i = ag.position
        r_com = r_com + r_i/len(preys)
        r_ip = r_p - r_i
        d_ip = np.linalg.norm(r_ip)
        if d_ip < d_closest:
            d_closest = d_ip
    r_comp = r_com - r_p
    d_com = np.linalg.norm(r_comp)

    if d_closest > 70:
        #calc of social force
        F_att = agent.mu_prey * (r_comp/d_com)

    else:
        r = np.array([r_comp[0]*np.cos(angle) - r_comp[1]*np.sin(angle), r_comp[0]*np.sin(angle) + r_comp[1]*np.cos(angle)])
        F_att = agent.mu_prey * (r/d_com)
    
    return F_att

def predpred_force(agent, preds):
    return np.array([0,0])

def move_prey(agent,t_step, force):
    v_i = np.array([np.cos(agent.phi), np.sin(agent.phi)])
    u_i = np.array([-np.sin(agent.phi), np.cos(agent.phi)])
    F_phi = np.dot(force, u_i)
    F_vel = np.dot(force, v_i)
    dphi = t_step * F_phi
    phi_between = angle_between(force, v_i)
    if np.absolute(dphi) > np.absolute(phi_between):
        dphi = phi_between
    ds = t_step * F_vel + agent.alpha*(agent.s0 - agent.s)

    new_phi = agent.phi + dphi + agent.sigma*np.sqrt(t_step)*normal()
    new_s = agent.s + ds + agent.sigma*np.sqrt(t_step)*normal()
    new_r = agent.position + t_step * new_s * np.array([np.cos(new_phi),np.sin(new_phi)])
    return new_r, new_phi, new_s

def move_pred(agent, t_step, force):
    u_i = np.array([-np.sin(agent.phi), np.cos(agent.phi)])
    F_phi = np.dot(force, u_i)
    dphi = t_step * F_phi

    new_phi = agent.phi + dphi + agent.sigma*np.sqrt(t_step)*normal()
    new_s = agent.s
    new_r = agent.position + t_step * new_s * np.array([np.cos(new_phi),np.sin(new_phi)])
    return new_r, new_phi, new_s






            





        
            