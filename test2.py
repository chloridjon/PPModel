import model as m
import numpy as np
from importlib import reload
m = reload(m)
import matplotlib.pyplot as plt
import time

n_prey = 10
n_pred = 1
r_max = 5*n_prey
pred_start = 4*r_max

x = [np.random.uniform(-r_max,r_max,n_prey)]
y = [np.random.uniform(-r_max,r_max,n_prey)] 
r_prey = np.concatenate([x,y], axis = 0)
r_prey = np.transpose(r_prey, axes = (1,0))
r_pred = np.array([-pred_start,0])
#other initials
phis = np.zeros(n_prey)
s = np.full(n_prey, 8)
mus = [0.5,5,1]

M = m.model()
M.add_agents(n = n_prey, r = r_prey, phi = phis, s = s, interaction_con = "range", mu_con = mus, interaction_pred = "all")
M.add_agents(type = "pred", r = r_pred, s = 16)
M.live_simulation(100, sub = 10)