import model as m
import numpy as np
from importlib import reload
m = reload(m)
import matplotlib.pyplot as plt
import time

N = 50
mu_range = np.linspace(0,2,5)
attack_angles = np.linspace(0,np.pi/18,5)
n_prey = 10
n_pred = 1
P = np.zeros((len(mu_range), len(attack_angles)))

N_run = 0
for _ in range(N):
    N_run+=1
    print("Starts Run " + str(N_run))
    start = time.time()
    mu_count = 0
    for mu in mu_range:
        angle_count = 0
        for angle in attack_angles:
            #random position
            x = [np.random.uniform(-50,50,n_prey)]
            y = [np.random.uniform(-50,50,n_prey)] 
            r_prey = np.concatenate([x,y], axis = 0)
            r_prey = np.transpose(r_prey, axes = (1,0))
            #other initials
            phis = np.zeros(n_prey)
            s = np.full(n_prey, 5)
            mus = [0.2,5, mu]
            r_pred = np.array([-150., 0.])

            M = m.model()
            M.add_agents(n = n_prey, r = r_prey, phi = phis, s = s, interaction_con = "voronoi", mu_con = mus)
            M.add_agents(type = "pred", r = r_pred, s = 12, attack_angle = angle)
            ts = M.create_timeseries(5)
            P[mu_count, angle_count] += ts.polarization()/N
            angle_count += 1
        mu_count += 1
    end = time.time()
    print("Ending Run " +str(N_run) + " at " + str(end-start) + " seconds")

plt.pcolormesh(mu_range, attack_angles*180/np.pi, P)
plt.xlabel("$\mu_{alg}$ (Prey)")
plt.ylabel("atack angle (Predator) [°]")
plt.yticks(attack_angles*180/np.pi)
plt.xticks(mu_range)
plt.colorbar(label = "Polarization")
plt.savefig("polarization_heatmap.pdf")
plt.show()