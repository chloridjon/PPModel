import model as m
import numpy as np
from importlib import reload
m = reload(m)
import matplotlib.pyplot as plt
import time
import calculations as calc


points = np.array([
    (2.5, 2.5), (4, 7.5), (7.5, 2.5), (6, 7.5), (4, 4), (3, 3), (6, 3)
])

V = calc.voronoi_regions(points)
print(calc.voronoi_adj(V, 0))