import numpy as np
from elephant import asset

L = 100
N = 7
D = 3

u = np.arange(L * D, dtype=np.float32).reshape((-1, D))
u /= np.max(u)

P_total = asset._jsf_uniform_orderstat_3d(u=u, n=N)
