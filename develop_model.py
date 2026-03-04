import gwflow
import numpy as np
import pandas as pd

# todo: get isactive set up in the coeficient matrix. TODO!!!
#   think about how to recast stuff for internal storage and unstructured
#   grids (no entries for inactive cells)

# todo: look at CHD stuff, this doesn't make sense to me
# todo: GHB is good for now. Maybe think about doing a transient solution?

# todo: Test RIV packages
# todo: Test WEL package
# todo: Develop DRN package

# todo: Look at best practices for iterating over solutions once we start transient

nlay = 2
nrow = 10
ncol = 10
delx = np.full((ncol), 10)
dely = np.full((nrow), 10)
top = np.full((nrow, ncol), 10)
bottom = np.full((nlay, nrow, ncol), 0)
bottom[-1] = -10
isactive = np.ones(bottom.shape, dtype=int)
hk = np.full((2, 10, 10), 2.54)
vk = hk * 0.01
shead = np.full((nlay, nrow, ncol), 10, dtype=float)
# shead[0:5] = 2
# shead[5:10] = 8

chds = [9, 6]
cheads = []
for c, chd in zip([0, 9], chds):
    for r in range(nrow):
        cheads.append([0, r, c, chd, 1000])

df = pd.DataFrame(cheads, columns=["k", "i", "j", "elev", "cond"])

model = gwflow.GroundwaterFlow(modelname="test_model")
dis = gwflow.packages.Discretization(model, nlay, nrow, ncol, delx, dely, top, bottom, isactive)
n = dis.neighbors
vn = dis.vertical_neighbors
hyd = gwflow.packages.Hydraulics(model, hk, vk)
ic = gwflow.packages.InitialConditions(model, shead)
ghb = gwflow.packages.GeneralHead(model, df)
x = ghb.rhs
# chd = gwflow.ConstantHead(model, df)

model.solve(maxiters=5)
