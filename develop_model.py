import gwflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("tkagg")
# todo: get isactive set up in the coeficient matrix. TODO!!!
# #   think about how to recast stuff for internal storage and unstructured
#   grids (no entries for inactive cells)

# todo: look at CHD stuff, this doesn't make sense to me
# todo: GHB is good for now. Maybe think about doing a transient solution?

# todo: Test RIV packages
# todo: Test WEL package
# todo: Test DRN package
# todo: Develop RCH
# todo: Develop EVT

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

elevs = [9, 6]
cheads = []
for c, chd in zip([0, 9], elevs):
    for r in range(nrow):
        cheads.append([0, r, c, chd, 1000])


rch = np.full((nrow, ncol), 4e-02)
rch[:, -1] = 0
irch = np.zeros(rch.shape, dtype=int)

df = pd.DataFrame(cheads, columns=["k", "i", "j", "elev", "cond"])
df = df[df["j"] == 9]
df = df.reset_index(drop=True)

pet = np.full((nrow, ncol), 0., dtype=float)
pet[:, -1] = 0.40
pet_surf = top.copy()
pet_ext = pet_surf - 6

model = gwflow.GroundwaterFlow(modelname="test_model")
dis = gwflow.packages.Discretization(model, nlay, nrow, ncol, delx, dely, top, bottom, isactive)
n = dis.neighbors
vn = dis.vertical_neighbors
hyd = gwflow.packages.Hydraulics(model, hk, vk)
ic = gwflow.packages.InitialConditions(model, shead)
ghb = gwflow.packages.GeneralHead(model, df)
# evt = gwflow.packages.Evapotranspiration(model, pet, pet_surf, pet_ext)
rch = gwflow.packages.Recharge(model, rch, irch)

ssor = gwflow.solvers.SorSolver(model, mxiter=100, relax=1.6)
#xe = np.sum(evt.rhs)
#xr = np.sum(rch.rhs)
# x = ghb.rhs

h = model.solve()

h = h.reshape(model.shape)
vmin = np.min(h)
vmax = np.max(h)
fig, axs = plt.subplots(ncols=2)
for i in range(2):
    pc = axs[i].imshow(h[i], vmin=vmin, vmax=vmax)

plt.colorbar(pc)
plt.show()
# plt.savefig("test.png")