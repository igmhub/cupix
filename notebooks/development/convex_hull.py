# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from scipy.spatial import ConvexHull
import pandas as pd
from cupix.utils.utils import get_path_repo
import os
from forestflow.archive import GadgetArchive3D
import forestflow
import numpy as np
# %load_ext autoreload 
# %autoreload 2

# %%
igm_pars = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc', 'T0']
arinyo_pars = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']


# Figure out the ForestFlow training central simulation
path_program = get_path_repo("forestflow")
path_program
folder_lya_data = path_program + "/data/best_arinyo/"
print("Loading archive.")
Archive3D = GadgetArchive3D(
    path_program
)
sim_dict_central =  Archive3D.get_testing_data("mpg_central")
training_data = Archive3D.training_data

# %%
gadget_zs = []
for sim_z in sim_dict_central:
    gadget_zs.append(sim_z['z']) # save the redshifts
gadget_zs = np.asarray(gadget_zs)

# %%
gadget_zs

# %%
# input a z
z = 2.2
# find the closest
sim_select_i = np.argmin(np.abs(gadget_zs - z))
sim_select_i

# %%
len(training_data) / len(gadget_zs)

# %%
training_data[0]['mF']

# %%
points = []

nsims = int(len(training_data) / len(gadget_zs))
nz = len(gadget_zs)
for i in range(nsims):
    nsim = int(i * nz + sim_select_i)
    points_thissim = []
    for par in igm_pars:
        points_thissim.append(training_data[nsim][par])
    points.append(points_thissim)
points = np.asarray(points)
print(points.shape)


# %%
ch = ConvexHull(points)

# %%
import matplotlib.pyplot as plt
plt.plot(points[:,0], points[:,1], 'o')
for simplex in ch.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')


# %%
def isInHull(P,hull):
    '''
    Datermine if the list of points P lies inside the hull
    :return: list
    List of boolean where true means that the point is inside the convex hull
    '''
    A = hull.equations[:,0:-1]
    b = np.transpose(np.array([hull.equations[:,-1]]))
    isInHull = np.all((A @ np.transpose(P)) <= np.tile(-b,(1,len(P))),axis=0)
    return isInHull


# %%
P = [[3, 4, 6, 1, 3, 9, 2]]
isInHull(P, ch)

# %%
points[0]

# %%
isInHull([points[30]+.0001], ch)

# %%
points[30].shape

# %%
2 * 3 * 11 * 5

# %%
