# Run a chi2 scan for a given set of parameters, using the forecast data.
import numpy as np
import sys
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.lya_theory import Theory
import matplotlib.pyplot as plt
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import h5py as h5
import cupix
import pandas as pd
from itertools import combinations
import os
import multiprocessing as mp
import psutil

cupixpath = cupix.__path__[0].rsplit('/', 1)[0]

# --------- fixed settings ---------------
savepath  = "/pscratch/sd/m/mlokken/desi-lya/px/"
# forecast_file = f"{cupixpath}/data/px_measurements/forecast/forecast_ffcentral_real_binned_out_px-zbins_4-thetabins_9_w_res_noiseless_z0.hdf5"
forecast_file = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
# "central" uses the central simulation in the training set
# "random" randomly selects points within the minimum and maximum range of each training parameter
add_noise = False
test = False
grid_density = 16# 32

# ---------- command-line inputs ---------------
param_mode = sys.argv[1].lower() # "igm" or "arinyo"
param_range = sys.argv[2].lower() # use the full parameter space of the Gadget simulations ('broad') or small user-defined ranges ('tight')?
if param_mode == 'Arinyo':
   param_mode = 'arinyo'
pars_to_test = []
for par in sys.argv[3:]:
    pars_to_test.append(par)   
# -------------------------- --------------------------
print("Testing", pars_to_test)
assert param_range in ['tight','broad'], "Parameter range argument not supported"
if test:
    ncores_available = 1
else:
    ncores_available = len(os.sched_getaffinity(0))
    print(f'ncore_available={ncores_available}')
    mp.set_start_method('spawn')
igm_pars = ['Delta2_p', 'n_p', 'mF', 'gamma', 'T0', 'kF_Mpc']
arinyo_pars = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']
# set cosmo+IGM pars and Arinyo pars
if param_mode == "igm":
    pars = igm_pars
elif param_mode == "arinyo":
    pars = arinyo_pars



##### Get the truth from forecast file ######
forecast = DESI_DR2(forecast_file, kM_max_cut_AA=1.5, km_max_cut_AA=1.8)
# get default theory that was used for forecast
with h5.File(forecast_file, 'r') as f:
    default_theory_label = f['metadata'].attrs['true_lya_theory']
print(f"Default theory label: {default_theory_label}")
zs = forecast.z
iz_choice = 3
z_choice  = zs[iz_choice]
cosmo = {}
truth_params = {}
with h5.File(forecast_file) as f:
    for cosmo_par in f['cosmo_params'].attrs.keys():
        cosmo[cosmo_par] = f['cosmo_params'].attrs[cosmo_par]
    if param_mode=='igm':
        for like_par in f['like_params'].attrs.keys():
            truth_params[like_par] = f['like_params'].attrs[like_par]
    elif param_mode=='arinyo':
        for arinyo_par in f['arinyo_pars'].attrs.keys():
            truth_params[arinyo_par] = f['arinyo_pars'].attrs[arinyo_par]
print("Passing forecast cosmology to theory object:", cosmo)
theory = Theory(zs, cosmo_dict=cosmo, default_lya_theory=default_theory_label, emulator_label="forestflow_emu", verbose=False)
# check the parameters
print("Truth params are:")
for p, val in truth_params.items():
    print(p, val)

###############################################
## set the ranges ##
ranges_dict = {"beta":[1.4,1.65], "bias":[-0.122,-0.11], "q1":[0.15,0.45], "mF":[0.8,0.83], "T0":[8000,12000], "gamma":[1.2,1.9]} # can input the ranges manually here
like = Likelihood(forecast, theory, z=z_choice, verbose=False)

ranges = []
if param_range=='broad':
    for par_to_test in pars_to_test:
        # get the min/ max values of parameters from the training simulations
        gadget_short_info_file = cupixpath + '/data/emulator/ff_training_info.csv'
        train_test_info = pd.read_csv(gadget_short_info_file)
        train_test_z = np.where(z_choice==train_test_info["z"])[0][0]
        min_par_val = train_test_info.iloc[train_test_z][f'{par_to_test}_min']
        max_par_val = train_test_info.iloc[train_test_z][f'{par_to_test}_max']
        ranges.append(np.linspace(min_par_val, max_par_val, grid_density))
elif param_range=='tight':
    for par_to_test in pars_to_test:
        print(par_to_test)
        if par_to_test in ranges_dict.keys():
            print(par_to_test, ranges_dict[par_to_test][0], ranges_dict[par_to_test][1])
            ranges.append(np.linspace(ranges_dict[par_to_test][0], ranges_dict[par_to_test][1], grid_density))
        else:
            sys.exit(f"Error: no ranges dictionary specified for {par_to_test}.")
################################################
## Set up grid for chi2 scan
grid = np.meshgrid(*ranges, indexing='ij')
# shape: (npars, N1, N2, ..., Nn)
stacked = np.stack(grid, axis=-1)
# reshape to list of points: (N_total_points, npars)
points = stacked.reshape(-1, len(pars_to_test))
chi2_save = np.ones(len(points))*1000
print("Will evaluate chi2 at ", len(points), "points.")
pars_to_test_iz = [par+f"_{iz_choice}" for par in pars_to_test]

def compute_chi2(p):
    par_dict = dict(zip(pars_to_test_iz, p))
    for par in truth_params:
        if (par not in par_dict) and (not (('T0' in pars_to_test) and ('sigT' in par))): # don't overwrite new T0 with old sigT
            par_dict[par] = truth_params[par] # add back the truth for all non-varied parameters
    chi2 = like.get_chi2(like_params=par_dict)
    return chi2
    
with mp.Pool(ncores_available) as pool:
    chi2_save = pool.map(compute_chi2, points)
chi2_save = np.array(chi2_save)

# reshape chi2 back to grid shape
chi2_grid = chi2_save.reshape(grid[0].shape)
min_chi2 = np.amin(chi2_grid)
fig, ax = plt.subplots()
for par in truth_params:
    if 'bias' in par:
        truth_params[par]*=-1

# plot
if len(pars_to_test)==1:
    x_ = grid[0]
    ax.plot(x_, chi2_grid-min_chi2)
    ax.set_xlabel(pars_to_test[0])
    ax.set_ylabel(r'$\Delta \chi^2$')
    ax.axvline(truth_params[pars_to_test_iz[0]], color='k', linestyle='--', label='Truth')
    ax.legend()
    plt.savefig(f"{savepath}/plots/chi2_scans/chi2_{pars_to_test[0]}_kmax_1p5A")

elif len(pars_to_test)==2:
    x_ = grid[0]
    y_ = grid[1]

    ndof = len(pars_to_test)
    cs = ax.contourf(x_, y_, chi2_grid-min_chi2, levels=[2.30, 6.17, 11.8], cmap=plt.cm.bone)
    ax.set_xlabel(pars_to_test[0])
    ax.set_ylabel(pars_to_test[1])
    cbar = fig.colorbar(cs)
    cbar.set_label(r'$\Delta \chi^2$')
    # plot the truth
    plt.axvline(truth_params[pars_to_test_iz[0]], color='k', linestyle='--')
    plt.axhline(truth_params[pars_to_test_iz[1]], color='k', linestyle='--')
    plt.savefig(f"{savepath}/plots/chi2_scans/chi2_{pars_to_test[0]}_{pars_to_test[1]}")


np.savez(f"{savepath}/data/chi2_scans/chi2_{pars_to_test[0]}_kmax_1p5A", chi2=chi2_grid, forecast_file=forecast_file)


# work-in-progress for higher dimensions

# plot every parameter combination
# pairs = list(combinations(pars_to_test, 2))
# print("Will plot parameter combos", pairs)

# Add the contour line levels to the colorbar

# print("chi2 shape", chi2_grid.shape)
# for pair in pairs:
#     mesh_loc_x = pars_to_test.index(pair[0])
#     mesh_loc_y = pars_to_test.index(pair[1])
#     x_ = grid[mesh_loc_x]
#     y_ = grid[mesh_loc_y]
#     plt.contour(x_, y_, chi2_grid)
