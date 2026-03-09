# Run a chi2 scan for a given set of parameters, using the forecast data.
import numpy as np
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.lya_theory import set_theory
import matplotlib.pyplot as plt
from cupix.likelihood.likelihood_parameter import LikelihoodParameter, par_index, dict_from_likeparam
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import h5py as h5
import cupix
import pandas as pd
from itertools import combinations
cupixpath = cupix.__path__[0].rsplit('/', 1)[0]

# --------- settings ---------
forecast = "central" # "central" or "random"
param_mode = "igm" # "igm" or "arinyo"
add_noise = False
Nfree = 2 # number of free parameters to fit in validation
run_minimizer = True # if false, will only generate the forecast
pars_to_test = ['mF', 'T0']
grid_density = 20
# "central" uses the central simulation in the training set
# "random" randomly selects points within the minimum and maximum range of each training parameter
# --------------------------
forecast_file = f"{cupixpath}/data/px_measurements/forecast/forecast_ffcentral_{param_mode}_real_binned_out_px-zbins_4-thetabins_9_w_res_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=1.2)

# get default theory that was used for forecast
with h5.File(forecast_file, 'r') as f:
    default_theory_label = f['metadata'].attrs['true_lya_theory']
print(f"Default theory label: {default_theory_label}")


# Load theory
omnuh2 = 0.0006
mnu = omnuh2 * 93.14
H0 = 67.36
omch2 = 0.12
ombh2 = 0.02237
As = 2.1e-9
ns = 0.9649
nrun = 0.0
w = -1.0
omk = 0
cosmo = {
    'H0': H0,
    'omch2': omch2,
    'ombh2': ombh2,
    'mnu': mnu,
    'omk': omk,
    'As': As,
    'ns': ns,
    'nrun': nrun,
    'w': w
}
zs = forecast.z

theory = set_theory(zs, cosmo_dict=cosmo, default_theory=f'best_fit_{param_mode}_from_p1d', p3d_label='arinyo', emulator_label='forestflow_emu', verbose=True)

iz_choice = 0
z_choice  = zs[iz_choice]

# read the truth params from forecast file
truth_param_dict = {}
with h5.File(forecast_file, "r") as f:
    params = f['like_params']
    attrs = params.attrs
    for key in attrs:
        truth_param_dict[key] = params.attrs[key]
# check the parameters
print("Truth params are:")
for p, val in truth_param_dict.items():
    print(p, val)

like = Likelihood(forecast, theory, z=z_choice, verbose=True)

ranges = []
for par_to_test in pars_to_test:
    # get the min/ max values of parameters from the training simulations
    gadget_short_info_file = cupixpath + '/data/emulator/ff_training_info.csv'
    train_test_info = pd.read_csv(gadget_short_info_file)
    train_test_z = np.where(z_choice==train_test_info["z"])[0][0]
    print("Found min/max values at z=", train_test_info.iloc[train_test_z]['z'])
    min_par_val = train_test_info.iloc[train_test_z][f'{par_to_test}_min']
    max_par_val = train_test_info.iloc[train_test_z][f'{par_to_test}_max']
    print(f"Min/max values of {par_to_test} in training sims at this z: {min_par_val}, {max_par_val}")
    ranges.append(np.linspace(min_par_val, max_par_val, grid_density))


grid = np.meshgrid(*ranges, indexing='ij')
# shape: (npars, N1, N2, ..., Nn)
stacked = np.stack(grid, axis=-1)
print("stacked shape", stacked.shape)
# reshape to list of points: (N_total_points, npars)
points = stacked.reshape(-1, len(pars_to_test))
print("points shape", points.shape)
chi2_save = np.ones(len(points))*1000

print("Will evaluate chi2 at ", len(points), "points.")

for i,p in enumerate(points):
    par_dict = dict(zip(pars_to_test, p))
    chi2 = like.get_chi2(like_params=par_dict)
    chi2_save[i] = chi2

# plot every parameter combination
pairs = list(combinations(pars_to_test, 2))
print("Will plot parameter combos", pairs)

# reshape chi2 back to grid shape
chi2_grid = chi2_save.reshape(grid[0].shape)


x_ = grid[0]
y_ = grid[1]

fig, ax = plt.subplots()
ndof = len(pars_to_test)
min_chi2 = np.amin(chi2_grid)
cs = ax.contourf(x_, y_, chi2_grid-min_chi2, levels=[2.30, 6.17, 11.8], cmap=plt.cm.bone)

plt.xlabel(pars_to_test[0])
plt.ylabel(pars_to_test[1])
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel(r'$\chi^2$')

plt.savefig(f"{cupixpath}/data/chi2_scans/chi2_{pars_to_test[0]}_{pars_to_test[1]})")
# Add the contour line levels to the colorbar

# print("chi2 shape", chi2_grid.shape)
# for pair in pairs:
#     mesh_loc_x = pars_to_test.index(pair[0])
#     mesh_loc_y = pars_to_test.index(pair[1])
#     x_ = grid[mesh_loc_x]
#     y_ = grid[mesh_loc_y]
#     plt.contour(x_, y_, chi2_grid)

# np.save(f"{cupixpath}/data/chi2_scans/