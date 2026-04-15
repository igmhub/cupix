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
import numpy as np
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.generate_fake_data import FakeData
from cupix.likelihood.theory import Theory
import forestflow
from forestflow.archive import GadgetArchive3D
import matplotlib.pyplot as plt
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
import cupix
import os
from pathlib import Path
import pandas as pd
import h5py as h5
from lace.cosmo import cosmology

# %load_ext autoreload
# %autoreload 2
from cupix.utils.utils import get_path_repo
cupixpath = get_path_repo('cupix')
print(cupixpath)


# %% [markdown]
# ## Set up the forecast configuration

# %%
# --------- settings ---------
param_mode = "igm" # "igm" or "arinyo"
add_noise  = False
lya_model  = "default_igm_from_gadget" # other option is gadget_simulation
# --------------------------

# %%
# input the data file from which the covariance matrix and binning configurations will be used
data_file = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5"
data = DESI_DR2(data_file)
data_label = 'real'
# overwrite data z with gadget z to facilitate gadget theory
data.z = [2.25, 2.5, 2.75, 3.0]

# %%
theories = []
# set the default cosmology
cosmo = cosmology.Cosmology()

if 'gadget' in lya_model:
    # choose forecast location
    floc = "central" # "central" or "random"
    # "central" uses the central simulation in the training set
    # "random" randomly selects points within the minimum and maximum range of each training parameter
else:
    floc = ""

# set the Gadget central theory values
if 'gadget' in lya_model and floc=="central":
    for z in data.z:
        print(z)
        theories.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': True, 'default_lya_model':lya_model}))
        

# %%
# prepare to generate and write fake data
filepath = cupixpath + "/data/px_measurements/forecast/"

if add_noise:
    noise_str = 'noisy'
else:
    noise_str = 'noiseless'

if floc=="random":
    rng = np.random.default_rng()
    forecast_str = f"_{rng.choice(1000):03d}"
else:
    forecast_str = ""

if 'gadget' in lya_model:
    ftype = 'gadget'
savestr = f"{filepath}/forecast_{ftype}{floc}{forecast_str}_{data_label}_{Path(data_file).stem}_{noise_str}.hdf5"
if os.path.exists(savestr):
    print("File already exists at", savestr)
else:
    print("Will generate forecast and save to", savestr)

# %%
if floc=='random':
    theories = []
    gadget_short_info_file = cupixpath + '/data/emulator/ff_training_info.csv'
    train_test_info = pd.read_csv(gadget_short_info_file)
    for z in data.z:
        print(train_test_info.columns)
        igm_parnames = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
        ff_parnames = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']
        sim_iz = np.argmin(np.abs(train_test_info['z'] - z))
        assert np.isclose(train_test_info['z'][sim_iz], z), "Redshift in training info file does not match data redshift"
        if param_mode == "igm":
            pars = igm_parnames
        elif param_mode == "arinyo":
            pars = ff_parnames
        # randomly select a value within the min and max range of each parameter at this redshift
        params = {}
        for par in pars:
            if par+"_min" not in train_test_info.columns or par+"_max" not in train_test_info.columns:
                print("Min/max of parameter", par, "not found in training info file")
            else:
                params[par] = rng.uniform(train_test_info[par+"_min"][sim_iz], train_test_info[par+"_max"][sim_iz])
        print("Inputting defualts for z=", z, "with parameters", params)
        theories.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': True, 'default_lya_model':lya_model, **params}))


# %%
# initialize Likelihoods
likes = []
for iz, z in enumerate(data.z):
    likes.append(Likelihood(data=data, theory=theories[iz], iz=iz, verbose=False))


# %%
# make a copy of the data file at the forecast location
with h5.File(data_file, 'r') as f:
    with h5.File(savestr, 'w') as f_out:
        # copy all groups and datasets from the original file to the new file
        for group_name in f.keys():
            if group_name != "P_Z_AM":
                f.copy(group_name, f_out)
        # for the P_Z_AM group, we will overwrite the datasets with our forecast
        px_group = f_out.create_group("P_Z_AM")
        cosmo = f_out.create_group('cosmo_params')
        # save the iz=0 cosmo since it doesn't change with z
        for par in likes[0].theory.fid_cosmo.background_params:
            print(par, likes[0].theory.fid_cosmo.background_params[par])
            cosmo.attrs[par] = likes[0].theory.fid_cosmo.background_params[par]
        for iz, like in enumerate(likes):
            px_group_z = px_group.create_group(f'z_{iz}')
            px_theory, lya_params = like.generate_px_forecast(add_noise=add_noise)
            for theta_A_ind in range(px_theory.shape[0]):
                px_group_z[f'theta_rebin_{theta_A_ind}/'] = px_theory[theta_A_ind]
            # add the theory parameters used for the forecast as attributes to the P_Z_AM group,
            # and the lya params from the emulator
            px_group_z.attrs['default_lya_model'] = like.theory.lya_model.default_lya_model
            lya_params_group = px_group_z.create_group('lya_params')
            for par in lya_params:
                lya_params_group.attrs[par] = lya_params[par]
            igm_params_group = px_group_z.create_group('igm_params')
            if like.theory.lya_model.default_igm_params is not None:
                for par in like.theory.lya_model.default_igm_params:
                    igm_params_group.attrs[par] = like.theory.lya_model.default_igm_params[par]
                

# %% [markdown]
# ## Test the forecast output

# %%
# make sure it worked
forecast_res = DESI_DR2(savestr)


# %%
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = forecast_res.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_M={Nk_M}, Nk_m={Nk_m}")
# rebinned values
Nz, Nt_A, Nk_M = forecast_res.Px_ZAM.shape
print(f"rebinned values: Nz={Nz}, Nt_A={Nt_A}, Nk_M={Nk_M}")

# %%
# get the central value of each redshift bin, of length Nz
zs = forecast_res.z
print(zs)
# get a 1D array of central values of the measured k bins, of length Nk_M
k_M = data.k_M_centers_AA
# get two 1D arrays with the edges of each theta bin, of length Nt_A each
theta_A_min = data.theta_min_A_arcmin
theta_A_max = data.theta_max_A_arcmin


# %%
# make a plot for a couple of theta bins, and one redshift bin
def plot_theta_bins(data, k_M, iz, it_M):
    label = '{:.2f} < theta < {:.2f}'.format(theta_A_min[it_M], theta_A_max[it_M])
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it_M]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_M]))
    plt.errorbar(k_M, Px, sig_Px, label=label)
    


# %%
plot_theta_bins(forecast_res, k_M, iz=3, it_M=5)
plot_theta_bins(data, k_M, iz=3, it_M=5)
plt.legend()

# %%
for iz in range(4):
    for i in range(8):
        plot_theta_bins(forecast_res, k_M, iz=iz, it_M=i)        
    plt.title("z={:.2f}".format(zs[iz]))
    plt.legend()
    plt.show()
    plt.clf()

# %%
with h5.File(savestr) as f:
    print(f.keys())
    print(f['cosmo_params'].attrs.keys(), f['cosmo_params'].attrs['H0'], f['cosmo_params'].attrs['omch2'])
    print(f['P_Z_AM']['z_0'].keys())
    print(f['P_Z_AM']['z_0']['igm_params'].attrs.keys(), f['P_Z_AM']['z_0']['igm_params'].attrs['Delta2_p'], f['P_Z_AM']['z_0']['igm_params'].attrs['mF'])
    print(f['P_Z_AM']['z_0']['lya_params'].attrs.keys(), f['P_Z_AM']['z_0']['lya_params'].attrs['bias'], f['P_Z_AM']['z_0']['lya_params'].attrs['beta'])
    

# %%

# %%
