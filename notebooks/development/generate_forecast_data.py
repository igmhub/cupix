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
from cupix.likelihood.theory import Theory
import matplotlib.pyplot as plt
from cupix.px_data.data_DESI_DR2 import DESI_DR2
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
# --------- user settings --------
lya_model  = "best_fit_arinyo_from_p1d" # options: "central_igm_from_gadget", "random_igm_from_gadget", "best_fit_arinyo_from_p1d", "best_fit_igm_from_p1d", "best_fit_arinyo_from_colore"
add_noise  = False
params = {} # input a dictionary of specific params and values if you want to modify the default theory for the forecast
# --------------------------

assert 'igm' in lya_model or 'arinyo' in lya_model, "lya_model must contain either 'igm' or 'arinyo'"


if 'gadget' in lya_model:
    # set forecast location
    if "central" in lya_model:
        floc = "central" # "central" or "random"
    elif "random" in lya_model:
        floc = "random"
else:
    floc = ""

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

# set the Gadget central theory values
if ('gadget' in lya_model and 'central' in lya_model) or 'colore' in lya_model or 'p1d' in lya_model:
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
    rand_str = f"_{rng.choice(1000):03d}"
else:
    rand_str = ""

savestr = f"{filepath}/fcast_{lya_model}{rand_str}_{data_label}_{Path(data_file).stem}_{noise_str}.hdf5"
if os.path.exists(savestr):
    print("File already exists at", savestr)
else:
    print("Will generate forecast and save to", savestr)

# %%
if 'gadget' in lya_model and floc=='random':
    theories = []
    gadget_short_info_file = cupixpath + '/data/emulator/ff_training_info.csv'
    train_test_info = pd.read_csv(gadget_short_info_file)
    for z in data.z:
        print(train_test_info.columns)
        igm_parnames = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
        ff_parnames = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']
        sim_iz = np.argmin(np.abs(train_test_info['z'] - z))
        assert np.isclose(train_test_info['z'][sim_iz], z), "Redshift in training info file does not match data redshift"
        if 'igm' in lya_model:
            pars = igm_parnames
        elif "arinyo" in lya_model:
            pars = ff_parnames
        # randomly select a value within the min and max range of each parameter at this redshift
        gadget_params = {}
        for par in pars:
            if par+"_min" not in train_test_info.columns or par+"_max" not in train_test_info.columns:
                print("Min/max of parameter", par, "not found in training info file")
            else:
                gadget_params[par] = rng.uniform(train_test_info[par+"_min"][sim_iz], train_test_info[par+"_max"][sim_iz])
        print("Inputting defualts for z=", z, "with parameters", gadget_params)
        theories.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': True, 'default_lya_model':lya_model, **gadget_params}))


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
            if group_name not in ["P_Z_AM", 'z_centers']:
                f.copy(group_name, f_out)
        f_out['metadata'].attrs['z_centers'] = data.z # do this manually in case data.z was overwritten by the user earlier in the notebook
        f_out['metadata'].attrs['z_centers_orig'] = f['metadata'].attrs['z_centers'] # save the original z_centers as well for reference. In some cases this will be the same as the new z_centers.
        # for the P_Z_AM group, we will overwrite the datasets with our forecast
        px_group = f_out.create_group("P_Z_AM")
        cosmo_group = f_out.create_group('cosmo_params')
        # save the iz=0 cosmo since it doesn't change with z
        for par in likes[0].theory.fid_cosmo.background_params:
            print(par, likes[0].theory.fid_cosmo.background_params[par])
            cosmo_group.attrs[par] = likes[0].theory.fid_cosmo.background_params[par]
        for iz, like in enumerate(likes):
            px_group_z = px_group.create_group(f'z_{iz}')
            if 'igm' in like.theory.lya_model.default_lya_model:
                # we want to save the lya params in the output file
                cosmo = like.theory.get_cosmology(params=params)
                ff_params = like.theory.lya_model.get_lya_params(cosmo, params)
                print("will save ff_params")
            else:
                ff_params = None
            px_theory = like.generate_px_forecast(params=params, add_noise=add_noise)
            for theta_A_ind in range(px_theory.shape[0]):
                px_group_z[f'theta_rebin_{theta_A_ind}/'] = px_theory[theta_A_ind]
            # add the theory parameters used for the forecast as attributes to the P_Z_AM group,
            # and the lya params from the emulator
            px_group_z.attrs['default_lya_model'] = like.theory.lya_model.default_lya_model
            if ff_params is not None:
                ff_params_group = px_group_z.create_group('ff_emulated_params')
                for par in ff_params:
                    ff_params_group.attrs[par] = ff_params[par]
            if like.theory.lya_model.default_igm_params is not None:
                igm_params_group = px_group_z.create_group('igm_params')
                for par in like.theory.lya_model.default_igm_params:
                    igm_params_group.attrs[par] = like.theory.lya_model.default_igm_params[par]
            if like.theory.lya_model.default_lya_params is not None:
                lya_params_group = px_group_z.create_group('lya_params')
                for par in like.theory.lya_model.default_lya_params:
                    lya_params_group.attrs[par] = like.theory.lya_model.default_lya_params[par]

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
plot_theta_bins(data, k_M, iz=3, it_M=5) # original data, not saved in forecast file
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
    print(f['P_Z_AM']['z_0'].attrs['default_lya_model'])
    print(f['P_Z_AM']['z_0'].keys())
    if 'igm_params' in f['P_Z_AM']['z_0'].keys():
        print(f['P_Z_AM']['z_0']['igm_params'].attrs.keys(), f['P_Z_AM']['z_0']['igm_params'].attrs['Delta2_p'], f['P_Z_AM']['z_0']['igm_params'].attrs['mF'])
    if 'lya_params' in f['P_Z_AM']['z_0'].keys():
        print(f['P_Z_AM']['z_0']['lya_params'].attrs.keys(), f['P_Z_AM']['z_0']['lya_params'].attrs['bias'], f['P_Z_AM']['z_0']['lya_params'].attrs['beta'])
    if 'ff_emulated_params' in f['P_Z_AM']['z_0'].keys():
        print("Here")
        print(f['P_Z_AM']['z_0']['ff_emulated_params'].attrs.keys(), f['P_Z_AM']['z_0']['ff_emulated_params'].attrs['bias'], f['P_Z_AM']['z_0']['ff_emulated_params'].attrs['beta'])

# %%
