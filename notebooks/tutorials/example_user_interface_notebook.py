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
import matplotlib.pyplot as plt
import h5py as h5
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.theory import Theory


# %% [markdown]
# ### Step 1: Load some data

# %%
data_file = "../../data/px_measurements/forecast/fcast_best_fit_arinyo_from_p1d_real_bf3_binned_out_px-zbins_4-thetabins_10_w_res_noisy.hdf5"
data = DESI_DR2(config={'data_file':data_file, 'kM_max_cut_AA':1, 'km_max_cut_AA':1.2})
# kM_max_cut determines max of the widely-binned k; km_max_cut cuts the finely-binned k that goes into the window matrix

# %%
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = data.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_m={Nk_m}")
# rebinned values
Nz, Nt_A, Nk_M = data.Px_ZAM.shape
print(f"rebinned values: Nz={Nz}, Nt_A={Nt_A}, Nk_M={Nk_M}")

# %% [markdown]
# ### Plot a given Px measurement

# %%
# get the central value of each redshift bin, of length Nz
zs = data.z
# get a 1D array of central values of the measured k bins, of length Nk_M
k_M = data.k_M_centers_AA
# get two 1D arrays with the edges of each theta bin, of length Nt_A each
theta_A_min = data.theta_min_A_arcmin
theta_A_max = data.theta_max_A_arcmin


# %%
# make a plot for a couple of theta bins, and one redshift bin
def plot_theta_bins(data, k_M, iz, it_M):
    label = '{} < theta < {}'.format(theta_A_min[it_M], theta_A_max[it_M])
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it_M]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_M]))
    print(len(k_M), len(Px), len(sig_Px))
    plt.errorbar(k_M, Px, sig_Px, label=label)


# %%
plot_theta_bins(data, k_M, iz=0, it_M=0)

# %% [markdown]
# # Set up theory with minimal effort

# %%

print("Input data file has redshift at,", data.z)
cosmo_dict = {'H0': 67}
cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_dict)
# default_theory options are:
# 'best_fit_arinyo_from_p1d': best fit to the DESI DR1 P1D data from Chaves+2026
# 'best_fit_igm_from_p1d': same but for IGM parameters
# 'best_fit_arinyo_from_colore': best fit to xi from colore mocks. Only works for z=2.2, 2.4, 2.6, 2.8
# for each z, set up a Theory object
theory_config = {'default_lya_theory': 'best_fit_arinyo_from_p1d', 'emulator_label': 'forest_mpg', 'verbose': True}
theories = []
for z in data.z:
    theories.append(Theory(z, fid_cosmo=cosmo, config=theory_config))
# theory_colore = Theory(z, bkgd_cosmo=cosmo, default_lya_theory='best_fit_arinyo_from_colore', p3d_label='arinyo', emulator_label='forestflow_emu', k_unit='iAA', verbose=True)

# %%
# check full cosmo dictionary to see what other default parameters were used
theories[0].fid_cosmo.get_background_params()

# %%
# you can get a parameter by specifying its name.
print(theories[0].get_param('bias'))


# %%
print(theories[1].get_param('mF'))

# %% [markdown]
# As you can see, an error will be raised if the parameter doesn't exist.

# %%
print(theories[3].get_param('b_H'), theories[3].include_hcd)

# %% [markdown]
# In this case, we check a parameter that we are not even going to use, so be cautious to not be confused. You can always check
# include_hcd, etc, to see what will be used.

# %%
# pick a redshift
iz = 0
theory = theories[iz]

# %%
# numpy array of kpar values (in inverse AA)
kp_AA = np.linspace(0.01, 2.0, 100)
# numpy array of theta values (in arcmin)
theta_arc = np.linspace(0.1, 60.0, 100)
# get a 2D array prediction
Px_model = theory.get_px_obs(theta_arc=theta_arc, k_AA=kp_AA)

# %%
# plot the prediction for a couple of theta values

for it in [0, 5, 10]:
    label = 'theta = {}'.format(theta_arc[it])
    plt.plot(kp_AA, Px_model[it], label=label)
plt.title(f'Theory prediction for z={z}')
# # plot the data on top
# plot_theta_bins(data, k_M, iz=iz, it_M=0)
# plot_theta_bins(data, k_M, iz=iz, it_M=5)



# %% [markdown]
# ### Now make predictions for different parameter values

# %%
# this can be a list of likelihood parameters or a dictionary
params = {'bias': 0.1, 'beta': 1.6, 'q1': .3}
# params = {'mF': 1., 'n_p': .3}
# this will only modify the input parameters, and leave others unchanged
Px_model_newparams = theory.get_px_obs(theta_arc=theta_arc, k_AA=kp_AA, params=params)


# %%
# plot the prediction for a couple of theta values
iz = 0 # Px_model has shape [Nz, Nt_A, Nk_M], so this is the redshift index of the evaluated redshifts, which is 0 if only one was used
for it in [0, 5, 10]:
    label = 'theta = {:.1f}'.format(theta_arc[it])
    plt.plot(kp_AA, Px_model[it], label=label)
    plt.plot(kp_AA, Px_model_newparams[it], label=label + ' new params', linestyle='dashed')
plt.title(f'Theory prediction for z={z}')
plt.legend()

# %%
# you will not be able to pass a parameter that doesn't make sense for the theory default model
# For example, the default model was setup with the Arinyo model params, so mean flux does not make sense to pass.
# This line should throw an error.

mF = 0.7
# these would use the initial values for other params
Px_model = theory.get_px_obs(theta_arc=theta_arc, k_AA=kp_AA, params={'mF':mF})


# %% [markdown]
# ### Set up the likelihood with one redshift 

# %%
like = Likelihood(data=data, theory=theory, iz=iz, 
                  config={'verbose':True})

# %%
# get the convolved Px for a chosen theta bin
it_M = 0
theta_bin_choice = data.theta_centers_arcmin[it_M]
print("Getting the convolved Px for theta bin {} arcmin".format(theta_bin_choice))
Px_convolved = like.get_convolved_px()
# plot the convolved Px
# plot the convolved Px. Always has shape Nt_A, Nk_M so we need to specify the theta bin index

plt.plot(data.k_M_centers_AA, Px_convolved[it_M], label='convolved Px')
# without convolution, it would have been:
Px_model = theory.get_px_obs(theta_arc=theta_bin_choice, k_AA=kp_AA)
# Px_model always has shape [Nz, Nt_A, Nk_M], so we need to specify the redshift and theta bin indices or just squeeze the result
plt.plot(kp_AA, Px_model, label='unconvolved Px') 
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')


# plot the data on top
plot_theta_bins(data, k_M, iz=iz, it_M=it_M)


# %%
# same, but with special parameters passed

params = {'bias': -.15, 'beta': .1}

# get the convolved Px for a chosen theta bin
it_M = 0
theta_bin_choice = data.theta_centers_arcmin[it_M]
print("Getting the convolved Px for theta bin {} arcmin".format(theta_bin_choice))
Px_convolved = like.get_convolved_px(params=params)
# plot the convolved Px
# plot the convolved Px. Always has shape Nt_A, Nk_M so we need to specify the theta bin index

plt.plot(data.k_M_centers_AA, Px_convolved[it_M], label='convolved Px')
# without convolution, it would have been:
Px_model = theory.get_px_obs(theta_arc=theta_bin_choice, k_AA=kp_AA, params=params)
# Px_model always has shape [Nz, Nt_A, Nk_M], so we need to specify the redshift and theta bin indices or just squeeze the result
plt.plot(kp_AA, Px_model, label='unconvolved Px') 
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')

# plot the data on top
plot_theta_bins(data, k_M, iz=iz, it_M=it_M)


# %% [markdown]
# ### Examine the chi2

# %%
# pass nothing to check the default
like.get_chi2()

# %%
# pass by dicitonary to check what happens when params are fiddled with
like.get_chi2(params=params)

# %% [markdown]
# If data_file is the forecast file, you can check for the truth, including the correct cosmology

# %%
with h5.File(data_file) as f:
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
    if 'contamination' in f.keys():
        print(f['contamination'].attrs.keys())
        print(f['contamination'].attrs['b_H'], f['contamination'].attrs['pC'])

# %%

# %%
