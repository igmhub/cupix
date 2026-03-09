# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: cupix
# ---

# %%
import numpy as np
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.lya_theory import Theory
from cupix.likelihood.forestflow_emu import FF_emulator
from lace.cosmo import cosmology
import matplotlib.pyplot as plt
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import h5py as h5
import cupix
import pandas as pd
cupixpath = cupix.__path__[0].rsplit('/', 1)[0]
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### Step 1: Load some data

# %%
data_file = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/binned_out_px-zbins_4-thetabins_9_w_res.hdf5"
# "../../data/px_measurements/forecast/forecast_ffcentral_cosmo_igm_real_binned_out_px-zbins_4-thetabins_9_w_res_noiseless.hdf5"
data = DESI_DR2(data_file, kM_max_cut_AA=1, km_max_cut_AA=1.2)
# kmax_cut determines max of the widely-binned k; finely-binned k will still be fully used for the window matrix application

# %%
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = data.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_M={Nk_M}, Nk_m={Nk_m}")
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
z = data.z
cosmo = cosmology.Cosmology(cosmo_params_dict={'H0': 67})
# default_theory options are:
# 'best_fit_arinyo_from_p1d': best fit to the DESI DR1 P1D data from Chaves+2026
# 'best_fit_igm_from_p1d': same but for IGM parameters
# 'best_fit_arinyo_from_colore': best fit to xi from colore mocks. Only works for z=2.2, 2.4, 2.6, 2.8
theory_p1d = Theory(z, fid_cosmo=cosmo, default_lya_theory='best_fit_igm_from_p1d', p3d_label='arinyo', emulator_label='forestflow_emu', k_unit='iAA', verbose=True)
# theory_colore = set_theory(z, bkgd_cosmo=cosmo, default_theory='best_fit_arinyo_from_colore', p3d_label='arinyo', emulator_label='forestflow_emu', k_unit='iAA', verbose=True)

# %%
theory_p1d.default_param_dict

# %%
# you can get a parameter by specifying its name and redshift:
# print(theory_colore.get_param('q1', iz_choice=2))
print(theory_p1d.get_param('mF', iz_choice=2))
# or by only naming it without an index, if you want all redshift values
# theory_colore.get_param('q1')
print(theory_p1d.get_param('mF'))
# or by naming it with the underscore
# theory_colore.get_param('q1_2')
print(theory_p1d.get_param('sigT_Mpc_3'))

# %%
# theory = theory_colore
theory = theory_p1d

# %%
# numpy array of kpar values (in inverse AA)
kp_AA = np.linspace(0.01, 2.0, 100)
# numpy array of theta values (in arcmin)
theta_arc = np.linspace(0.1, 60.0, 100)
# get a 2D array prediction
Px_model = theory.get_px_AA(k_AA=kp_AA, theta_arcmin=theta_arc)

# %%
# plot the prediction for a couple of theta values
# choose a redshift
iz = 0
z  = theory.zs[iz]
for it in [0, 5, 10]:
    label = 'theta = {}'.format(theta_arc[it])
    plt.plot(kp_AA, Px_model[iz][it], label=label)
plt.title(f'Theory prediction for z={z}')
# # plot the data on top
# plot_theta_bins(data, k_M, iz=iz, it_M=0)
# plot_theta_bins(data, k_M, iz=iz, it_M=5)



# %% [markdown]
# ### Now make predictions for different parameter values

# %%
# this could be a list of likelihood parameters, but easier to write it here as a dictionary
params = {'bias': 0.1, 'beta': 1.6, 'q1': .3}
# params = {'mF': 1., 'n_p': .3}
# this would only modify the input parameters, and leave others (like k_p or av) unchanged
Px_model = theory.get_px_AA(k_AA=kp_AA, theta_arcmin=theta_arc, like_params=params, zs=2.4)

# %%
# plot the prediction for a couple of theta values
iz = 0 # Px_model has shape [Nz, Nt_A, Nk_M], so this is the redshift index of the evaluated redshifts, which is 0 if only one was used
for it in [0, 5, 10]:
    label = 'theta = {}'.format(theta_arc[it])
    plt.plot(kp_AA, Px_model[iz][it], label=label)
plt.title(f'Theory prediction for z={z}')
plt.legend()

# %%
z_eval = 2.4
for mF in [0.7, 0.8, 0.9]:
    # these would use the initial values for other params
    Px_model = theory.get_px_AA(k_AA=kp_AA, theta_arcmin=theta_arc, like_params={'mF':mF}, zs=z_eval) 
    iz = 0
    for it in [0]:
        label = 'theta = {} arcmin, mF={}'.format(theta_arc[it], mF)
        plt.plot(kp_AA, Px_model[iz][it], label=label)
plt.title(f'Theory prediction for z={z_eval}')
plt.legend()

# %%
for kF in [8,9,10]:
    # these would use the initial values for other params
    Px_model = theory.get_px_AA(k_AA=kp_AA, theta_arcmin=theta_arc, like_params={'kF_Mpc':kF}, zs=2.4) 
    iz = 0
    z  = theory.zs[iz]
    for it in [0]:
        label = 'theta = {}, kF={} Mpc-1'.format(theta_arc[it], kF)
        plt.plot(kp_AA, Px_model[iz][it], label=label)
plt.title(f'Theory prediction for z={z}')
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')

# %%
for sigT in [.09,.12,.16]:
    # these would use the initial values for other params
    Px_model = theory.get_px_AA(k_AA=kp_AA, theta_arcmin=theta_arc, like_params={'sigT_Mpc':sigT}, zs=2.4) 
    iz = 0
    z  = theory.zs[iz]
    for it in [0]:
        label = 'theta = {}, sigT={} Mpc'.format(theta_arc[it], sigT)
        plt.plot(kp_AA, Px_model[iz][it], label=label)
plt.title(f'Theory prediction for z={z}')
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')

# %% [markdown]
# ### Set up the likelihood with one redshift 

# %%
likelihood = Likelihood(data, theory, z=2.6, verbose=True)

# %%
data.theta_centers_arcmin

# %%
data.theta_max_A_arcmin

# %%
# get the convolved Px for a chosen theta bin
it_M = 0
theta_bin_choice = data.theta_centers_arcmin[it_M]
print("Getting the convolved Px for theta bin {} arcmin".format(theta_bin_choice))
Px_convolved = likelihood.get_convolved_Px_AA(theta_A=it)
# plot the convolved Px
# plot the convolved Px. Always has shape Nt_A, Nk_M so we need to specify the theta bin index or just squeeze the result

plt.plot(data.k_M_centers_AA, np.squeeze(Px_convolved), label='convolved Px')
# without convolution, it would have been:
Px_model = theory.get_px_AA(k_AA=data.k_M_centers_AA, theta_arcmin=theta_bin_choice, zs=2.6)
# Px_model always has shape [Nz, Nt_A, Nk_M], so we need to specify the redshift and theta bin indices or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_model), label='unconvolved Px') 
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')


# %%
# same, but with special parameters passed

params = {'mF': 0.71, 'kF_Mpc': 3}

it_M = 0
theta_bin_choice = data.theta_centers_arcmin[it_M]
print("Getting the convolved Px for theta bin {} arcmin".format(theta_bin_choice))
Px_convolved = likelihood.get_convolved_Px_AA(theta_A=it, like_params=params)
# plot the convolved Px
# plot the convolved Px. Always has shape Nt_A, Nk_M so we need to specify the theta bin index or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_convolved), label='convolved Px')
# without convolution, it would have been:
Px_model = theory.get_px_AA(k_AA=data.k_M_centers_AA, theta_arcmin=theta_bin_choice, zs=2.6, like_params=params)
# Px_model always has shape [Nz, Nt_A, Nk_M], so we need to specify the redshift and theta bin indices or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_model), label='unconvolved Px') 
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')


# plot the data on top
iz_data = np.where(np.isclose(data.z, 2.6))[0][0]
plot_theta_bins(data, k_M, iz=iz_data, it_M=it_M)


# %%
# you could pass these as LikelihoodParameters objects:
# same, but with special parameters passed


like_params = []
like_params.append(LikelihoodParameter(
                name='mF',
                value=0.71,
                min_value=-10., # arbitrary for now
                max_value=10.
            ))
like_params.append(LikelihoodParameter(
                name='kF_Mpc',
                value=3,
                min_value=-10., # arbitrary for now
                max_value=10.
            ))  
# check the parameters
for p in like_params:
    print(p.name, p.value)

# %%

it_M = 0
theta_bin_choice = data.theta_centers_arcmin[it_M]
print("Getting the convolved Px for theta bin {} arcmin".format(theta_bin_choice))
Px_convolved = likelihood.get_convolved_Px_AA(theta_A=it, like_params=like_params)
# plot the convolved Px. Always has shape Nt_A, Nk_M so we need to specify the theta bin index or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_convolved), label='convolved Px')
# without convolution, it would have been:
Px_model = theory.get_px_AA(k_AA=data.k_M_centers_AA, theta_arcmin=theta_bin_choice, zs=2.6, like_params=like_params)
# Px_model always has shape [Nz, Nt_A, Nk_M], so we need to specify the redshift and theta bin indices or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_model), label='unconvolved Px') 
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')


# plot the data on top
iz_data = np.where(np.isclose(data.z, 2.6))[0][0]
plot_theta_bins(data, k_M, iz=iz_data, it_M=it_M)


# %% [markdown]
# ### Examine the chi2

# %%
# pass by dicitonary
likelihood.get_chi2(like_params=params)

# %%
# pass by likelihoodParameters object
likelihood.get_chi2(like_params=like_params)

# %%
# pass nothing
likelihood.get_chi2()

# %%
