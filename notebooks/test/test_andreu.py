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
import matplotlib.pyplot as plt
import h5py as h5
import pandas as pd
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
#from cupix.likelihood.lya_theory import Theory
#from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.px_data.data_DESI_DR2 import DESI_DR2

# %% [markdown]
# ### Step 1: Load some data

# %%
#data_file = "../../data/px_measurements/forecast/forecast_ffcentral_real_binned_out_px-zbins_4-thetabins_9_w_res_noiseless_z0.hdf5"
data_file = "../../data/px_measurements/forecast/test.hdf5"
data = DESI_DR2(data_file, kM_max_cut_AA=1, km_max_cut_AA=1.2)

# %%
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = data.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_M={Nk_M}, Nk_m={Nk_m}")
# rebinned values
Nz, Nt_A, Nk_M = data.Px_ZAM.shape
print(f"rebinned values: Nz={Nz}, Nt_A={Nt_A}, Nk_M={Nk_M}")

# %%
# get a 1D array of central values of the measured k bins, of length Nk_M
k_M = data.k_M_centers_AA
# get two 1D arrays with the edges of each theta bin, of length Nt_A each
theta_A_min = data.theta_min_A_arcmin
theta_A_max = data.theta_max_A_arcmin


# %%
# make a plot for a couple of theta bins, and one redshift bin
def plot_theta_bins(data, k_M, iz, it_A):
    label = '{:.2f} < theta < {:.2f}'.format(theta_A_min[it_A], theta_A_max[it_A])
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it_A]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_A]))
    plt.errorbar(k_M, Px, sig_Px, label=label)


# %%
for it_A in [6, 7, 8]:
    plot_theta_bins(data, k_M, iz=0, it_A=it_A)
plt.xlabel('k [1/A]')
plt.ylabel('Px [A]')
plt.legend();

# %% [markdown]
# # Set up theory with minimal effort

# %%
cosmo_params = {"H0": 67.66}
cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params)

# %%
theory = Theory(zs=data.z, fid_cosmo=cosmo, config={'verbose':True})

# %%
# numpy array of kpar values (in inverse AA)
kp_AA = np.linspace(0.01, 2.0, 1000)
# numpy array of theta values (in arcmin)
theta_arc = np.linspace(0.1, 60.0, 100)
# get a 2D array prediction
px_obs = theory.get_px_lya_obs(iz=0, theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params={})

# %%
# plot the prediction for a couple of theta values
for it_A in [10, 20, 40]:
    label = "theta = {:.2f}' ".format(theta_arc[it_A])
    plt.plot(kp_AA, px_obs[it_A], label=label)
plt.xlim([0,1])
plt.xlabel('k [1/A]')
plt.ylabel('Px [A]')
plt.legend();

# %% [markdown]
# ### Now make predictions for different parameter values

# %%
theory.lya_models[0].default_lya_params

# %%
# this can be a list of likelihood parameters or a dictionary
params = {'bias': -0.12, 'beta': 1.6, 'q1': .3}
px_obs = theory.get_px_lya_obs(iz=0, theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params=params)

# %%
for beta in [0.0, 1.0, 2.0]:
    px_obs = theory.get_px_lya_obs(iz=0, theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params={'beta':beta})
    for it_A in [5]:
        label = 'theta = {:.2f} arcmin, beta={:.2f}'.format(theta_arc[it_A], beta)
        plt.plot(kp_AA, px_obs[it_A], label=label)
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')

# %%
for q1 in [0.0, 0.5, 0.9]:
    px_obs = theory.get_px_lya_obs(iz=0, theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params={'q1':q1})
    
    for it_A in [5]:
        label = 'theta = {:.2f} arcmin, q1={:.2f}'.format(theta_arc[it_A], q1)
        plt.plot(kp_AA, px_obs[it_A], label=label)
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')

# %% [markdown]
# ### Set up the likelihood with one redshift 

# %%
z=data.z[0]
print(z)
likelihood = Likelihood(data, theory, z=z, verbose=True)

# %%
# get the convolved Px for a chosen theta bin
it_A = 5
theta_bin_choice = data.theta_centers_arcmin[it_A]
print("Getting the convolved Px for theta bin {} arcmin".format(theta_bin_choice))
Px_convolved = likelihood.get_convolved_Px_AA(theta_A=it_A)

# %%
Px_convolved.shape

# %%
# plot the convolved Px
# plot the convolved Px. Always has shape Nt_A, Nk_M so we need to specify the theta bin index or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_convolved), label='convolved Px')
# without convolution, it would have been:
Px_model = theory.get_px_AA(k_AA=data.k_M_centers_AA, theta_arcmin=theta_bin_choice, z=[z])
# Px_model always has shape [Nz, Nt_A, Nk_M], so we need to specify the redshift and theta bin indices or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_model), label='unconvolved Px') 
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')


# %%
theory.zs

# %%
data.z

# %%
len(data.theta_max_A_arcmin)

# %%
import cupix.likelihood.new_likelihood as new_like

# %%
likelihood = new_like.Likelihood(data=data, theory=theory, iz=0, verbose=True)

# %%
model_px=likelihood.get_convolved_px(params={})

# %%
it_A = 5
new_px=model_px[it_A]
new_px.shape
plt.plot(k_M, Px_convolved[0])
plt.plot(k_M, new_px)
plt.xlabel('k [1/A]')
plt.ylabel('Px [A]')

# %%
chi2=likelihood.get_chi2()
print(chi2)

# %%
chi2=likelihood.get_chi2(params={'bias':-0.125})
print(chi2)

# %%

# %%

# %%
# same, but with special parameters passed

params = {'beta': 1.7}

it_A = 3
theta_bin_choice = data.theta_centers_arcmin[it_A]
print("Getting the convolved Px for theta bin {} arcmin".format(theta_bin_choice))
Px_convolved = likelihood.get_convolved_Px_AA(theta_A=it_A, like_params=params)
# plot the convolved Px
# plot the convolved Px. Always has shape Nt_A, Nk_M so we need to specify the theta bin index or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_convolved), label='convolved Px')
# without convolution, it would have been:
Px_model = theory.get_px_AA(k_AA=data.k_M_centers_AA, theta_arcmin=theta_bin_choice, zs=[z], like_params=params)
# Px_model always has shape [Nz, Nt_A, Nk_M], so we need to specify the redshift and theta bin indices or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_model), label='unconvolved Px') 
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')
# plot the data on top
plot_theta_bins(data, k_M, iz=likelihood.data_iz, it_A=it_A)


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
it_A = 0
theta_bin_choice = data.theta_centers_arcmin[it_A]
print("Getting the convolved Px for theta bin {} arcmin".format(theta_bin_choice))
Px_convolved = likelihood.get_convolved_Px_AA(theta_A=it_A, like_params=like_params)
# plot the convolved Px. Always has shape Nt_A, Nk_M so we need to specify the theta bin index or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_convolved), label='convolved Px')
# without convolution, it would have been:
Px_model = theory.get_px_AA(k_AA=data.k_M_centers_AA, theta_arcmin=theta_bin_choice, zs=[z], like_params=like_params)
# Px_model always has shape [Nz, Nt_A, Nk_M], so we need to specify the redshift and theta bin indices or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_model), label='unconvolved Px') 
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')

# plot the data on top
plot_theta_bins(data, k_M, iz=likelihood.data_iz, it_A=it_A)


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

# %% [markdown]
# If data_file is the forecast file, you can pass the truth, including the correct cosmology

# %%
f = h5.File(data_file)
dict(f)

# %%
f['arinyo_pars'].attrs.keys()

# %%
f['like_params'].attrs.keys()

# %%
f['metadata'].attrs.keys()

# %%
f['cosmo_params'].attrs.keys()

# %%
param_mode = 'arinyo' # enter the type of parameters you want
truth_params = {}
with h5.File(data_file) as f:
    for cosmo_par in f['cosmo_params'].attrs.keys():
        cosmo_dict[cosmo_par] = f['cosmo_params'].attrs[cosmo_par]
    if param_mode=='igm':
        for like_par in f['like_params'].attrs.keys():
            truth_params[like_par] = f['like_params'].attrs[like_par]
    elif param_mode=='arinyo':
        for arinyo_par in f['arinyo_pars'].attrs.keys():
            truth_params[arinyo_par] = f['arinyo_pars'].attrs[arinyo_par]
print("Passing forecast cosmology to theory object:", cosmo_dict)

theory = Theory(zs, cosmo_dict=cosmo, default_lya_theory='best_fit_arinyo_from_p1d', emulator_label="forestflow_emu", verbose=True)
# check the parameters
print("Truth params are:")
for p, val in truth_params.items():
    print(p, val)

# %%
likelihood = Likelihood(data, theory, z=z, verbose=True)

# %%
# still a bad chi2 if we don't pass the correct theory params
likelihood.get_chi2()

# %%
like_params = []


arinyo_par_names_iz = [par+"_0" for par in theory.arinyo_par_names]
def update_likepar_list(like_params, par_names):
    for par in par_names:
        like_params.append(LikelihoodParameter(
    name=par,
    value = truth_params[par]
    ))
    return like_params

like_params = update_likepar_list(like_params, arinyo_par_names_iz)

for par in like_params:
    print(par.name, par.value)


# %%
# now should be a nearly-0 chi2 passing the exact truth theory params
likelihood.get_chi2(like_params)

# %%
