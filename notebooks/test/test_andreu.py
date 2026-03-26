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
iz=0
z=data.z[iz]
theory = Theory(z=z, fid_cosmo=cosmo, config={'verbose':True})

# %%
# numpy array of kpar values (in inverse AA)
kp_AA = np.linspace(0.01, 2.0, 1000)
# numpy array of theta values (in arcmin)
theta_arc = np.linspace(0.1, 60.0, 100)
# get a 2D array prediction
px_obs = theory.get_px_lya_obs(theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params={})

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
theory.lya_model.default_lya_params

# %%
# this can be a list of likelihood parameters or a dictionary
params = {'bias': -0.12, 'beta': 1.6, 'q1': .3}
px_obs = theory.get_px_lya_obs(theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params=params)

# %%
for beta in [0.0, 1.0, 2.0]:
    px_obs = theory.get_px_lya_obs(theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params={'beta':beta})
    for it_A in [5]:
        label = 'theta = {:.2f} arcmin, beta={:.2f}'.format(theta_arc[it_A], beta)
        plt.plot(kp_AA, px_obs[it_A], label=label)
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')

# %%
for q1 in [0.0, 0.5, 0.9]:
    px_obs = theory.get_px_lya_obs(theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params={'q1':q1})
    
    for it_A in [5]:
        label = 'theta = {:.2f} arcmin, q1={:.2f}'.format(theta_arc[it_A], q1)
        plt.plot(kp_AA, px_obs[it_A], label=label)
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')

# %% [markdown]
# ### Set up the likelihood with one redshift 

# %%
# first with the old code
from cupix.likelihood import likelihood
like = likelihood.Likelihood(data, theory, z=z, verbose=True)

# %%
# get the convolved Px for a chosen theta bin
it_A = 5
theta_bin_choice = data.theta_centers_arcmin[it_A]
print("Getting the convolved Px for theta bin {} arcmin".format(theta_bin_choice))
Px_convolved = like.get_convolved_Px_AA(theta_A=it_A)

# %%
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
# now with the new likelihood class
from cupix.likelihood import new_likelihood
new_like = new_likelihood.Likelihood(data=data, theory=theory, iz=iz, verbose=True)

# %%
model_px=new_like.get_convolved_px(params={})

# %%
# compare old and new likelihoods
it_A = 5
new_px = model_px[it_A]
old_px = like.get_convolved_Px_AA(theta_A=it_A)[0]
plt.plot(k_M, old_px, label='old like')
plt.plot(k_M, new_px, ls=':', label='new like')
plt.legend()
plt.xlabel('k [1/A]')
plt.ylabel('Px [A]')

# %%
# compare both likelihoods
old_chi2=like.get_chi2()
new_chi2=new_like.get_chi2()
print(old_chi2, new_chi2)

# %%
# chi2 scan for bias
N=20
bias_arr = np.linspace(-0.120, -0.115, N)
chi2_arr = np.zeros(N)
new_like.verbose=False
new_like.theory.verbose=False
for i in range(N):
    chi2_arr[i] = new_like.get_chi2(params={'bias':bias_arr[i]})
plt.plot(bias_arr, chi2_arr)
plt.axhline(y=0, ls=':')
plt.xlabel('bias')
plt.ylabel('chi2');

# %% [markdown]
# ### Try to reproduce exactly the theory used in the forecast

# %%
f = h5.File(data_file)
dict(f)

# %%
lya_params = {
    'bias': f['arinyo_pars'].attrs['bias_0'],
    'beta': f['arinyo_pars'].attrs['beta_0'],
    'av': f['arinyo_pars'].attrs['av_0'],
    'bv': f['arinyo_pars'].attrs['bv_0'],
    'kp_Mpc': f['arinyo_pars'].attrs['kp_0'],
    'q1': f['arinyo_pars'].attrs['q1_0'],
    'q2': f['arinyo_pars'].attrs['q2_0'],
    'kv_Mpc': np.exp( np.log(f['arinyo_pars'].attrs['kvav_0']) / f['arinyo_pars'].attrs['av_0'] )
}

# %%
cosmo_params = {}
for key in f['cosmo_params'].attrs.keys():
    cosmo_params[key] = f['cosmo_params'].attrs[key]
print(cosmo_params)

# %%
params = cosmo_params | lya_params
print(params.keys())

# %%
new_like.verbose=True
new_like.theory.verbose=True
new_chi2=new_like.get_chi2(params=params)

# %%
from cupix.likelihood.likelihood_parameter import likeparam_from_dict
like_params = likeparam_from_dict(params)
like.verbose=True
like.theory.verbose=True
old_chi2=like.get_chi2(like_params=like_params)

# %%
print(old_chi2, new_chi2)

# %%

# %%
