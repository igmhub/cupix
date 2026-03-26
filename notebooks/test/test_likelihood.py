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

# %% [markdown]
# # Setup the theory and likelihood class, and play with it

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.likelihood.theory import Theory
from cupix.px_data.data_DESI_DR2 import DESI_DR2


# %% [markdown]
# ### Setup data

# %%
#data_file = "../../data/px_measurements/forecast/forecast_ffcentral_real_binned_out_px-zbins_4-thetabins_9_w_res_noiseless_z0.hdf5"
data_file = "../../data/px_measurements/forecast/test.hdf5"
data = DESI_DR2(data_file, kM_max_cut_AA=1, km_max_cut_AA=1.2)

# %%
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = data.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_m={Nk_m}")
# rebinned values
Nz, Nt_A, Nk_M = data.Px_ZAM.shape
print(f"rebinned values: Nz={Nz}, Nt_A={Nt_A}, Nk_M={Nk_M}")

# %% [markdown]
# ### Set up theory

# %%
cosmo_params = {"H0": 67.66}
cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params)

# %%
iz=0
z=data.z[iz]
theory = Theory(z=z, fid_cosmo=cosmo, config={'verbose':True})

# %% [markdown]
# ### Set up old likelihood

# %%
# first with the old code
from cupix.likelihood import likelihood
old_like = likelihood.Likelihood(data, theory, z=z, verbose=True)

# %%
# get the convolved Px for a chosen theta bin
it_A = 5
theta_bin_choice = data.theta_centers_arcmin[it_A]
print("Getting the convolved Px for theta bin {} arcmin".format(theta_bin_choice))
Px_convolved = old_like.get_convolved_Px_AA(theta_A=it_A)

# %%
# plot the convolved Px. Always has shape Nt_A, Nk_M so we need to specify the theta bin index or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_convolved), label='convolved Px')
# without convolution, it would have been:
Px_model = theory.get_px_AA(k_AA=data.k_M_centers_AA, theta_arcmin=theta_bin_choice, zs=[z])
# Px_model always has shape [Nz, Nt_A, Nk_M], so we need to specify the redshift and theta bin indices or just squeeze the result
plt.plot(data.k_M_centers_AA, np.squeeze(Px_model), label='unconvolved Px') 
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')


# %% [markdown]
# ### Set up new likelihood

# %%
# now with the new likelihood class
from cupix.likelihood import new_likelihood
new_like = new_likelihood.Likelihood(data=data, theory=theory, iz=iz, verbose=True)

# %%
model_px=new_like.get_convolved_px(params={})

# %%
# compare old and new likelihoods
k_M = data.k_M_centers_AA
it_A = 5
new_px = model_px[it_A]
old_px = old_like.get_convolved_Px_AA(theta_A=it_A)[0]
plt.plot(k_M, old_px, label='old like')
plt.plot(k_M, new_px, ls=':', label='new like')
plt.legend()
plt.xlabel('k [1/A]')
plt.ylabel('Px [A]')

# %%
# compare both likelihoods
old_chi2=old_like.get_chi2()
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

# %%
# chi2 scan for ns
N=20
ns_arr = np.linspace(0.90, 1.0, N)
chi2_arr = np.zeros(N)
new_like.verbose=False
new_like.theory.verbose=False
for i in range(N):
    chi2_arr[i] = new_like.get_chi2(params={'ns':ns_arr[i]})
plt.plot(ns_arr, chi2_arr)
plt.axhline(y=0, ls=':')
plt.xlabel('n_s')
plt.ylabel('chi2');

# %% [markdown]
# ### Time the new likelihood

# %% [markdown]
# First with defaut params

# %%
# %timeit chi2=new_like.get_chi2()

# %% [markdown]
# Now with rescaled cosmology

# %%
# %timeit chi2=new_like.get_chi2(params={'ns':0.95})

# %% [markdown]
# Now with new CAMB calls

# %%
# %timeit chi2=new_like.get_chi2(params={'ombh2':0.022})

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
old_like.verbose=True
old_like.theory.verbose=True
old_chi2=old_like.get_chi2(like_params=like_params)

# %%
print(old_chi2, new_chi2)

# %%
