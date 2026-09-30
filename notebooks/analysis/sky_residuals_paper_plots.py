# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: cupix
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
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.inference.free_parameter import FreeParameter
from cupix.inference.posterior import Posterior
from cupix.inference.minimize_posterior import Minimizer

# %%
basedir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/"
fname = basedir + "/wp1d/BALs_weighted_to_zero/bf3_binned_out_px-zbins_4-thetabins_10_w_res_wp1d.hdf5"
data = DESI_DR2(config={'data_file':fname, 'kM_min_cut_AA':0.5, 'kM_max_cut_AA':1.5, 'km_max_cut_AA':1.7, 'theta_min_cut_arcmin':8.0})

# %%
# get the central value of each redshift bin, of length Nz
zs = data.z
# get a 1D array of central values of the measured k bins, of length Nk_M
k_M = data.k_M_centers_AA
# get two 1D arrays with the edges of each theta bin, of length Nt_A each
theta_A_min = data.theta_min_A_arcmin
theta_A_max = data.theta_max_A_arcmin

# %%
print(theta_A_min)

# %%
cosmo = cosmology.Cosmology()
config={'verbose': True, 'include_hcd': False, 'include_metal': False,
        'include_sky': True, 'include_continuum': True}
likes_withsky = []
for iz, z in enumerate(data.z): 
    theory = Theory(z=z, fid_cosmo=cosmo, config=config)
    like = Likelihood(data=data, theory=theory, iz=iz, config={'verbose':True})
    likes_withsky.append(like)
    

config={'verbose': True, 'include_hcd': False, 'include_metal': False,
        'include_sky': False, 'include_continuum': True}
likes_nosky = []
for iz, z in enumerate(data.z): 
    theory = Theory(z=z, fid_cosmo=cosmo, config=config)
    like = Likelihood(data=data, theory=theory, iz=iz, config={'verbose':True})
    likes_nosky.append(like)


# %%
for iz, z in enumerate(data.z): 
    likes_withsky[iz].plot_px(multiply_by_k=False, theorylabel="model with sky residuals", datalabel='DESI DR2 (z={})'.format(z), include_probability=False, include_chi2=True, title=)
    likes_nosky[iz].plot_px(multiply_by_k=False, theorylabel="model without sky residuals", datalabel='DESI DR2 (z={})'.format(z), include_probability=False, include_chi2=True)

# %%

# %%
