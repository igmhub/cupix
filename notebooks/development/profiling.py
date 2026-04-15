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
# # Profile different parts of the code

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import h5py as h5
from pathlib import Path
import time
# # %load_ext autoreload
# # %autoreload 2
# %load_ext line_profiler

# %%
from lace.cosmo import cosmology
import cupix
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood

# %% [markdown]
# ### Load some data

# %%
basedir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/"
fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5"
data = DESI_DR2(fname, kM_max_cut_AA=1.0, km_max_cut_AA=1.2)

# %% [markdown]
# ### Setup a theory and likelihood object

# %%
# specify one redshift bin
iz = 0
z = data.z[iz]
cosmo = cosmology.Cosmology()
config = {'verbose': True, 'default_lya_model': 'best_fit_igm_from_p1d'}
#config = {'verbose': True, 'default_lya_model': 'best_fit_arinyo_from_p1d'}
theory = Theory(z=z, fid_cosmo=cosmo, config=config)

# %%
like = Likelihood(data=data, theory=theory, iz=iz, verbose=True)

# %% [markdown]
# ### Start profiling the likelihood evaluation

# %%
# %%time
_ = like.get_log_like()

# %% [markdown]
# ANDREU: We should update the conclustions here. Old text below.
#
# The following line shows that the function get_convolved_Px_AA takes the vast majority of the time. 

# %%
# %lprun -f like.get_log_like like.get_log_like()

# %% [markdown]
# ### Profile the theory evaluations

# %% [markdown]
# ANDREU: We should update the conclustions here. Old text below.
#
# The following line shows that get_px_AA takes the majority of time from get_convolved_Px_AA (1s) rather than convolve_window (.0015 s)

# %%
# %lprun -f like.get_convolved_px like.get_convolved_px() # compare to .897s total, 4 calls per get_Px

# %% [markdown]
# ANDREU: We should update the conclustions here. Old text below.
#
# The following line shows that lyap3d.model_Px takes the most time (~1s)

# %%
k_m = data.k_m[iz]
theta_a = (data.theta_min_a_arcmin + data.theta_max_a_arcmin)/2.

# %%
# %lprun -f like.theory.get_px_obs like.theory.get_px_obs(theta_arc=theta_a, k_AA=k_m)


# %%
# %lprun -f like.theory.get_px_lya_obs like.theory.get_px_lya_obs(theta_arc=theta_a, k_AA=k_m)

# %% [markdown]
# ANDREU: We should update the conclustions here. Old text below.
#
# The emulator is taking .02s, the majority of time (1.02s) still spent in predicting Px

# %% [markdown]
# ### Profile the emulator calls themselves

# %%
# %%time
_ = theory.lya_model.get_lya_params(cosmo=theory.fid_cosmo, params={})

# %%
