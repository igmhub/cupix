# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Example use of a yaml file for setting up the config

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
#from cupix.likelihood.likelihood_parameter import LikelihoodParameter, like_parameter_by_name
#from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
from cupix.likelihood.free_parameter import FreeParameter
from cupix.likelihood.posterior import Posterior
from cupix.likelihood.minimize_posterior import Minimizer
from cupix.utils.utils import get_path_repo
from cupix.likelihood.config import Config
cupixpath = get_path_repo('cupix')

# %% [markdown]
# ### Step 1: Import a noiseless forecast

# %%
forecast_file = f"{cupixpath}/data/px_measurements/forecast/fcast_best_fit_arinyo_from_p1d_real_bf3_binned_out_px-zbins_4-thetabins_10_w_res_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=1.2)
iz = 0
z = forecast.z[iz]

# %%
with h5.File(forecast_file, 'r') as f:
    print(f['P_Z_AM']['z_0']['lya_params'])
    print(f.keys())
    for attr in f['cosmo_params'].attrs:
        print(attr, f['cosmo_params'].attrs[attr])

# %%
config = Config(cupixpath+"/data/px_measurements/forecast/fcast_best_fit_arinyo_from_p1d_config.yaml")

# %%
# these are not meant to be updated by the user, only including this to check
config.all_params

# %%
cosmo = cosmology.Cosmology(cosmo_params_dict=config.all_params['theory_params']['cosmo_params'])

# %%
theory = Theory(z=z, fid_cosmo=cosmo, config=config.all_params)

# %%
theory.lya_model.default_lya_params, theory.lya_model.default_lya_model

# %%
# old forecasts did not average over theta
like = Likelihood(data=forecast, theory=theory, iz=iz, 
                  config=config.all_params['like_params'])

# %%
like.plot_px()
