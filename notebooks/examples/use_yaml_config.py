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
from cupix.sampling.minimize_posterior import Minimizer
from cupix.utils.utils import get_path_repo
from cupix.likelihood.config import Config
cupixpath = get_path_repo('cupix')

# %% [markdown]
# ### Step 1: Import a noiseless forecast

# %%
config = Config(cupixpath+"/data/px_measurements/forecast/fcast_best_fit_arinyo_from_p1d_config.yaml")

# %%
config.print_all()

# %%
# we should eventually update data class to accept a config dictinoary
forecast = DESI_DR2(config.data_config['data_file'], kM_max_cut_AA=config.data_config['kM_max_cut_AA'])
iz = config.theory_config['iz']
z = forecast.z[iz]

# %%
print(z)

# %%
# If you want to check the true values from the forecast file
# with h5.File(config.data_config['data_file'], 'r') as f:
#     print(f['P_Z_AM']['z_0']['lya_params'])
#     print(f.keys())
#     for attr in f['cosmo_params'].attrs:
#         print(attr, f['cosmo_params'].attrs[attr])

# %%
cosmo = cosmology.Cosmology(cosmo_params_dict=config.cosmo_config)

# %%
theory = Theory(z=z, fid_cosmo=cosmo, config=config.theory_config)

# %%
# old forecasts did not average over theta
like = Likelihood(data=forecast, theory=theory, iz=iz, 
                  config=config.like_config)

# %%
like.plot_px()

# %%
