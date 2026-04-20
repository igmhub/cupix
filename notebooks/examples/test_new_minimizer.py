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
# # Example use of the posterior minimizer

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
import cupix
cupixpath = cupix.__path__[0].rsplit('/', 1)[0]

# %% [markdown]
# ### Step 1: Import a noiseless forecast

# %%
forecast_file = f"{cupixpath}/data/px_measurements/forecast/fcast_best_fit_arinyo_from_p1d_real_bf3_binned_out_px-zbins_4-thetabins_10_w_res_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=1.2)
iz = 0
z = forecast.z[iz]

# %%
true_cosmo_params = {}
with h5.File(forecast_file) as f:
    for key in f['cosmo_params'].attrs.keys():
        true_cosmo_params[key] = f['cosmo_params'].attrs[key]
print(true_cosmo_params)

# %%
with h5.File(forecast_file) as f:
    true_lya_params = {}        
    if 'lya_params' in f['P_Z_AM'][f'z_{iz}'].keys():
        lya_params = f['P_Z_AM'][f'z_{iz}']['lya_params'].attrs
        for par in lya_params:
            true_lya_params[par] = lya_params[par]
print(true_lya_params)

# %%
# use the true cosmology as fiducial
cosmo = cosmology.Cosmology(cosmo_params_dict=true_cosmo_params)

# %%
# use the true Lya parameters (Arinyo / bias / beta)
config = true_lya_params | {'verbose': True}
# make your life a bit harder by changing a bit the value of beta
wrong_beta = False
if wrong_beta:
    config['beta'] = 1.02*config['beta']
print(config)
theory = Theory(z=z, fid_cosmo=cosmo, config=config)

# %%
# old forecasts did not average over theta
N_theta_average=1
like = Likelihood(data=forecast, theory=theory, iz=iz, 
                  config={'verbose':True, 'N_theta_average':N_theta_average})

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
bias = FreeParameter(
    name='bias',
    min_value=-0.5,
    max_value=-0.01,
    ini_value=-0.15,
    delta=0.01,   
)
q1 = FreeParameter(
    name='q1',
    min_value=0.0,
    max_value=5.0,
    ini_value=0.5,
    delta=0.02,
)
free_params = [bias, q1]
for par in free_params:
    print(par.name, par.ini_value)

# %%
post = Posterior(like, free_params, config={'verbose': True})
mini = Minimizer(post, config={'verbose':True})

# %%
mini.silence()
mini.minimize()

# %%
mini.minimizer

# %%
true_lya_params

# %%
mini.get_best_fit_params()

# %%
chi2 = mini.get_best_fit_chi2()
print("chi2 of fit", chi2)

# %%
mini.plot_best_fit(multiply_by_k=False, every_other_theta=True, xlim=[-.01, .4], datalabel="Mock Data", show=True)

# %%
mini.get_best_fit_value("q1", return_hesse=True), mini.get_best_fit_value("bias", return_hesse=True)


# %%
def plot_ellipses(pname_x, pname_y):
    mini.plot_ellipses(pname_x, pname_y, nsig=3, true_vals={pname_x: true_lya_params[pname_x], pname_y: true_lya_params[pname_y]}) 


# %%
plot_ellipses('bias','q1')

# %%
