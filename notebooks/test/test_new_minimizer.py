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
# # Example use of the Minuit minimizer (new likelihood, new theory)

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.likelihood_parameter import LikelihoodParameter, like_parameter_by_name
from cupix.likelihood.new_likelihood import Likelihood
from cupix.likelihood.theory import Theory
from cupix.likelihood.new_minimizer import IminuitMinimizer
import cupix
cupixpath = cupix.__path__[0].rsplit('/', 1)[0]

# %% [markdown]
# ### Step 1: Import a noiseless forecast

# %%
forecast_file = f"{cupixpath}/data/px_measurements/forecast/forecast_ffcentral_real_binned_out_px-zbins_4-thetabins_9_w_res_noiseless_z0.hdf5"
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
# translate these to our Lya params
true_lya_params = {}
with h5.File(forecast_file) as f:
    true_lya_params['bias'], = -1.0 * f['arinyo_pars'].attrs['bias_0'],
    true_lya_params['beta'], = f['arinyo_pars'].attrs['beta_0'],
    true_lya_params['av'], = f['arinyo_pars'].attrs['av_0'],
    true_lya_params['bv'], = f['arinyo_pars'].attrs['bv_0'],
    true_lya_params['kp_Mpc'], = f['arinyo_pars'].attrs['kp_0'],
    true_lya_params['q1'], = f['arinyo_pars'].attrs['q1_0'],
    true_lya_params['q2'], = f['arinyo_pars'].attrs['q2_0'],
    true_lya_params['kv_Mpc'] = np.exp( np.log(f['arinyo_pars'].attrs['kvav_0']) / f['arinyo_pars'].attrs['av_0'] )
print(true_lya_params)

# %%
# use the true cosmology as fiducial
cosmo = cosmology.Cosmology(cosmo_params_dict=true_cosmo_params)

# %%
# use the true Lya parameters (Arinyo / bias / beta)
config = true_lya_params | {'verbose': True}
# make your life a bit harder by changing a bit the value of beta
wrong_beta = True
if wrong_beta:
    config['beta'] = 1.02*config['beta']
print(config)
theory = Theory(z=z, fid_cosmo=cosmo, config=config)

# %%
like = Likelihood(data=forecast, theory=theory, iz=iz, verbose=True)
#like = Likelihood(forecast, theory, z=z, verbose=False)

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
like_params = []
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-.13,
    max_value=-.1,
    ini_value=-.11,
    value =-.11
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.5,
    max_value=2.5,
    ini_value=1.5,
    value =1.5
    ))
like_params.append(LikelihoodParameter(
    name='q1',
    min_value=0,
    max_value=1,
    ini_value=0.3,
    value =0.3
    ))
for par in like_params:
    print(par.name)


# %%
mini = IminuitMinimizer(like, free_params=like_params, verbose=True)

# %%
mini.verbose=False
mini.like.verbose=False
mini.like.theory.verbose=False
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
#mini.plot_best_fit(multiply_by_k=True, every_other_theta=False, xlim=[-.01, .4], datalabel="Mock Data", show=True)

# %%
mini.get_best_fit_value("q1", return_hesse=True), mini.get_best_fit_value("bias", return_hesse=True)

# %%
#mini.plot_ellipses("bias", "q1", nsig=3, cube_values=False, 
#                   true_vals={'bias':true_lya_params['bias'], 'q1':true_lya_params['q1']}, 
#                   xrange=[-.12, -.11], yrange=[.2,.4])
