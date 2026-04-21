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
# # Test the new posterior class

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
from cupix.likelihood.free_parameter import FreeParameter
from cupix.likelihood.posterior import Posterior
from cupix.likelihood.minimize_posterior import Minimizer
import cupix
cupixpath = cupix.__path__[0].rsplit('/', 1)[0]

# %% [markdown]
# ### Step 1: Import a noiseless forecast

# %%
forecast_file = f"{cupixpath}/data/px_measurements/forecast/fcast_best_fit_arinyo_from_p1d_real_bf3_binned_out_px-zbins_4-thetabins_10_w_res_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=1.2, theta_min_cut_arcmin=20.0)
iz = 0
z = forecast.z[iz]

# %% [markdown]
# ### Step 2: Setup theory / likelihood (using true parameters from forecast)

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
print(config)
theory = Theory(z=z, fid_cosmo=cosmo, config=config)

# %%
like = Likelihood(data=forecast, theory=theory, iz=iz, config={'verbose':True})

# %%
like.get_chi2()

# %% [markdown]
# ### Step 3: Setup free parameters and posterior

# %%
# start a bit off
ini_bias = 1.05 * true_lya_params['bias']
ini_beta = 0.9 * true_lya_params['beta']

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
bias = FreeParameter(
    name='bias',
    min_value=-0.5,
    max_value=-0.01,
    ini_value=ini_bias,
    true_value=true_lya_params['bias'],
    delta=0.01,
    gauss_prior_mean=ini_bias,
    gauss_prior_width=0.05,    
)
beta = FreeParameter(
    name='beta',
    min_value=0.1,
    max_value=5.0,
    ini_value=ini_beta,
    delta=0.1,
    true_value=true_lya_params['beta'],
    gauss_prior_mean=ini_beta,
    gauss_prior_width=0.2,    
)


# %%
free_params = [bias, beta]
for par in free_params:
    print(par.name, par.ini_value, par.true_value)

# %%
post = Posterior(like, free_params, config={'verbose': True})

# %%
test = post.get_log_posterior()

# %%
post.get_log_prior()

# %% [markdown]
# ### Step 4: Setup posterior minimizer

# %%
mini = Minimizer(post, config={'verbose':True})

# %%
true_params = {'bias': bias.true_value, 'beta': beta.true_value}
true_chi2 = post.like.get_chi2(params=true_params)
true_post = post.get_log_posterior(params=true_params)
print(true_chi2, true_post, -0.5*true_chi2)

# %%
mini.silence()
betas = np.linspace(beta.true_value-0.1, beta.true_value+0.1, 11)
chi2 = [post.like.get_chi2(params={'beta': beta}) for beta in betas]
plt.plot(betas, chi2)
plt.axvline(x=beta.true_value, color='gray', ls=':')

# %%
betas = np.linspace(beta.true_value-0.1, beta.true_value+0.1, 11)
log_post = [post.get_log_posterior(params={'bias': bias.true_value, 'beta': val}) for val in betas]
plt.plot(betas, log_post)
plt.axvline(x=beta.true_value, color='gray', ls=':')

# %%
mini.silence()
mini.minimize()

# %%
best_params = mini.get_best_fit_params()
print(best_params)
best_chi2 = like.get_chi2(params=best_params)
print(best_chi2)

# %%
# these should not agree perfectly, since our prior is a bit off
mini.plot_ellipses('bias','beta', true_vals=true_params)

# %%
