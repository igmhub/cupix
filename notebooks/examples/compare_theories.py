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
# # Plot theory predictions with different components turned ON / OFF

# %%
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.utils.utils import get_path_repo
cupixpath = get_path_repo('cupix')

# %% [markdown]
# ## Start by reproducing the theory in a fake data vector (forecast)

# %%
forecast_file = f"{cupixpath}/data/px_measurements/forecast/fcast_best_fit_arinyo_from_p1d_real_bf3_binned_out_px-zbins_4-thetabins_10_w_res_noiseless.hdf5"
data = DESI_DR2(forecast_file)
iz = 1
z = data.z[iz]
print(iz, z)

# %%
true_cosmo_params = {}
with h5.File(forecast_file) as f:
    for key in f['cosmo_params'].attrs.keys():
        true_cosmo_params[key] = f['cosmo_params'].attrs[key]
print('cosmo params used in forecast', true_cosmo_params)

# %%
with h5.File(forecast_file) as f:
    true_lya_params = {}        
    if 'lya_params' in f['P_Z_AM'][f'z_{iz}'].keys():
        lya_params = f['P_Z_AM'][f'z_{iz}']['lya_params'].attrs
        for par in lya_params:
            true_lya_params[par] = lya_params[par]
print('Lya params used in forecast', true_lya_params)

# %%
# use the true cosmology as fiducial
cosmo = cosmology.Cosmology(cosmo_params_dict=true_cosmo_params)

# %%
# use the true Lya parameters (Arinyo / bias / beta)
config = true_lya_params | {'verbose': False}
print('theory config =', config)
theory = Theory(z=z, fid_cosmo=cosmo, config=config)

# %%
# ANDREU: for old forecasts, there should be no theta averaging in the likelihood
N_theta_average=1
like = Likelihood(data=data, theory=theory, iz=iz, config={'verbose': False, 'N_theta_average': N_theta_average})
print('chi2 = ', like.get_chi2())

# %%
datalabel = 'Forecast Px'
theorylabel = 'True theory'
like.plot_px(multiply_by_k=False, every_other_theta=True, residual_to_theory=True,
            datalabel=datalabel, theorylabel=theorylabel)

# %% [markdown]
# ## Explore the role of the Arinyo parameters

# %%
true_lya_params

# %% [markdown]
# ### Pressure only (no non-linear growth)
#
# We can see below that the non-linear growth (Arinyo model) is important for theta < 15 arcmin

# %%
pressure_only_params = {'q1': 0, 'q2': 0}
pressure_only_label = 'Pressure only (q1 = q2 = 0)'

# %%
# first make a plot with residuals wrt to the theory
like.plot_px(multiply_by_k=False, every_other_theta=False, residual_to_theory=True,
             datalabel=datalabel, ylim2=[-0.1,  0.1],
             params={}, theorylabel=theorylabel,
             extra_params=pressure_only_params, extra_label=pressure_only_label)

# %%
# now a plot with residuals wrt to DR2 errorbars
like.plot_px(multiply_by_k=False, every_other_theta=False, residual_to_theory=False,
             datalabel=datalabel, ylim2=[-1, 1],
             params={}, theorylabel=theorylabel,
             extra_params=pressure_only_params, extra_label=pressure_only_label)

# %% [markdown]
# ### Changes in pressure
#
# We can see that the effect of pressure is very small even on scales of theta = 1 arcmin

# %%
low_pressure_params = {'kp_Mpc': 100.0}
low_pressure_label = 'Low pressure (kp_Mpc = 100)'
high_pressure_params = {'kp_Mpc': 5.0}
high_pressure_label = 'High pressure (kp_Mpc = 5)'

# %%
# first make a plot with residuals wrt to the theory
like.plot_px(multiply_by_k=False, every_other_theta=False, residual_to_theory=True,
             datalabel=datalabel, ylim2=[-0.1,  0.1],
             params=low_pressure_params, theorylabel=low_pressure_label,
             extra_params=high_pressure_params, extra_label=high_pressure_label)

# %%
# first make a plot with residuals wrt to the theory
like.plot_px(multiply_by_k=False, every_other_theta=False, residual_to_theory=False,
             datalabel=datalabel, ylim2=[-1,  1],
             params=low_pressure_params, theorylabel=low_pressure_label,
             extra_params=high_pressure_params, extra_label=high_pressure_label)

# %% [markdown]
# ### Change of individual Arinyo parameters
#
# We will use priors from ForestFlow based on DESI DR1 P1D
#
# We can see that impact of bias / beta is always large, as expected, while the others are very minor.

# %%
from forestflow import priors
prior_info = priors.get_arinyo_priors(z=z, tag='DESI_DR1_P1D')

# %%
pnames = list(prior_info['mean'].keys())
for pname in pnames:
    mean = prior_info['mean'][pname]
    std = prior_info['std'][pname]
    print('{} = {:.3f} +/- {:.3f}'.format(pname, mean, std)) 

# %%
for pname in pnames:
    percen_5 = prior_info['percen_5'][pname]
    percen_95 = prior_info['percen_95'][pname]
    print('{:.3f} < {} < {:.3f}'.format(percen_5, pname, percen_95)) 


# %%
def compare_param(pname):
    low_value = prior_info['percen_5'][pname]
    high_value = prior_info['percen_95'][pname]
    low_params = {pname: low_value}
    high_params = {pname: high_value}
    low_label = '{} = {:.4f}'.format(pname, low_value)
    high_label = '{} = {:.4f}'.format(pname, high_value)
    print('{:.4f} < {} < {:.4f}'.format(low_value, pname, high_value))

    low_chi2 = like.get_chi2(params=low_params)
    high_chi2 = like.get_chi2(params=high_params)
    print('Delta chi2 (low) = {:.2f} , Delta chi2 (high) = {:.2f}'.format(low_chi2, high_chi2))
    
    # first make a plot with residuals wrt to the theory
    like.plot_px(multiply_by_k=False, every_other_theta=False, residual_to_theory=True,
             datalabel=datalabel, ylim2=[-0.2, 0.2],
             params=low_params, theorylabel=low_label,
             extra_params=high_params, extra_label=high_label)

    # then make a plot with residuals wrt to the DR2 errors
    like.plot_px(multiply_by_k=False, every_other_theta=False, residual_to_theory=False,
             datalabel=datalabel, ylim2=[-1, 1],
             params=low_params, theorylabel=low_label,
             extra_params=high_params, extra_label=high_label)


# %%
compare_param('q1')

# %%
compare_param('q2')

# %%
compare_param('av')

# %% [markdown]
# ## Study the impact of contaminants

# %%
print(theory.cont_model.default_hcd_params)
print(theory.cont_model.default_metal_params)
print(theory.cont_model.default_sky_params)
print(theory.cont_model.default_continuum_params)


# %%
def compare_contaminant(label):
    new_params = {}
    theorylabel = 'no contamination'
    if label=='continuum':
        new_config = config | {'include_continuum': True}
        params = {'kC_Mpc': 1e-10}
        new_label = '+ continuum distortion'
    elif label=='sky':
        new_config = config | {'include_sky': True}
        params = {'b_noise_Mpc': 0.0}
        new_label = '+ sky residuals'
    elif label=='metal':
        new_config = config | {'include_metal': True}
        params = {'b_X': 0.0}
        new_label = '+ metals'
    elif label=='hcd':
        new_config = config | {'include_hcd': True}
        params = {'b_H': 0.0}
        new_label = '+ HCDs'

    # setup theory with contaminant
    theory = Theory(z=z, fid_cosmo=cosmo, config=new_config)
    N_theta_average=1
    like = Likelihood(data=data, theory=theory, iz=iz, 
                      config={'verbose': False, 'N_theta_average': N_theta_average})

    # uncontaminated chi2
    good_chi2 = like.get_chi2(params=params)
    # contaminated chi2
    bad_chi2 = like.get_chi2(params=new_params)
    print('{}: good chi2 = {:.2f}, bad_chi2 = {:.2f}'.format(label, good_chi2, bad_chi2))

    # first make a plot with residuals wrt to the theory
    like.plot_px(multiply_by_k=False, every_other_theta=True, residual_to_theory=True,
             datalabel=datalabel, ylim2=[-0.2, 0.2],
             params=params, theorylabel=theorylabel,
             extra_params=new_params, extra_label=new_label)

    # now make a plot with residuals wrt to the DR2 errors
    like.plot_px(multiply_by_k=False, every_other_theta=True, residual_to_theory=False,
             datalabel=datalabel, ylim2=[-1, 1],
             params=params, theorylabel=theorylabel,
             extra_params=new_params, extra_label=new_label)


# %% [markdown]
# #### Continuum fitting

# %%
compare_contaminant(label='continuum')

# %% [markdown]
# #### Sky residuals

# %%
compare_contaminant(label='sky')

# %% [markdown]
# #### Metals (silicon III)

# %%
compare_contaminant(label='metal')

# %% [markdown]
# #### HCDs

# %%
compare_contaminant(label='hcd')

# %%

# %%
