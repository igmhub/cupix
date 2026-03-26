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
# Minuit minimizer
#

# %%
import numpy as np
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.lya_theory import Theory
from cupix.likelihood.forestflow_emu import FF_emulator
from lace.cosmo import camb_cosmo
import matplotlib.pyplot as plt
from cupix.likelihood.likelihood_parameter import LikelihoodParameter, par_index, dict_from_likeparam
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import h5py as h5
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
import cupix
import pandas as pd
cupixpath = cupix.__path__[0].rsplit('/', 1)[0]
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### Step 1: Import a noiseless forecast

# %%
forecast_file = f"{cupixpath}/data/px_measurements/forecast/forecast_ffcentral_real_binned_out_px-zbins_4-thetabins_9_w_res_noiseless_z0.hdf5"
forecast = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=1.2)
param_mode = 'arinyo'
##### Get the truth from forecast file ######
forecast = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=1.2)
# get default theory that was used for forecast
with h5.File(forecast_file, 'r') as f:
    default_theory_label = f['metadata'].attrs['true_lya_theory']
print(f"Default theory label: {default_theory_label}")
zs = forecast.z
iz_choice = 0
z_choice  = zs[iz_choice]
cosmo = {}
truth_params = {}
with h5.File(forecast_file) as f:
    for cosmo_par in f['cosmo_params'].attrs.keys():
        cosmo[cosmo_par] = f['cosmo_params'].attrs[cosmo_par]
    if param_mode=='igm':
        for like_par in f['like_params'].attrs.keys():
            truth_params[like_par] = f['like_params'].attrs[like_par]
    elif param_mode=='arinyo':
        for arinyo_par in f['arinyo_pars'].attrs.keys():
            truth_params[arinyo_par] = f['arinyo_pars'].attrs[arinyo_par]
print("Passing forecast cosmology to theory object:", cosmo)
theory = Theory(zs, cosmo_dict=cosmo, default_lya_theory='best_fit_arinyo_from_p1d', emulator_label="forestflow_emu", verbose=True)
# check the parameters
print("Truth params are:")
for p, val in truth_params.items():
    print(p, val)

# %%
like = Likelihood(forecast, theory, z=z_choice, verbose=True)

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
arinyo_par_names = theory.arinyo_par_names

like_params = []
like_params.append(LikelihoodParameter(
    name='bias_0',
    min_value=-.13,
    max_value=-.1,
    ini_value=-0.11,
    value =truth_params['bias_0']
    ))

like_params.append(LikelihoodParameter(
    name='q1_0',
    min_value=0,
    max_value=1,
    ini_value=.2,
    value =truth_params['q1_0']
    ))

# like_params.append(LikelihoodParameter(
#     name='beta_0',
#     min_value=1.3,
#     max_value=2.0,    
#     ini_value = 1.5,
#     value=truth_params['beta_0'],
#     # Gauss_priors_width=.5
#     ))

def update_likepar_list(like_params, par_names):
    for par in par_names:
        like_params.append(LikelihoodParameter(
    name=par,
    value = truth_params[par]
    ))
        print(par, truth_params[par])
    return like_params

truth_pars_to_add = []
for par in arinyo_par_names:
    added=False
    for lp in like_params:
        if lp.name == par + "_0":
            added = True
    if not added:
        truth_pars_to_add.append(par+"_0")
print(truth_pars_to_add)


like_params = update_likepar_list(like_params, truth_pars_to_add)


# %%
mini = IminuitMinimizer(like, like_params, ['bias_0', 'q1_0'], verbose=True)

# %%
mini.minimize()

# %%
mini.plot_best_fit(multiply_by_k=True, every_other_theta=False, xlim=[-.01, .4], datalabel="Mock Data", show=True)

# %%
mini.best_fit_value("q1_0", return_hesse=True), mini.best_fit_value("bias_0", return_hesse=True)

# %%
prob = mini.fit_probability()
print("Probability of fit", prob)
chi2 = mini.chi2()
print("chi2 of fit", chi2)


# %%
from cupix.likelihood.likelihood_parameter import like_parameter_by_name
mini.plot_ellipses("bias_0", "q1_0", nsig=3, cube_values=False, true_vals={'bias_0':-1*like_parameter_by_name(like_params, 'bias_0').value, 'q1_0':like_parameter_by_name(like_params, 'q1_0').value}, xrange=[-.122, -.11], yrange=[.2,.4])

# %%
nparam = np.arange(5)+1
plt.plot(nparam, [27/60., 69/60., 120/60., 3+16/60., 4+21/60.], 'o', color='blue', label='measured')
power = 1.43
baseline = 27/60.
nparam_test = np.arange(7)+1
plt.plot(nparam_test, baseline*nparam_test**power, '-', color='red', label=r'$n_\mathrm{par}^{1.43}$')
plt.plot(7, 7+16/60., 'o', color='blue')
plt.legend()
plt.ylabel("Time [min]")
plt.xlabel("Number of Arinyo model parameters")
plt.title("iMinuit time for minimization")

# %%
final_values = np.asarray([mini.best_fit_value(pname) for pname in like.free_param_names])
ini_values   = np.asarray([p.ini_value for p in like_params if p.name in like.free_param_names])
priors       = np.asarray([p.Gauss_priors_width for p in like_params if p.name in like.free_param_names])
hesse_errs   = np.asarray([mini.best_fit_value(pname, return_hesse=True)[1] for pname in like.free_param_names])

plt.plot(like.free_param_names, ini_values, 'o', label='Initial values')
plt.errorbar(like.free_param_names, final_values, yerr= hesse_errs, fmt='x', label='Final values with Hesse errors')
# plot the priors as a shaded vertical band per parameter
for i, param in enumerate(like.free_param_names):
    plt.fill_between([i-0.2, i+0.2], final_values[i]-priors[i], final_values[i]+priors[i], color='gray', alpha=0.3, label='Priors' if i==0 else "")
plt.ylabel("Parameter values")
plt.xlabel("Parameters")
plt.legend()

# %%
plt.plot(like.free_param_names, ini_values/ini_values, 'o', label='Initial values')
plt.errorbar(like.free_param_names, final_values/ini_values, yerr= np.abs(hesse_errs/ini_values), fmt='x', label='Final values with Hesse errors')
# plot the priors as a shaded vertical band per parameter
for i, param in enumerate(like.free_param_names):
    plt.fill_between([i-0.2, i+0.2], (final_values[i]-priors[i])/ini_values[i], (final_values[i]+priors[i])/ini_values[i], color='gray', alpha=0.3, label='Priors' if i==0 else "")
plt.ylabel("Parameter values/ initial values")
plt.xlabel("Parameters")
plt.legend()

# %% [markdown]
# There is a problem with the determinant of the covariance matrix for these mocks when using all k. If I cut the kmax to 1, it is fine (see below).

# %%
# test the log det
icov_ZAM = np.linalg.inv(MockData.cov_ZAM[0,0,:,:])
# plt.imshow(MockData.cov_ZAM[0,0,:,:])
plt.imshow(icov_ZAM)
plt.colorbar()
print(np.linalg.det(icov_ZAM))
print(np.abs(1 / np.linalg.det(icov_ZAM)))
log_det_cov = np.log(
                    np.abs(1 / np.linalg.det(icov_ZAM))
                )


# %%
