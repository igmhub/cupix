# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %%
import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from lace.cosmo import camb_cosmo, fit_linP
import matplotlib.pyplot as plt
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta
from cupix.px_data.data_lyacolore import Px_Lyacolore
import scipy
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
import copy
# %load_ext autoreload
# %autoreload 2

# %%

# Load emulator
z = np.array([2.2])
omnuh2 = 0.0006
mnu = omnuh2 * 93.14
H0 = 67.36
omch2 = 0.12
ombh2 = 0.02237
As = 2.1e-9
ns = 0.9649
nrun = 0.0
w = -1.0
omk = 0
fid_cosmo = {
    'H0': H0,
    'omch2': omch2,
    'ombh2': ombh2,
    'mnu': mnu,
    'omk': omk,
    'As': As,
    'ns': ns,
    'nrun': nrun,
    'w': w
}
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)

ffemu = FF_emulator(z, fid_cosmo, cc)
ffemu.kp_Mpc = 1 # set pivot point

theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu


# %% [markdown]
# Load the data

# %%
cf_deltas_data = Px_Lyacolore("kbinned_out_px-nhp-zbins_4-thetabins_7.hdf5", kmax_cut_AA=1)
p1d_deltas_data = Px_Lyacolore("kbinned_out_P1D_deltas_px-nhp-zbins_4-thetabins_7.hdf5", kmax_cut_AA=1)

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values

like_params = []
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=1.0,
    ini_value=-0.115,
    value =-0.115,
    Gauss_priors_width=.1
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.0,
    max_value=2.0,    
    ini_value = 1.55,
    value=1.55,
    Gauss_priors_width=1
    ))
like_params.append(LikelihoodParameter(
    name='q1',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.1112,
    value=0.1112,
    Gauss_priors_width=0.3
    ))
like_params.append(LikelihoodParameter(
    name='kvav',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.0001**0.2694,
    value=0.0001**0.2694,
    Gauss_priors_width=0.0006**0.2694,
    ))
like_params.append(LikelihoodParameter(
    name='av',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.2694,
    value=0.2694,
    Gauss_priors_width=0.6
    ))
like_params.append(LikelihoodParameter(
    name='bv',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.0002,
    value=0.0002,
    Gauss_priors_width=0.0004
    ))
like_params.append(LikelihoodParameter(
    name='kp',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.5740,
    value=0.5740,
    Gauss_priors_width=1
    ))

like_p1d_deltas = Likelihood(p1d_deltas_data, theory_AA, free_param_names=["bias", "beta", "q1", "kvav", "av", "bv", "kp"], iz_choice=0, like_params=like_params, verbose=True)
like_cf_deltas = Likelihood(cf_deltas_data, theory_AA, free_param_names=["bias", "beta", "q1", "kvav", "av", "bv", "kp"], iz_choice=0, like_params=like_params, verbose=True)

# %%
mini_p1d_deltas = IminuitMinimizer(like_p1d_deltas, like_params, verbose=True)
mini_cf_deltas = IminuitMinimizer(like_cf_deltas, like_params, verbose=True)

# %%
mini_p1d_deltas.minimize()

# %%
mini_cf_deltas.minimize()

# %%
ini_values = []
final_values = []
hesse_errs = []
priors = []
for param in like_p1d_deltas.free_param_names:
    for param2 in like_params:
        if param == param2.name:
            ini_values.append(param2.ini_value)
            priors.append(param2.Gauss_priors_width)
            break

    for param3 in mini_p1d_deltas.minimizer.params:
        print(param3)
        if param == param3.name:
            final_values.append(mini_p1d_deltas.minimizer.params[param3.name].value)
            hesse_errs.append(mini_p1d_deltas.minimizer.params[param3.name].error)
            break

like_params_to_plot = copy.deepcopy(like_params)
for i, lp in enumerate(like_params_to_plot):
    if lp.name in like_p1d_deltas.free_param_names:
        index = like_p1d_deltas.free_param_names.index(lp.name)
        lp.value = final_values[index]

like_p1d_deltas.plot_px(0, like_params_to_plot, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None)


# %%
ini_values_cf = []
final_values_cf = []
hesse_errs_cf = []
priors_cf = []
for param in like_cf_deltas.free_param_names:
    for param2 in like_params:
        if param == param2.name:
            ini_values_cf.append(param2.ini_value)
            priors_cf.append(param2.Gauss_priors_width)
            break

    for param3 in mini_cf_deltas.minimizer.params:
        print(param3)
        if param == param3.name:
            final_values_cf.append(mini_cf_deltas.minimizer.params[param3.name].value)
            hesse_errs_cf.append(mini_cf_deltas.minimizer.params[param3.name].error)
            break

like_params_to_plot = copy.deepcopy(like_params)
for i, lp in enumerate(like_params_to_plot):
    if lp.name in like_cf_deltas.free_param_names:
        index = like_cf_deltas.free_param_names.index(lp.name)
        lp.value = final_values_cf[index]

like_cf_deltas.plot_px(0, like_params_to_plot, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None)


# %%
plt.plot(like_cf_deltas.free_param_names, ini_values_cf, 'o', label='Initial values')
plt.errorbar(like_cf_deltas.free_param_names, final_values_cf, yerr= hesse_errs_cf, fmt='x', label='Final values with Hesse errors, CF deltas')
plt.errorbar(like_p1d_deltas.free_param_names, final_values, yerr= hesse_errs, fmt='x', label='Final values with Hesse errors, P1D deltas')
# plot the priors as a shaded vertical band per parameter
for i, param in enumerate(like_cf_deltas.free_param_names):
    plt.fill_between([i-0.2, i+0.2], ini_values_cf[i]-priors[i], ini_values_cf[i]+priors_cf[i], color='gray', alpha=0.3, label='Priors' if i==0 else "")
    # plt.fill_between([i-0.2, i+0.2], ini_values[i]-priors[i], ini_values[i]+priors[i], color='gray', alpha=0.3, label='Priors' if i==0 else "")
plt.ylabel("Parameter values")
plt.xlabel("Parameters")
plt.legend(loc='lower left')
plt.ylim([-.3,.3])

# %%
ini_values = []
final_values = []
hesse_errs = []
priors = []
for param in like_cf_deltas.free_param_names:
    for param2 in like_params:
        if param == param2.name:
            ini_values.append(param2.ini_value)
            priors.append(param2.Gauss_priors_width)
            break

    for param3 in mini_cf_deltas.minimizer.params:
        print(param3)
        if param == param3.name:
            final_values.append(mini_cf_deltas.minimizer.params[param3.name].value)
            hesse_errs.append(mini_cf_deltas.minimizer.params[param3.name].error)
            break

like_params_to_plot = copy.deepcopy(like_params)
for i, lp in enumerate(like_params_to_plot):
    if lp.name in like_cf_deltas.free_param_names:
        index = like_cf_deltas.free_param_names.index(lp.name)
        lp.value = final_values[index]

like_cf_deltas.plot_px(0, like_params_to_plot, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None)


# %%
plt.plot(like_p1d_deltas.free_param_names, ini_values, 'o', label='Initial values')
plt.errorbar(like_p1d_deltas.free_param_names, final_values, yerr= hesse_errs, fmt='x', label='Final values with Hesse errors')
# plot the priors as a shaded vertical band per parameter
for i, param in enumerate(like_p1d_deltas.free_param_names):
    plt.fill_between([i-0.2, i+0.2], final_values[i]-priors[i], final_values[i]+priors[i], color='gray', alpha=0.3, label='Priors' if i==0 else "")
plt.ylabel("Parameter values")
plt.xlabel("Parameters")
plt.legend()

# %%
plt.plot(like_cf_deltas.free_param_names, ini_values_cf, 'o', label='Initial values')
plt.errorbar(like_cf_deltas.free_param_names, final_values_cf, yerr= hesse_errs_cf, fmt='x', label='Final values with Hesse errors')
# plot the priors as a shaded vertical band per parameter
for i, param in enumerate(like_cf_deltas.free_param_names):
    plt.fill_between([i-0.2, i+0.2], final_values_cf[i]-priors[i], final_values_cf[i]+priors[i], color='gray', alpha=0.3, label='Priors' if i==0 else "")
plt.ylabel("Parameter values")
plt.xlabel("Parameters")
plt.legend()

# %%
mini_cf_deltas.minimize()
