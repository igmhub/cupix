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

# %% [markdown]
# Minuit minimizer
#

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
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import scipy
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
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

# %%
ffemu = FF_emulator(z, fid_cosmo, cc)
ffemu.kp_Mpc = 1 # set pivot point

# %%
theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu

# %%
# MockData = Px_Lyacolore("binned_out_trucont_px-zbins_2-thetabins_18.hdf5", theta_min_cut_arcmin=11, kmax_cut_AA=1)
MockData = DESI_DR2("binned_out_trucont_px-zbins_2-thetabins_10_res.hdf5", theta_min_cut_arcmin=15, kmax_cut_AA=1)
# MockData = Px_Lyacolore("binned_out_trucont_px-zbins_2-thetabins_10_res.hdf5", kmax_cut_AA=1)

# %%
MockData.theta_min_A_arcmin, MockData.theta_max_A_arcmin

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values

like_params = []
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=0,
    ini_value=-0.115,
    value =-0.115,
    Gauss_priors_width=.05
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.0,
    max_value=2.0,    
    ini_value = 1.55,
    value=1.55,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='q1',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.1112,
    value=0.1112,
    Gauss_priors_width=0.111
    ))
like_params.append(LikelihoodParameter(
    name='kvav',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.0001**0.2694,
    value=0.0001**0.2694,
    Gauss_priors_width=0.0003**0.2694,
    ))
like_params.append(LikelihoodParameter(
    name='av',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.2694,
    value=0.2694,
    Gauss_priors_width=0.27
    ))
like_params.append(LikelihoodParameter(
    name='bv',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.0002,
    value=0.0002,
    Gauss_priors_width=0.0002
    ))
like_params.append(LikelihoodParameter(
    name='kp',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.5740,
    value=0.5740,
    Gauss_priors_width=0.5
    ))


# likelihood_params = []
# likelihood_params.append(LikelihoodParameter(
#     name='Delta2_p',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='n_p',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='mF',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='gamma',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='kF_Mpc',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='sigT_Mpc',
#     min_value=-1.0,
#     max_value=1.0,
#     ))

# %%
like = Likelihood(MockData, theory_AA, free_param_names=["bias", "beta"], iz_choice=0, like_params=like_params, verbose=False)

# %%
mini = IminuitMinimizer(like, verbose=False)

# %%
mini.minimize()

# %%
mini.best_fit_value("beta", return_hesse=True), mini.best_fit_value("bias", return_hesse=True)

# %%
prob = like.fit_probability(mini.minimizer.values)


# %%
prob

# %%
mini.plot_best_fit(multiply_by_k=True, every_other_theta=False, xlim=[-.01, .4], datalabel="Mock Data", show=True)

# %%
mini.minimizer.values

# %%
like.get_chi2(mini.minimizer.values)

# %%
mini.plot_ellipses("bias", "beta", nsig=2, cube_values=False)

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
