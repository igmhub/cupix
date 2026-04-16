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
# # Average theory prediction over theta bin

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.theory import Theory
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer

# %% [markdown]
# ### Read the Px from the stack of 50 mocks

# %%
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
# true continuum
true_fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
data = DESI_DR2(true_fname, kM_max_cut_AA=0.3, km_max_cut_AA=0.35, theta_min_cut_arcmin=5.0)

# %% [markdown]
# ### Start by fitting bias/beta from the stack of true-continuum mocks (one-z at a time)

# %%
# define fiducial cosmo
cosmo = cosmology.Cosmology()
# starting point for Lya bias parameters in mocks
default_lya_model = 'best_fit_arinyo_from_colore'

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
free_params = []
free_params.append(LikelihoodParameter(
    name='bias',
    min_value=-.5,
    max_value=-.02,
    ini_value=-0.15,
    value=-0.15
    ))
free_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.2,
    max_value=3.0,
    ini_value=1.5,
    value=1.5
    ))
for par in free_params:
    print(par.name)

# %%
iz=1
z=data.z[iz]
theory = Theory(z=z, fid_cosmo=cosmo, config={'verbose': True, 'default_lya_model': default_lya_model})
like = Likelihood(data=data, theory=theory, iz=iz, verbose=True)
mini = IminuitMinimizer(like, free_params=free_params, verbose=True)

# %%
# number of data points (per z bin)
Nz, Nt_A, Nk_M = data.Px_ZAM.shape
Ndp = Nt_A * Nk_M
# silence and minimize
mini.silence()
mini.minimize(compute_hesse=True)
chi2 = mini.get_best_fit_chi2()
best_fit = mini.get_best_fit_params()
print('best fit chi2 and params')
print(z, Ndp, chi2, best_fit)
mini.plot_ellipses(pname_x='bias', pname_y='beta', nsig=2)
label=""
for key, par in mini.get_best_fit_params().items():
    label += "{} = {:.3f}   ".format(key, par)
mini.plot_best_fit(multiply_by_k=False, theorylabel=label, datalabel='Stack (true continuum)')

# %%
params = mini.get_best_fit_params()
print(params)
model_px = mini.like.get_convolved_px(params=params)


# %%

# %%
def plot_theta_bin(it_A):
    k_M = data.k_M_centers_AA
    theta = data.theta_centers_arcmin[it_A]
    Px = data.Px_ZAM[iz][it_A]
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_A]))
    plt.errorbar(k_M, Px, sig_Px)
    plt.plot(k_M, model_px[it_A], label='center of bin')
    plt.legend()
    plt.axhline(y=0, ls=':', color='gray')
    plt.title(r"$\theta = {:.2f}'$".format(theta))


# %%
plot_theta_bin(it_A=2)

# %%
# array with discrete k values (inverse Angstroms)
k_m = data.k_m[iz]
# array with centers of the original theta bins (in arcmin)
theta_a = (data.theta_min_a_arcmin + data.theta_max_a_arcmin)/2.
# evaluate theory at these values (no rebinning, no convolution)
Px_Zam = theory.get_px_obs(theta_arc=theta_a, k_AA=k_m, params=params)


# %%
def compare_px_averaging(it_a, Nt=11):
    theta_min = data.theta_min_a_arcmin[it_a]
    theta_max = data.theta_max_a_arcmin[it_a]
    # evaluate theory at midle of bin
    theta_mid = 0.5*(theta_min+theta_max)
    print(theta_min, theta_mid, theta_max)
    mid_Px = theory.get_px_obs(theta_arc=theta_mid, k_AA=k_m, params=params)
    # evaluate theory at many intermediate points
    thetas = np.linspace(theta_min, theta_max, Nt)
    Px = theory.get_px_obs(theta_arc=thetas, k_AA=k_m, params=params)
    # compute weighted mean
    mean_Px = np.average(Px, axis=0, weights=thetas)
    # plot all of the above
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    for it in range(Nt):
        ax[0].plot(k_m, Px[it], color='gray', alpha=0.5)
    ax[0].plot(k_m, mid_Px, label=r"$\theta = {:.2f}'$".format(theta_mid))
    ax[0].plot(k_m, mean_Px, label="weighted average")
    ax[0].legend()
    ax[0].set_ylabel(r'$P_\times(\theta, k_\parallel) [\AA]$')
    # now residuals
    ax[1].plot(k_m, mid_Px / mid_Px, label=r"$\theta = {:.2f}'$".format(theta_mid))
    ax[1].plot(k_m, mean_Px / mid_Px, label="weighted average")
    ax[1].set_xlabel(r'$k_\parallel [1/ \AA]$')
    ax[1].set_ylabel(r'ratio of $P_\times(\theta, k_\parallel)$')


# %%
compare_px_averaging(it_a=0)

# %%
compare_px_averaging(it_a=3)

# %%
