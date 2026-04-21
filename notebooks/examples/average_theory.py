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
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.posterior import Posterior
from cupix.likelihood.free_parameter import FreeParameter
from cupix.likelihood.minimize_posterior import Minimizer

# %% [markdown]
# ### Read the Px from the stack of 50 mocks

# %%
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
# true continuum
true_fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
data = DESI_DR2(true_fname, kM_max_cut_AA=0.5, km_max_cut_AA=0.55, theta_min_cut_arcmin=10.0)

# %% [markdown]
# ### Start by fitting bias/beta from the stack of true-continuum mocks (one-z at a time)

# %%
# define fiducial cosmo
cosmo = cosmology.Cosmology()
# starting point for Lya bias parameters in mocks
default_lya_model = 'best_fit_arinyo_from_colore'

# %%
ini_bias=-0.14
ini_beta=1.5

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
bias = FreeParameter(
    name='bias',
    min_value=-0.5,
    max_value=-0.01,
    ini_value=ini_bias,
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
    gauss_prior_mean=ini_beta,
    gauss_prior_width=0.2,    
)
free_params = [bias, beta]
for par in free_params:
    print(par.name, par.ini_value)

# %%
iz=1
z=data.z[iz]
theory = Theory(z=z, fid_cosmo=cosmo, config={'verbose': True, 'default_lya_model': default_lya_model})
# first a likelihood without theta averaging
like = Likelihood(data=data, theory=theory, iz=iz, config={'N_theta_average':1})
post = Posterior(like, free_params, config={'verbose': True})
mini = Minimizer(post, config={'verbose':True})

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
model_px = mini.post.like.get_convolved_px(params=params)


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
    ax[0].axhline(y=0.0, ls=':', color='gray')
    # now residuals
    ax[1].plot(k_m, mid_Px / mid_Px, label=r"$\theta = {:.2f}'$".format(theta_mid))
    ax[1].plot(k_m, mean_Px / mid_Px, label="weighted average")
    ax[1].set_xlabel(r'$k_\parallel [1/ \AA]$')
    ax[1].set_ylabel(r'ratio of $P_\times(\theta, k_\parallel)$')
    ax[1].set_ylim([0.95, 1.05])
    ax[1].axhline(y=0.99, ls=':', color='gray')
    ax[1].axhline(y=1.01, ls=':', color='gray')


# %%
compare_px_averaging(it_a=0)

# %%
compare_px_averaging(it_a=7)

# %% [markdown]
# ## Modify get_convolved_px to do the theta averaging

# %%
iz = like.iz
k_m = like.data.k_m[like.iz]
theta_a_mid = (like.data.theta_min_a_arcmin + like.data.theta_max_a_arcmin)/2.
Nk_m = len(k_m)
Nt_a = len(theta_a_mid)
print(Nk_m, Nt_a)

# %%
Px_Zam_mid = like.theory.get_px_obs(theta_arc=theta_a_mid, k_AA=k_m)


# %%
def compute_fine_thetas(it_a, N):
    theta_min=like.data.theta_min_a_arcmin[it_a]
    theta_max=like.data.theta_max_a_arcmin[it_a]
    #print(it_a, ' bin, with range', theta_min, '< theta <', theta_max)
    dtheta=(theta_max-theta_min)/N
    #print('dtheta = ', dtheta)
    thetas = np.linspace(theta_min + 0.5*dtheta, theta_max-0.5*dtheta, N)
    return thetas


# %%
# average factor 
N=10
compute_fine_thetas(it_a=2, N=N)

# %%
fine_thetas = np.concatenate( [ compute_fine_thetas(it_a=it_a, N=N) for it_a in range(Nt_a) ] , axis=0 )
fine_thetas.shape

# %%
Px_Zam_fine = like.theory.get_px_obs(theta_arc=fine_thetas, k_AA=k_m)
Px_Zam_fine.shape

# %%
# option one: compute the mean within the bin
M = Px_Zam_fine.shape[0] // N
L = Px_Zam_fine.shape[1]
print(N, M, L)
Px_Zam_mean = Px_Zam_fine.reshape(M, N, L).mean(axis=1)
print(Px_Zam_mean.shape)

# %%
# option two: weighted average, using theta as weights
weights = fine_thetas
Px_Zam_w = (
    (Px_Zam_fine.reshape(-1, N, Px_Zam_fine.shape[1]) *
     weights.reshape(-1, N)[:, :, None]).sum(axis=1)
    / weights.reshape(-1, N).sum(axis=1)[:, None]
)
print(Px_Zam_w.shape)


# %%
def compare_px_averaging(it_a):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax[0].plot(k_m, Px_Zam_mid[it_a], label=r"$\theta = {:.2f}'$".format(theta_a_mid[it_a]))
    ax[0].plot(k_m, Px_Zam_mean[it_a], label="mean")
    ax[0].plot(k_m, Px_Zam_w[it_a], label="weighted average")
    ax[0].legend()
    ax[0].set_ylabel(r'$P_\times(\theta, k_\parallel) [\AA]$')
    ax[0].axhline(y=0.0, ls=':', color='gray')
    # now residuals
    ax[1].plot(k_m, Px_Zam_mid[it_a] / Px_Zam_mid[it_a])
    ax[1].plot(k_m, Px_Zam_mean[it_a] / Px_Zam_mid[it_a])
    ax[1].plot(k_m, Px_Zam_w[it_a] / Px_Zam_mid[it_a])
    ax[1].set_xlabel(r'$k_\parallel [1/ \AA]$')
    ax[1].set_ylabel(r'ratio of $P_\times(\theta, k_\parallel)$')
    ax[1].axhline(y=1.01, ls=':', color='gray')
    ax[1].axhline(y=0.99, ls=':', color='gray')
    ax[1].set_ylim([0.95, 1.05])


# %%
compare_px_averaging(it_a=0)

# %%
compare_px_averaging(it_a=2)

# %%
compare_px_averaging(it_a=5)

# %% [markdown]
# ### Now use the new code in the likelihood

# %%
like.N_theta_average=1
px_model_mid = like.get_convolved_px()

# %%
like.N_theta_average=10
px_model_ave = like.get_convolved_px()


# %%
def compare_px_averaging(it_A):
    k_M = data.k_M_centers_AA
    theta = data.theta_centers_arcmin[it_A]
    Px = data.Px_ZAM[like.iz][it_A]
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[like.iz][it_A]))
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax[0].errorbar(k_M, Px, sig_Px)
    ax[0].plot(k_M, px_model_mid[it_A], label=r"$\theta = {:.2f}'$".format(theta))
    ax[0].plot(k_M, px_model_ave[it_A], label='theta averaged')
    ax[0].axhline(y=0, ls=':', color='gray')
    ax[0].legend()
    ax[0].set_ylabel(r'$P_\times(\theta, k_\parallel) [\AA]$')
    # now residuals
    ax[1].plot(k_M, px_model_mid[it_A] / px_model_ave[it_A])
    ax[1].plot(k_M, px_model_ave[it_A] / px_model_ave[it_A])
    ax[1].set_xlabel(r'$k_\parallel [1/ \AA]$')
    ax[1].set_ylabel(r'ratio of $P_\times(\theta, k_\parallel)$')
    ax[1].axhline(y=1.01, ls=':', color='gray')
    ax[1].axhline(y=0.99, ls=':', color='gray')
    ax[1].set_ylim([0.95, 1.05])


# %%
compare_px_averaging(it_A=3)

# %%
compare_px_averaging(it_A=7)

# %% [markdown]
# ### Compare timing and best fits

# %%
for N in [1, 10, 100]:
    print(N, 'theta values per bin')
    like = Likelihood(data=data, theory=theory, iz=iz, config={'N_theta_average':N})
    # %timeit like.get_chi2()

# %%
for N in [1, 10, 100]:
    print('------- {} theta values per bin ---------'.format(N))
    like = Likelihood(data=data, theory=theory, iz=iz, config={'N_theta_average':N})
    post = Posterior(like, free_params, config={'verbose': False})
    mini = Minimizer(post, config={'verbose':False})
    mini.silence()
    mini.minimize(compute_hesse=False)
    chi2 = mini.get_best_fit_chi2()
    best_fit = mini.get_best_fit_params()
    print('best fit chi2 and params')
    print(chi2, best_fit)

# %%
a=3

# %%
