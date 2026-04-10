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
#     name: cupix
# ---

# %% [markdown]
# # Use iminuit to fit Px from DESI DR2

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
#from cupix.likelihood.likelihood_parameter import LikelihoodParameter, like_parameter_by_name
from cupix.likelihood.new_likelihood import Likelihood
from cupix.likelihood.theory import Theory
from cupix.likelihood.new_minimizer import IminuitMinimizer

# %% [markdown]
# ## Step 1: Read the data from DESI DR2 and plot it

# %%
fname = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5"
data = DESI_DR2(fname, kM_max_cut_AA=1.0, km_max_cut_AA=1.2, theta_min_cut_arcmin=1.5)

# %%
# get the central value of each redshift bin, of length Nz
zs = data.z
# get a 1D array of central values of the measured k bins, of length Nk_M
k_M = data.k_M_centers_AA
# get two 1D arrays with the edges of each theta bin, of length Nt_A each
theta_A_min = data.theta_min_A_arcmin
theta_A_max = data.theta_max_A_arcmin

# %%
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = data.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_m={Nk_m}")
# rebinned values
Nz, Nt_A, Nk_M = data.Px_ZAM.shape
print(f"rebinned values: Nz={Nz}, Nt_A={Nt_A}, Nk_M={Nk_M}")


# %%
def plot_theta_bin(iz, it_M):
    label = r"${:.2f}' < \theta < {:.2f}'$".format(theta_A_min[it_M], theta_A_max[it_M])
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it_M]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_M]))
#    print(len(k_M), len(Px), len(sig_Px))
    plt.errorbar(k_M, Px, sig_Px, label=label)


# %%
def plot_z_bin(iz, its_M):
    for it_M in its_M:
        plot_theta_bin(iz=iz, it_M=it_M)
    plt.title('DESI DR2 at z={:.1f}'.format(zs[iz]))
    plt.legend()
    plt.xlabel(r'$k_\parallel$ [1/A]')
    plt.ylabel(r'$P_\times(\theta, k_\parallel)$ [A]')


# %%
for iz in range(4):
    plt.figure(figsize=[8,3])
    plot_z_bin(iz=iz, its_M=range(Nt_A))

# %% [markdown]
# ## Step 2: setup theory objects, with and without contaminants (one per z)

# %%
# define fiducial cosmo
cosmo = cosmology.Cosmology()

# %%
theories_lya = []
theories_cont = []
for z in zs:
    theories_lya.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False}))
    theories_cont.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False, 
                                                            'include_hcd': True, 'include_metal': True,
                                                            'include_sky': True, 'include_continuum': True} ))

# %% [markdown]
# ## Step 3: create Likelihoods and compare data vs theory (no fits)

# %%
likes_lya = []
likes_cont = []
for iz, z in enumerate(zs):
    likes_lya.append(Likelihood(data=data, theory=theories_lya[iz], iz=iz, verbose=False))
    likes_cont.append(Likelihood(data=data, theory=theories_cont[iz], iz=iz, verbose=False))

# %%
models_lya = []
models_cont = []
for iz, z in enumerate(zs):
    models_lya.append(likes_lya[iz].get_convolved_px(params={}))
    models_cont.append(likes_cont[iz].get_convolved_px(params={}))


# %%
def compare_theta_bin(iz, it_M):
    plt.title(r"DESI DR2,   z = {:.1f},   ${:.2f}' < \theta < {:.2f}'$".format(
                                                zs[iz], theta_A_min[it_M], theta_A_max[it_M]))
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it_M]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_M]))
    plt.errorbar(k_M, Px, sig_Px, label='data')    
    plt.plot(k_M, models_lya[iz][it_M], label='Lya only')
    plt.plot(k_M, models_cont[iz][it_M], label='Lya + cont')
    plt.legend()
    plt.xlabel(r'$k_\parallel$ [1/A]')
    plt.ylabel(r'$P_\times(\theta, k_\parallel)$ [A]')
    plt.axhline(y=0, ls=':', color='gray')


# %%
# one z, multiple theta
for it_M in range(Nt_A):
    plt.figure()
    compare_theta_bin(iz=2, it_M=it_M)

# %%
# one theta, multiple z
for iz, z in enumerate(zs):
    plt.figure()
    compare_theta_bin(iz=iz, it_M=0)

# %% [markdown]
# ## Step 4: setup iminuit minimizers and fit for parameters

# %%
from cupix.likelihood.new_minimizer import IminuitMinimizer
from cupix.likelihood.likelihood_parameter import LikelihoodParameter

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
like_params = []
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-.5,
    max_value=-.05,
    ini_value=-.15,
    value =-.15
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.5,
    max_value=2.5,
    ini_value=1.5,
    value =1.5
    ))
for par in like_params:
    print(par.name)

# %%
# do this only for one z bin
fit_iz=1
mini_lya = IminuitMinimizer(likes_lya[fit_iz], free_params=like_params, verbose=True)
mini_cont = IminuitMinimizer(likes_cont[fit_iz], free_params=like_params, verbose=True)

# %%
mini_lya.silence()
mini_lya.minimize()

# %%
mini_cont.silence()
mini_cont.minimize()

# %%
# number of data points (per z bin)
Ndp = Nt_A * Nk_M
chi2_lya = mini_lya.get_best_fit_chi2()
chi2_cont = mini_cont.get_best_fit_chi2()
print(Ndp, chi2_lya, chi2_cont)

# %% [markdown]
# ## Step 5: fit for contaminants

# %%
print('HCD', mini_cont.like.theory.cont_model.default_hcd_params)
print('Metal', mini_cont.like.theory.cont_model.default_metal_params)
print('Sky', mini_cont.like.theory.cont_model.default_sky_params)
print('Cont', mini_cont.like.theory.cont_model.default_continuum_params)

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
like_params = []
free_b_H=False
free_b_X=True
free_b_noise_Mpc=False
free_kC_Mpc=False
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-.5,
    max_value=-.05,
    ini_value=-.15,
    value =-.15
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.5,
    max_value=2.5,
    ini_value=1.5,
    value =1.5
    ))
if free_b_H:
    like_params.append(LikelihoodParameter(
        name='b_H',
        min_value=-0.1,
        max_value=-0.0,
        ini_value=-0.02,
        value = -0.02
        ))
if free_b_X:
    like_params.append(LikelihoodParameter(
        name='b_X',
        min_value=-0.1,
        max_value=-0.0,
        ini_value=-0.01,
        value = -0.01
        ))
if free_b_noise_Mpc:
    like_params.append(LikelihoodParameter(
        name='b_noise_Mpc',
        min_value=1e-4,
        max_value=1e-1,
        ini_value=0.01,
        value = 0.01
        ))
if free_kC_Mpc:
    like_params.append(LikelihoodParameter(
        name='kCb_Mpc',
        min_value=1e-3,
        max_value=1e-1,
        ini_value=0.01,
        value = 0.01
        ))    
for par in like_params:
    print(par.name)

# %%
mini = IminuitMinimizer(likes_cont[fit_iz], free_params=like_params, verbose=True)

# %%
mini.silence()
mini.minimize()

# %%
mini.get_best_fit_chi2()

# %%
mini.plot_ellipses(pname_x='bias', pname_y='beta', nsig=2)

# %%
mini.plot_ellipses(pname_x='bias', pname_y='b_X', nsig=2)

# %%
mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, xlim=[-.01, .6], datalabel="DR2 (z = {})".format(zs[fit_iz]), show=True)

# %%

# %%
