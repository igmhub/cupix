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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Use iminuit to fit Px from DESI DR2 for IGM parameters

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
from cupix.inference.free_parameter import FreeParameter
from cupix.inference.posterior import Posterior
from cupix.inference.minimize_posterior import Minimizer
from cupix.inference.sampling_funcs import prepare_free_parameters

# %% [markdown]
# ## Step 1: Read the data from DESI DR2 and plot it

# %%
basedir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/"
#fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5"
# fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_20_w_res.hdf5"
# fname = basedir + "wp1d/drop_DLAs/GP_plus_snrcut/bf3_binned_out_px-zbins_4-thetabins_20_w_res_wp1d.hdf5"
fname = basedir + "wp1d/drop_BALs_and_DLAs/fs_cut/bf3_binned_out_px-zbins_4-thetabins_20_w_res_wp1d.hdf5"
kM_max_cut_AA = .7
km_max_cut_AA = 1.1 * kM_max_cut_AA
print(km_max_cut_AA)
data = DESI_DR2(config={'data_file':fname, 'kM_min_cut_AA':0.03, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':1})

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
b_noise = [0.0040, 0.0017, 0.0017, 0.0016]

# %%
theories_lya = []
theories_cont = []
iz = 0
for z in zs:
    theories_lya.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False, 'default_lya_model':'best_fit_igm_from_p1d'}))
    theories_cont.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False, 
                                                            'include_hcd': True, 'include_metal': True,
                                                            'include_sky': True, 'include_continuum': True, 'default_lya_model':'best_fit_igm_from_p1d',
                                                            'b_noise_Mpc': b_noise[iz]} ))
    iz += 1

# %% [markdown]
# ## Step 3: create Likelihoods and compare data vs theory (no fits)

# %%
likes_lya = []
likes_cont = []
for iz, z in enumerate(zs):
    likes_lya.append(Likelihood(data=data, theory=theories_lya[iz], iz=iz, config={'verbose':False}))
    likes_cont.append(Likelihood(data=data, theory=theories_cont[iz], iz=iz, config={'verbose':False}))

# %%
models_lya = []
models_cont = []
for iz, z in enumerate(zs):
    models_lya.append(likes_lya[iz].get_convolved_px(params={}))
    models_cont.append(likes_cont[iz].get_convolved_px(params={}))

# %% [markdown]
# ## Step 4: setup iminuit minimizers and fit for parameters

# %%
free_params = []
free_params.append(FreeParameter(
    name='mF',
    min_value=0.65,
    max_value=0.9,
    ini_value=0.8,
    gauss_prior_mean=0.8,
    gauss_prior_width=0.15,
    delta=0.02
    ))
free_params.append(FreeParameter(
    name='gamma',
    min_value=1.24,
    max_value=1.82,
    ini_value=1.5,
    gauss_prior_mean=1.5,
    gauss_prior_width=0.3,
    delta=0.1
    ))
free_params.append(FreeParameter(
    name='sigT_Mpc',
    min_value=0.1,
    max_value=0.16,
    ini_value=0.13,
    gauss_prior_mean=0.13,
    gauss_prior_width=0.03,
    delta=0.005
 ))

# %%
# # set the likelihood parameters as the Arinyo params with some fiducial values
# freepars = []
# for iz, z in enumerate(zs):
#     freeparams = prepare_free_parameters(['mF','sigT_Mpc', 'gamma'], theories_cont[iz], theory_config={'verbose': False, 
#                                                             'include_hcd': True, 'include_metal': True,
#                                                             'include_sky': True, 'include_continuum': True, 'default_lya_model':'best_fit_igm_from_p1d',
#                                                             'b_noise': b_noise[iz]} )
#     print(theories_cont[iz].z)
#     for par in freeparams:
#         print(par.name, par.true_value, par.ini_value, par.min_value, par.max_value, par.delta)
#     freepars.append(freeparams)

# %%
# do this only for one z bin
fit_iz=3
# post_lya = Posterior(likes_lya[fit_iz], free_params, config={'verbose': True})
# post_cont = Posterior(likes_cont[fit_iz], freepars[fit_iz], config={'verbose': True})
post_cont = Posterior(likes_cont[fit_iz], free_params, config={'verbose': True})
# mini_lya = Minimizer(post_lya, config={'verbose':True})
mini_cont = Minimizer(post_cont, config={'verbose':True})

# %%
# mini_lya.silence()
# mini_lya.minimize()

# %%
mini_cont.silence()
mini_cont.minimize()

# %%
mini_cont.print_results()

# %%
likes_cont[3].theory.cont_model.default_metal_params

# %%
# 'mF':0.75, 'gamma':1.24, 'sigT_Mpc':0.16, 

# %%
plt.imshow(data.cov_ZAM[3,1,:,:])

# %%
likes_cont[3].plot_px(params={'pC':1.4, 'mF':0.75, 'gamma':1.2, 'sigT_Mpc':0.15, 'b_H':-0.005, 'L_H_Mpc':2, 'beta_H':2, 'b_X':-0.02, 'beta_X':0.1}, multiply_by_k=False,every_other_theta=True, include_chi2=False)

# %%
mini_cont.plot_best_fit(include_chi2=True, multiply_by_k=False, title=f"z={fit_iz}", every_other_theta=True)

# %%
mini_cont.plot_ellipse('mF','sigT_Mpc')
# mini_cont.plot_corner(true_val_label='DESI DR1 P1D')

# %%
mini_cont.save_results(outdir="/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/drop_dla_bal_fscut/", outfile=f"mini_mFgammasigT_defaultcont_propersky_z{fit_iz}")

# %% [markdown]
# ## Step 5: fit for contaminants

# %%
print('HCD', mini_cont.post.like.theory.cont_model.default_hcd_params)
print('Metal', mini_cont.post.like.theory.cont_model.default_metal_params)
print('Sky', mini_cont.post.like.theory.cont_model.default_sky_params)
print('Cont', mini_cont.post.like.theory.cont_model.default_continuum_params)

# %%
# set the free parameters 
free_params = [bias, beta]
free_b_H=False
free_b_X=True
free_b_noise_Mpc=False
free_kC_Mpc=False
if free_b_H:
    free_params.append(FreeParameter(
        name='b_H',
        min_value=-0.1,
        max_value=-0.0,
        ini_value=-0.02,
        delta=0.001  
        ))
if free_b_X:
    free_params.append(FreeParameter(
        name='b_X',
        min_value=-0.1,
        max_value=-0.0,
        ini_value=-0.01,
        delta=0.001 
        ))
if free_b_noise_Mpc:
    free_params.append(FreeParameter(
        name='b_noise_Mpc',
        min_value=1e-4,
        max_value=1e-1,
        ini_value=0.01,
        delta=0.001
        ))
if free_kC_Mpc:
    free_params.append(FreeParameter(
        name='kCb_Mpc',
        min_value=1e-3,
        max_value=1e-1,
        ini_value=0.01,
        delta=0.001
        ))    
for par in free_params:
    print(par.name)

# %%
post = Posterior(likes_cont[fit_iz], free_params, config={'verbose': True})
mini = Minimizer(post, config={'verbose':True})

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
mini.plot_best_fit(multiply_by_k=False, every_other_theta=True, xlim=[-.01, .6], datalabel="DR2 (z = {})".format(zs[fit_iz]), show=True)

# %%

# %%
