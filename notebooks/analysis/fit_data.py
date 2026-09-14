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
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.inference.free_parameter import FreeParameter
from cupix.inference.posterior import Posterior
from cupix.inference.minimize_posterior import Minimizer

# %% [markdown]
# ## Step 1: Read the data from DESI DR2 and plot it

# %%
basedir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/"
#fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5"
# fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_20_w_res.hdf5"
fname = basedir + "/baseline/wp1d/BALs_weighted_to_zero/bf3_binned_out_px-zbins_4-thetabins_20_w_res_wp1d.hdf5"
fname_dropdla = basedir + "/baseline/wp1d/drop_DLAs/GP_plus_snrcut/bf3_binned_out_px-zbins_4-thetabins_20_w_res_wp1d.hdf5"
fname_nomaskdla = basedir + "no_DLA_masking/bf3_binned_out_px-zbins_4-thetabins_20_w_res_wp1d.hdf5"
kM_max_cut_AA = .7
km_max_cut_AA = 1.1 * kM_max_cut_AA
data = DESI_DR2(config={'data_file':fname, 'kM_min_cut_AA':0.03, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':5})
data_nodla = DESI_DR2(config={'data_file':fname_dropdla, 'kM_min_cut_AA':0.03, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':5})
data_nomaskdla = DESI_DR2(config={'data_file':fname_nomaskdla, 'kM_min_cut_AA':0.03, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':5})

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
def plot_theta_bin(data, iz, it_M, ls = 'solid'):
    colors = plt.get_cmap('tab20')
    label = r"${:.2f}' < \theta < {:.2f}'$".format(theta_A_min[it_M], theta_A_max[it_M])
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it_M]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_M]))
#    print(len(k_M), len(Px), len(sig_Px))
    plt.errorbar(k_M, Px, sig_Px, label=label, ls=ls, color=colors(it_M))


# %%

def plot_z_bin(data, iz, its_M, ls= 'solid', show_legend=True):
    
    for it_M in its_M[10:]:
        plot_theta_bin(data=data, iz=iz, it_M=it_M, ls=ls)
    plt.title('DESI DR2 at z={:.1f}'.format(zs[iz]))
    if show_legend:
        plt.legend(fontsize=10)
    plt.xlabel(r'$k_\parallel$ [1/A]')
    plt.ylabel(r'$P_\times(\theta, k_\parallel)$ [A]')
    plt.xlim([0,0.2])
    plt.ylim([0,0.02])


# %%
for iz in range(4):
    plt.figure(figsize=[8,3])
    plot_z_bin(data, iz=iz, its_M=range(Nt_A))
    plot_z_bin(data_nodla, iz=iz, its_M = range(Nt_A), ls='dashed', show_legend=False)
    plot_z_bin(data_nomaskdla, iz=iz, its_M = range(Nt_A), ls='dotted', show_legend=False)
    

# %% [markdown]
# ## Step 2: setup theory objects, with and without contaminants (one per z)

# %%
# define fiducial cosmo
cosmo = cosmology.Cosmology()

# %%
theories_lya = []
theories_cont = []
iz = 0
for z in zs:
    theories_lya.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False, 'default_lya_model':'best_fit_arinyo_from_p1d'}))
    theories_cont.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False, 
                                                            'include_hcd': True, 'include_metal': True,
                                                            'include_sky': True, 'include_continuum': True, 'default_lya_model':'best_fit_arinyo_from_p1d'} ))
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
# set the likelihood parameters as the Arinyo params with some fiducial values
bias = FreeParameter(
    name='bias',
    min_value=-0.3,
    max_value=-0.01,
    ini_value=-0.15,
    delta=0.01,   
)
beta = FreeParameter(
    name='beta',
    min_value=0.1,
    max_value=5.0,
    ini_value=1.5,
    delta=0.1,
)
# q1 = FreeParameter(
#     name='q1',
#     min_value=0.1,
#     max_value=2.0,
#     ini_value=0.5,
#     delta=0.1,
# )
bH = FreeParameter(
    name='b_H',
    min_value=-0.15,
    max_value=0.0,
    ini_value=-0.014,
    delta=0.002,
)
bX = FreeParameter(
    name='b_X',
    min_value=-0.015,
    max_value=-0.001,
    ini_value=-0.007,
    delta=0.001,
)

free_params = [bias, beta, bH, bX]
for par in free_params:
    print(par.name, par.ini_value)

# %%
# do this only for one z bin
fit_iz=0
# post_lya = Posterior(likes_lya[fit_iz], free_params, config={'verbose': True})
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
theories_cont[0].get_param('b_X')

# %%
likes_cont[0].plot_px(include_chi2=True, multiply_by_k=False, params={'bias':-0.0958, 'beta':1.6929, 'b_H':-0.04, 'b_X':-0.015})

# %%
mini_cont.plot_best_fit(include_chi2=True, multiply_by_k=False)

# %%
mini_cont.plot_ellipse('bias','beta')

# %%
mini_cont.plot_corner(show_truth=False)

# %%
mini_cont.save_results(outdir="/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/july_baseline/", outfile=f"mini_biasbetabHbX_defaultcont_propersky_z{fit_iz}")

# %%
# read all results from directory


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
