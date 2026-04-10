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
# # Fit contaminants from DESI DR2

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
from cupix.likelihood.new_likelihood import Likelihood
from cupix.likelihood.theory import Theory
from cupix.likelihood.new_minimizer import IminuitMinimizer

# %% [markdown]
# ### Read the data from DESI DR2 (large angular separations only)

# %%
fname = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5"
data = DESI_DR2(fname, kM_max_cut_AA=1.0, km_max_cut_AA=1.2, theta_min_cut_arcmin=10.0)

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
# number of data points (per z bin)
Ndp = Nt_A * Nk_M


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
for iz, z in enumerate(zs):
    plt.figure(figsize=[8,3])
    plot_z_bin(iz=iz, its_M=range(Nt_A))

# %% [markdown]
# ### Setup (contaminated) theory and likelihood objects

# %%
# define fiducial cosmo
cosmo = cosmology.Cosmology()

# %%
config={'verbose': False, 'include_hcd': True, 'include_metal': True,
        'include_sky': True, 'include_continuum': True}
theories = []
for iz,z in enumerate(zs):
    # improve default contaminants
    if iz==0:
        config['b_noise_Mpc'] = 0.003
        config['kC_Mpc'] = 0.015
    else:
        config['b_noise_Mpc'] = 0.001
        config['kC_Mpc'] = 0.010
    theories.append(Theory(z=z, fid_cosmo=cosmo, config=config))

# %%
likes = []
for iz, z in enumerate(zs):
    likes.append(Likelihood(data=data, theory=theories[iz], iz=iz, verbose=False))

# %%
# plot data vs default theory for each z
for iz, z in enumerate(zs):
    datalabel = 'DESI DR2 at z={}'.format(z)
    theorylabel = 'Default contaminated theory'
    likes[iz].plot_px(params={}, multiply_by_k=False, datalabel=datalabel, theorylabel=theorylabel)

# %% [markdown]
# ### Setup minimizers and free parameters

# %%
print('HCD params =', theories[0].cont_model.default_hcd_params)
print('Metal params =', theories[0].cont_model.default_metal_params)
print('Sky params =', theories[0].cont_model.default_sky_params)
print('Cont params =', theories[0].cont_model.default_continuum_params)

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
free_params = []
free_b_H=False
free_b_X=False
free_b_noise_Mpc=False
free_kC_Mpc=False
free_params.append(LikelihoodParameter(
    name='bias',
    min_value=-.5,
    max_value=-.05,
    ini_value=-.15,
    value =-.15
    ))
free_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.5,
    max_value=2.5,
    ini_value=1.5,
    value =1.5
    ))
if free_b_H:
    free_params.append(LikelihoodParameter(
        name='b_H',
        min_value=-0.1,
        max_value=-0.0,
        ini_value=-0.02,
        value = -0.02
        ))
if free_b_X:
    free_params.append(LikelihoodParameter(
        name='b_X',
        min_value=-0.1,
        max_value=-0.0,
        ini_value=-0.01,
        value = -0.01
        ))
if free_b_noise_Mpc:
    free_params.append(LikelihoodParameter(
        name='b_noise_Mpc',
        min_value=1e-4,
        max_value=1e-1,
        ini_value=0.01,
        value = 0.01
        ))
if free_kC_Mpc:
    free_params.append(LikelihoodParameter(
        name='kC_Mpc',
        min_value=1e-4,
        max_value=1e-1,
        ini_value=0.01,
        value = 0.01
        ))    
for par in free_params:
    print(par.name)

# %%
minis = []
for iz in range(Nz):
#for iz in [2]:
    # chose more realistic initial values for bias/beta
    assert free_params[0].name == 'bias'
    free_params[0].ini_value = likes[iz].theory.lya_model.default_lya_params['bias']
    assert free_params[1].name == 'beta'
    free_params[1].ini_value = likes[iz].theory.lya_model.default_lya_params['beta']    
    mini = IminuitMinimizer(likes[iz], free_params=free_params, verbose=True)
    mini.silence()
    mini.minimize(compute_hesse=True)
    chi2 = mini.get_best_fit_chi2()
    best_fit = mini.get_best_fit_params()
    print('best fit chi2 and params')
    print(zs[iz], Ndp, chi2, best_fit)
    print('-------')
    print('-------')
    minis.append(mini)

# %%
for mini in minis:
    print('DESI DR2, z =', mini.like.theory.z)
    plt.figure()
    mini.plot_ellipses(pname_x='bias', pname_y='beta', nsig=2)
    if free_b_H:
        plt.figure()
        mini.plot_ellipses(pname_x='bias', pname_y='b_H', nsig=2)
    if free_b_X:
        plt.figure()
        mini.plot_ellipses(pname_x='bias', pname_y='b_X', nsig=2)
    if free_b_noise_Mpc:
        plt.figure()
        mini.plot_ellipses(pname_x='bias', pname_y='b_noise_Mpc', nsig=2)
    if free_kC_Mpc:
        plt.figure()
        mini.plot_ellipses(pname_x='bias', pname_y='kC_Mpc', nsig=2)

# %%
for mini in minis:
    z=mini.like.theory.z
    mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, xlim=[-.01, 1.0], 
                       datalabel="DR2 (z = {})".format(z), 
                       theorylabel="Best-fit model", show=True)

# %%
if True:
    z = [ mini.like.theory.z for mini in minis]
    chi2 = [ mini.get_best_fit_chi2() for mini in minis]
    plt.plot(z, chi2, label=r'$\chi^2$')
    plt.plot(z, Ndp*np.ones_like(z), label='Number of data points')
    plt.xlabel('z')
    plt.legend()

# %%
if True:
    z = [ mini.like.theory.z for mini in minis]
    val = [ mini.get_best_fit_value('bias', return_hesse=True)[0] for mini in minis]
    err = [ mini.get_best_fit_value('bias', return_hesse=True)[1] for mini in minis]
    plt.errorbar(z, val, err, label='Px fits')
    xi3d=theories[0].lya_model.default_lya_params['bias']
    plt.plot(2.33, xi3d, 'ro', label='Xi3D fit')
    plt.xlabel('z')
    plt.ylabel('bias')
    plt.legend()

# %%
if True:
    z = [ mini.like.theory.z for mini in minis]
    val = [ mini.get_best_fit_value('beta', return_hesse=True)[0] for mini in minis]
    err = [ mini.get_best_fit_value('beta', return_hesse=True)[1] for mini in minis]
    plt.errorbar(z, val, err, label='Px fits')
    xi3d=theories[0].lya_model.default_lya_params['beta']
    plt.plot(2.33, xi3d, 'ro', label='Xi3D fit')
    plt.xlabel('z')
    plt.ylabel('beta')
    plt.legend()

# %%
if free_b_H:
    z = [ mini.like.theory.z for mini in minis]
    val = [ mini.get_best_fit_value('b_H', return_hesse=True)[0] for mini in minis]
    err = [ mini.get_best_fit_value('b_H', return_hesse=True)[1] for mini in minis]
    plt.errorbar(z, val, err, label='Px fits')
    xi3d=theories[0].cont_model.default_hcd_params['b_H']
    plt.plot(2.33, xi3d, 'ro', label='Xi3D fit')
    plt.xlabel('z')
    plt.ylabel('b_H')
    plt.legend()

# %%
if free_b_H:
    z = [ mini.like.theory.z for mini in minis]
    val = [ mini.get_best_fit_value('b_H', return_hesse=True)[0] for mini in minis]
    err = [ mini.get_best_fit_value('b_H', return_hesse=True)[1] for mini in minis]
    plt.errorbar(z, val, err, label='Px fits')
    xi3d=theories[0].cont_model.default_hcd_params['b_H']
    plt.plot(2.33, xi3d, 'ro', label='Xi3D fit')
    plt.xlabel('z')
    plt.ylabel('b_H')
    plt.legend()

# %%
if free_b_X:
    z = [ mini.like.theory.z for mini in minis]
    val = [ mini.get_best_fit_value('b_X', return_hesse=True)[0] for mini in minis]
    err = [ mini.get_best_fit_value('b_X', return_hesse=True)[1] for mini in minis]
    plt.errorbar(z, val, err, label='Px fits')
    xi3d=theories[0].cont_model.default_metal_params['b_X']
    plt.plot(2.33, xi3d, 'ro', label='Xi3D fit')
    plt.xlabel('z')
    plt.ylabel('b_X')
    plt.legend()

# %%
if free_b_noise_Mpc:
    z = [ mini.like.theory.z for mini in minis]
    val = [ mini.get_best_fit_value('b_noise_Mpc', return_hesse=True)[0] for mini in minis]
    err = [ mini.get_best_fit_value('b_noise_Mpc', return_hesse=True)[1] for mini in minis]
    plt.errorbar(z, val, err, label='Px fits')
    xi3d=theories[0].cont_model.default_sky_params['b_noise_Mpc']
    plt.plot(2.33, xi3d, 'ro', label='Xi3D fit')
    plt.xlabel('z')
    plt.ylabel('b_noise [Mpc]')
    plt.legend()

# %%
if free_kC_Mpc:
    z = [ mini.like.theory.z for mini in minis]
    val = [ mini.get_best_fit_value('kC_Mpc', return_hesse=True)[0] for mini in minis]
    err = [ mini.get_best_fit_value('kC_Mpc', return_hesse=True)[1] for mini in minis]
    plt.errorbar(z, val, err, label='Px fits')
    xi3d=theories[0].cont_model.default_continuum_params['kC_Mpc']
    plt.plot(2.33, xi3d, 'ro', label='Xi3D fit (Blomqvist)')
    plt.xlabel('z')
    plt.ylabel('kC [1/Mpc]')
    plt.legend()
    plt.ylim([0,0.03])

# %%

# %%

# %%
