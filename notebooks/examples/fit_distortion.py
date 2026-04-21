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
# # Fitting continuum distortion from the stack of many mocks

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
from cupix.likelihood.free_parameter import FreeParameter
from cupix.likelihood.posterior import Posterior
from cupix.likelihood.minimize_posterior import Minimizer

# %% [markdown]
# ### Read the Px from the stack of 50 mocks

# %%
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
# true continuum
true_fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
true_data = DESI_DR2(true_fname, kM_max_cut_AA=0.3, km_max_cut_AA=0.35, theta_min_cut_arcmin=20.0)
# uncontaminated
unco_fname = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_10_w_res_avg50.hdf5"
unco_data = DESI_DR2(unco_fname, kM_max_cut_AA=0.3, km_max_cut_AA=0.35, theta_min_cut_arcmin=20.0)

# %% [markdown]
# ### Start by fitting bias/beta from the stack of true-continuum mocks (one-z at a time)

# %%
# define fiducial cosmo
cosmo = cosmology.Cosmology()
# starting point for Lya bias parameters in mocks
default_lya_model = 'best_fit_arinyo_from_colore'

# %%
# set free parameters
bias = FreeParameter(
    name='bias',
    min_value=-0.5,
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
free_params = [bias, beta]
for par in free_params:
    print(par.name, par.ini_value)

# %%
true_minis = []
for iz, z in enumerate(true_data.z): 
    theory = Theory(z=z, fid_cosmo=cosmo, 
                    config={'verbose': True, 'default_lya_model': default_lya_model, 'include_continuum': False})
    # reset bias/beta 
    assert free_params[0].name == 'bias'
    free_params[0].ini_value = theory.lya_model.default_lya_params['bias']
    assert free_params[1].name == 'beta'
    free_params[1].ini_value = theory.lya_model.default_lya_params['beta'] 
    like = Likelihood(data=true_data, theory=theory, iz=iz, config={'verbose':True})
    post = Posterior(like, free_params, config={'verbose': True})
    mini = Minimizer(post, config={'verbose':True})
    true_minis.append(mini)

# %%
for mini in true_minis:
    z = mini.post.like.theory.z
    print('--------- z = {:.2f} -------'.format(z))
    # number of data points (per z bin)
    Nz, Nt_A, Nk_M = mini.post.like.data.Px_ZAM.shape
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
if True:
    z = [ mini.post.like.theory.z for mini in true_minis]
    val = [ mini.get_best_fit_value('bias', return_hesse=True)[0] for mini in true_minis]
    err = [ mini.get_best_fit_value('bias', return_hesse=True)[1] for mini in true_minis]
    plt.errorbar(z, val, err, label='Px fits')
    val = [ mini.post.like.theory.lya_model.default_lya_params['bias'] for mini in true_minis ]
    plt.plot(z, val, 'ro', label='Xi3D fits')
    plt.xlabel('z')
    plt.ylabel('bias')
    plt.legend()

# %%
if True:
    z = [ mini.post.like.theory.z for mini in true_minis]
    val = [ mini.get_best_fit_value('beta', return_hesse=True)[0] for mini in true_minis]
    err = [ mini.get_best_fit_value('beta', return_hesse=True)[1] for mini in true_minis]
    plt.errorbar(z, val, err, label='Px fits')
    val = [ mini.post.like.theory.lya_model.default_lya_params['beta'] for mini in true_minis ]
    plt.plot(z, val, 'ro', label='Xi3D fits')
    plt.xlabel('z')
    plt.ylabel('beta')
    plt.legend()

# %% [markdown]
# ### Now fit continuum-fitted mocks (fixed bias/beta)

# %%
free_params = []
free_params.append(FreeParameter(
    name='kC_Mpc',
    min_value=1e-4,
    max_value=1e-1,
    ini_value=0.01,
    delta=0.001
))
free_params.append(FreeParameter(
    name='pC',
    min_value=0.01,
    max_value=2.0,
    ini_value=1.0,
    delta=0.01
))
for par in free_params:
    print(par.name, par.ini_value)

# %%
unco_minis = []
for iz, z in enumerate(unco_data.z): 
    config={'verbose': True, 'default_lya_model': default_lya_model, 'include_continuum': True}
    config['bias'] = true_minis[iz].get_best_fit_value('bias') 
    config['beta'] = true_minis[iz].get_best_fit_value('beta') 
    theory = Theory(z=z, fid_cosmo=cosmo, config=config)
    like = Likelihood(data=unco_data, theory=theory, iz=iz, config={'verbose':True})
    post = Posterior(like, free_params, config={'verbose': True})
    mini = Minimizer(post, config={'verbose':True})
    print('initial chi2', like.get_chi2())
    unco_minis.append(mini)

# %%
for mini in unco_minis:
    z = mini.post.like.theory.z
    print('--------- z = {:.2f} -------'.format(z))
    # number of data points (per z bin)
    Nz, Nt_A, Nk_M = mini.post.like.data.Px_ZAM.shape
    Ndp = Nt_A * Nk_M
    # silence and minimize
    mini.silence()
    mini.minimize(compute_hesse=True)
    chi2 = mini.get_best_fit_chi2()
    best_fit = mini.get_best_fit_params()
    print('best fit chi2 and params')
    print(z, Ndp, chi2, best_fit)
    mini.plot_ellipses(pname_x='kC_Mpc', pname_y='pC', nsig=2)
    label=""
    for key, par in mini.get_best_fit_params().items():
        label += "{} = {:.3f}   ".format(key, par)
    mini.plot_best_fit(multiply_by_k=False, theorylabel=label, datalabel='Stack (uncontaminated)')

# %%
if True:
    z = [ mini.post.like.theory.z for mini in unco_minis]
    val = [ mini.get_best_fit_value('kC_Mpc', return_hesse=True)[0] for mini in unco_minis]
    err = [ mini.get_best_fit_value('kC_Mpc', return_hesse=True)[1] for mini in unco_minis]
    plt.errorbar(z, val, err, label='Px fits')
    plt.xlabel('z')
    plt.ylabel('kC_Mpc')
    plt.tight_layout()
    plt.savefig('kC_Mpc_z.png')

# %%
if True:
    z = [ mini.post.like.theory.z for mini in unco_minis]
    val = [ mini.get_best_fit_value('pC', return_hesse=True)[0] for mini in unco_minis]
    err = [ mini.get_best_fit_value('pC', return_hesse=True)[1] for mini in unco_minis]
    plt.errorbar(z, val, err, label='Px fits')
    plt.xlabel('z')
    plt.ylabel('pC')
    plt.tight_layout()
    plt.savefig('pC_z.png')

# %%
if True:
    z = [ mini.post.like.theory.z for mini in unco_minis]
    ini_chi2 = [ mini.post.like.get_chi2() for mini in unco_minis]
    best_fit_chi2 = [ mini.get_best_fit_chi2() for mini in unco_minis]
    plt.plot(z, ini_chi2, label=r'initial $\chi^2$')
    plt.plot(z, best_fit_chi2, label=r'best-fit $\chi^2$')
    plt.plot(z, 0.1*Ndp*np.ones_like(z), label='Number of data points / 10')
    plt.xlabel('z')
    plt.legend()

# %%
for mini in unco_minis:
    z = mini.post.like.theory.z
    print(z, mini.get_best_fit_params())

# %%

# %%
