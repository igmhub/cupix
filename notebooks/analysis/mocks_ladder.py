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
# # Fitting the stack of true-cont / uncont / cont mocks
#
# Start by fitting the true-continuum mocks (vary bias / beta / kp_Mpc)
#
# Then fit uncontaminated mocks with priors (vary kC_Mpc, pC)
#
# Then fit contaminated mocks with priors (vary ...)

# %%
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
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

# %%
# In this notebook we will work with a single z bin
iz=1

# setup cosmology (should check this is the right cosmology in the mocks)
cosmo = cosmology.Cosmology()
# starting point for Lya bias parameters in mocks
default_lya_model = 'pressure_only_fits_from_colore'
#default_lya_model = 'best_fit_arinyo_from_colore'

# path to mocks
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
# will rescale covariance of the stacks with this number
rescale_cov = False
Nm = 50

# %% [markdown]
# ## Step 1: Fit true-continuum mocks (bias / beta / kp_Mpc)

# %%
#fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_10_w_res_avg{}.hdf5".format(Nm)
fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg{}.hdf5".format(Nm)

# speed-up code by only looking at low kpar
kM_max_cut_AA = 0.7
km_max_cut_AA = 1.1*kM_max_cut_AA
theta_min_cut_arcmin = 5.0
data = DESI_DR2(fname, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA,
                                 theta_min_cut_arcmin=theta_min_cut_arcmin)
z = data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))
if rescale_cov:
    data.cov_ZAM *= 1.0 / Nm

# %%
theory_config = {'verbose': False, 'default_lya_model': default_lya_model, 'include_continuum': False}
theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
print(theory.lya_model.default_lya_params)

# %%
like = Likelihood(data=data, theory=theory, iz=iz, config={'verbose':False})

# %%
# set initial value for bias / beta based on best-fit values from Laura
ini_bias = theory.lya_model.default_lya_params['bias']
ini_beta = theory.lya_model.default_lya_params['beta']
ini_kp = theory.lya_model.default_lya_params['kp_Mpc']
par_bias = FreeParameter(
    name='bias',
    min_value=-0.5,
    max_value=-0.01,
    ini_value=ini_bias,
    delta=0.01,   
)
par_beta = FreeParameter(
    name='beta',
    min_value=0.1,
    max_value=5.0,
    ini_value=ini_beta,
    delta=0.1,
)
par_kp = FreeParameter(
    name='kp_Mpc',
    min_value=0.0,
    max_value=5.0,
    ini_value=ini_kp,
    delta=0.01
)
free_params = [par_bias, par_beta, par_kp]
for par in free_params:
    print(par.name, par.ini_value)

# %%
post = Posterior(like, free_params, config={'verbose':False})
true_cont_mini = Minimizer(post, config={'verbose':False})
true_cont_mini.silence()
start = time.time()
true_cont_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
true_cont_mini.print_results()

# %%
true_cont_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True)

# %% [markdown]
# ## Step 2: Fit uncontaminated mocks (add kC_Mpc, pC)

# %%
# path to mocks
#fname = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_10_w_res_avg{}.hdf5".format(Nm)
fname = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg{}.hdf5".format(Nm)
# speed-up code by only looking at low kpar (should be enough for theta > 3 arcmin or so)
kM_max_cut_AA = 0.7
km_max_cut_AA = 1.1*kM_max_cut_AA
theta_min_cut_arcmin = 5.0
data = DESI_DR2(fname, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA,
                                 theta_min_cut_arcmin=theta_min_cut_arcmin)
z = data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))
if rescale_cov:
    data.cov_ZAM *= 1.0 / Nm


# %%
def copy_with_prior(par, mini):
    new_par = copy.deepcopy(par)
    val, err = mini.get_best_fit_value(par.name, return_hesse=True)
    print('set prior to {} = {:.4f} +/- {:.4f}'.format(par.name, val, err))  
    new_par.ini_value = val
    new_par.delta = 0.1*err
    new_par.gauss_prior_mean = val
    new_par.gauss_prior_width = err
    return new_par


# %%
# use priors on bias / beta / kp_Mpc based on previous fits above
par_bias = copy_with_prior(par_bias, true_cont_mini)
par_beta = copy_with_prior(par_beta, true_cont_mini)
par_kp = copy_with_prior(par_kp, true_cont_mini)

# %%
update_theory = True
theory_config = {'verbose': False, 'default_lya_model': default_lya_model, 'include_continuum': True}
if update_theory:
    for par in [par_bias, par_beta, par_kp]:
        theory_config[par.name] = par.ini_value
theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
print(theory.lya_model.default_lya_params)
print(theory.cont_model.default_continuum_params)
like = Likelihood(data=data, theory=theory, iz=iz, config={'verbose':False})

# %%
# add extra params for continuum fitting distortion
ini_kC = theory.cont_model.default_continuum_params['kC_Mpc']
ini_pC = theory.cont_model.default_continuum_params['pC']
par_kC = FreeParameter(
    name='kC_Mpc',
    min_value=1e-4,
    max_value=1e-1,
    ini_value=ini_kC,
    delta=0.0001
)
par_pC = FreeParameter(
    name='pC',
    min_value=0.001,
    max_value=2.0,
    ini_value=ini_pC,
    delta=0.001
)
free_params = [par_bias, par_beta, par_kp, par_kC, par_pC]
for par in free_params:
    if par.gauss_prior_mean is None:
        print(par.name, par.ini_value, 'no prior')
    else:
        print(par.name, par.ini_value, 'prior', par.gauss_prior_mean , '+/-', par.gauss_prior_width)

# %%
post = Posterior(like, free_params, config={'verbose':False})
unco_mini = Minimizer(post, config={'verbose':False})
unco_mini.silence()
start = time.time()
unco_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
unco_mini.print_results()

# %%
unco_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True)

# %% [markdown]
# ## Step 3: Fit contaminated mocks (add b_H, b_X)

# %%
# path to mocks
fname = mockdir + "contaminated/contaminated_binned_out_bf3_px-zbins_4-thetabins_10_w_res_avg{}.hdf5".format(Nm)
# speed-up code by only looking at low kpar
kM_max_cut_AA = 0.7
km_max_cut_AA = 1.1*kM_max_cut_AA
theta_min_cut_arcmin = 3.0
data = DESI_DR2(fname, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA,
                                 theta_min_cut_arcmin=theta_min_cut_arcmin)
z = data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))
if rescale_cov:
    data.cov_ZAM *= 1.0 / Nm

# %%
# use priors based on previous fits above
par_bias = copy_with_prior(par_bias, true_cont_mini)
par_beta = copy_with_prior(par_beta, true_cont_mini)
par_kp = copy_with_prior(par_kp, true_cont_mini)
par_kC = copy_with_prior(par_kC, unco_mini)
par_pC = copy_with_prior(par_pC, unco_mini)

# %%
update_theory = True
theory_config = {'verbose': False, 'default_lya_model': default_lya_model,
                 'include_continuum': True, 'include_hcd': True,
                 'include_metal': True, 'include_sky': False}
if update_theory:
    for par in [par_bias, par_beta, par_kp, par_kC, par_pC]:
        theory_config[par.name] = par.ini_value
theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
print(theory.lya_model.default_lya_params)
print(theory.cont_model.default_continuum_params)
print(theory.cont_model.default_hcd_params)
print(theory.cont_model.default_metal_params)
like = Likelihood(data=data, theory=theory, iz=iz, config={'verbose':False})

# %%
# add extra params for metals and HCDs
ini_bH = theory.cont_model.default_hcd_params['b_H']
ini_LH = theory.cont_model.default_hcd_params['L_H_Mpc']
ini_bX = theory.cont_model.default_metal_params['b_X']
par_bH = FreeParameter(
    name='b_H',
    min_value=-1.0,
    max_value=1.0,
    ini_value=ini_bH,
    delta=1e-4
)
par_LH = FreeParameter(
    name='L_H_Mpc',
    min_value=0.01,
    max_value=20.0,
    ini_value=ini_LH,
    delta=0.01
)
par_bX = FreeParameter(
    name='b_X',
    min_value=-1.0,
    max_value=0.0,
    ini_value=ini_bX,
    delta=1e-5
)
#free_params = [par_bias, par_beta, par_kp, par_kC, par_pC, par_bH, par_bX]
ignore_hcds = True
if ignore_hcds:
    like.theory.cont_model.default_hcd_params['b_H']=0.0
    like.theory.include_hcd = False
    free_params = [par_bias, par_beta, par_bX]
else:
    free_params = [par_bias, par_beta, par_bH, par_LH, par_bX]
for par in free_params:
    if par.gauss_prior_mean is None:
        print(par.name, par.ini_value, 'no prior')
    else:
        print(par.name, par.ini_value, 'prior', par.gauss_prior_mean , '+/-', par.gauss_prior_width)

# %%
post = Posterior(like, free_params, config={'verbose':False})
cont_mini = Minimizer(post, config={'verbose':False})
cont_mini.silence()
start = time.time()
cont_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
cont_mini.print_results()

# %%
true_cont_mini.print_results()

# %%
unco_mini.print_results()

# %%
cont_mini.minimizer

# %%
cont_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True)

# %% [markdown]
# ### Compare measurement of stacks

# %%
for it_A, theta in enumerate(data.theta_centers_arcmin):
    plt.figure()
    for mini in [unco_mini, cont_mini]:
        data = mini.post.like.data
        k_M = data.k_M_centers_AA
        err = np.diag(np.squeeze(data.cov_ZAM[iz, it_A, :, :]))**0.5
        px = data.Px_ZAM[iz, it_A, :]
        plt.errorbar(k_M, px, err)
    plt.title(r'$\theta = {:.1f}$ arcmin'.format(theta))
    plt.xlabel(r'$k_\parallel$ [1/A]')
    plt.ylabel(r'$P_\times(k_\parallel, \theta)$ [A]')
    plt.show()

# %%
