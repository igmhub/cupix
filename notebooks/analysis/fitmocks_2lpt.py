# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
from cupix.inference.free_parameter import FreeParameter
from cupix.inference.posterior import Posterior
from cupix.inference.minimize_posterior import Minimizer
from cupix.inference.sampling_funcs import prepare_free_parameters

# %%
# In this notebook we will work with a single z bin
iz=0

# setup cosmology (should check this is the right cosmology in the mocks)
cosmodict = {'ombh2':0.02237, 'omch2':0.12, 'omk':0, 'h':0.6736, 'As':2.0830e-9, 'ns':0.9649, 'w0':-1, 'wa':0}

cosmo = cosmology.Cosmology(cosmodict)
# starting point for Lya bias parameters in mocks
# default_lya_model = 'pressure_only_arinyo_from_colore'
default_lya_model = 'best_fit_arinyo_from_colore'

# path to mocks
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/colore/"


# %% [markdown]
# # Fit true-continuum mocks

# %%
fname = mockdir + "analysis-200/tru_cont/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"
# speed-up code by only looking at low kpar
kM_max_cut_AA = 0.7
km_max_cut_AA = 1.1*kM_max_cut_AA
theta_min_cut_arcmin = 10.0
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
trucont_data = DESI_DR2(data_config)
z = trucont_data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))


# %%
theory_config = {'verbose': True, 'default_lya_model': default_lya_model, 'include_continuum':False}
theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)

# %%
like = Likelihood(data=trucont_data, theory=theory, iz=iz, config={'verbose':False})

# %%
like.plot_px(params={'bias':-.13, 'beta':1.4, 'kp_Mpc':1.2, 'av':.1, 'bv':.1}, every_other_theta=True, multiply_by_k=False)

# %%
# # set broader priors than the defaults, since IFAE-QL mocks are different

params_config = {
    'bias': {'gauss_prior_width': None, 'gauss_prior_mean':None},
    'beta': {'gauss_prior_width': None, 'gauss_prior_mean':None},
    'q1': {'gauss_prior_width': None, 'gauss_prior_mean':None},
}

#     'kp_Mpc': {'gauss_prior_width': None, 'gauss_prior_mean':None},
#     'q1': {'gauss_prior_width': None, 'gauss_prior_mean':None},
#     'av': {'gauss_prior_width': None, 'gauss_prior_mean':None},
#     'bv': {'gauss_prior_width': None, 'gauss_prior_mean':None},
#

free_params = prepare_free_parameters(['bias','beta', 'q1'], theory, theory_config, params_config=params_config)
# free_params = prepare_free_parameters(['bias','beta','kp_Mpc', 'q1', 'av', 'bv'], theory, theory_config, params_config=params_config)
for p in free_params:
    # print all attributes
    print(p.__dict__)
    # set initial value for bias / beta based on best-fit values from Laura
# ini_bias = theory.lya_model.default_lya_params['bias']
# ini_beta = theory.lya_model.default_lya_params['beta']
# par_bias = FreeParameter(
#     name='bias',
#     min_value=-0.5,
#     max_value=-0.01,
#     ini_value=ini_bias,
#     delta=0.01,   
# )
# par_beta = FreeParameter(
#     name='beta',
#     min_value=0.1,
#     max_value=5.0,
#     ini_value=ini_beta,
#     delta=0.1,
# )

# # add other free parameters (without prior values for now)
# ini_q1 = theory.lya_model.default_lya_params['q1']
# par_q1 = FreeParameter(
#     name='q1',
#     min_value=0.0,
#     max_value=5.0,
#     ini_value=ini_q1,
#     delta=0.01
# )
# ini_av = theory.lya_model.default_lya_params['av']
# par_av = FreeParameter(
#     name='av',
#     min_value=0.0,
#     max_value=5.0,
#     ini_value=ini_av,
#     delta=0.01
# )
# ini_bv = theory.lya_model.default_lya_params['bv']
# par_bv = FreeParameter(
#     name='bv',
#     min_value=0.0,
#     max_value=5.0,
#     ini_value=ini_bv,
#     delta=0.01
# )
# ini_kp = theory.lya_model.default_lya_params['kp_Mpc']
# par_kp = FreeParameter(
#     name='kp_Mpc',
#     min_value=0.0,
#     max_value=5.0,
#     ini_value=ini_kp,
#     delta=0.01
# )
# ini_kv = theory.lya_model.default_lya_params['kv_Mpc']
# par_kv = FreeParameter(
#     name='kv_Mpc',
#     min_value=0.0,
#     max_value=5.0,
#     ini_value=ini_kv,
#     delta=0.01
# )

# %%
# free_params = [par_bias, par_beta, par_q1, par_av, par_bv, par_kp, par_kv]

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
true_cont_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
true_cont_mini.plot_ellipses('bias','beta')
true_cont_mini.plot_corner()

# %% [markdown]
# # Fit uncontaminated mocks

# %%
fname = mockdir + "analysis-200/uncontaminated/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"

# speed-up code by only looking at low kpar
kM_max_cut_AA = 0.7
km_max_cut_AA = 1.1*kM_max_cut_AA
theta_min_cut_arcmin = 5.0
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
uncont_data = DESI_DR2(data_config)
z = uncont_data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))


# %%
theory_config = {'verbose': False, 'default_lya_model': default_lya_model, 'include_continuum':True}
uncont_theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
print(uncont_theory.lya_model.default_lya_params)

# %%
like = Likelihood(data=uncont_data, theory=uncont_theory, iz=iz, config={'verbose':False})

# %% [markdown]
# Let's see how Laura's best-fit to IFAE-QL compares just for curiosity sake

# %%
like.plot_px(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
params_config = {'bias': {'min_value': -0.3, 'max_value': -0.01}, 'beta':{'min_value':0.1, 'max_value':3.0}, 'kp_Mpc':{'min_value':0.0, 'max_value':5.0}}
free_params = prepare_free_parameters(['bias','beta','kp_Mpc'], uncont_theory, theory_config, params_config=params_config) 
for p in free_params:
    print(p.name, p.true_value, p.ini_value, p.delta, p.min_value, p.max_value)

# %%
post = Posterior(like, free_params, config={'verbose':False})
uncont_mini = Minimizer(post, config={'verbose':False})
uncont_mini.silence()
start = time.time()
uncont_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
uncont_mini.print_results()

# %%
uncont_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
uncont_mini.plot_ellipses('bias', 'beta')

# %%
uncont_mini.plot_ellipses('bias', 'kp_Mpc')

# %% [markdown]
# # Now use bias and beta and try to fit b_HCD from partially-contaminated mocks

# %%
mask_dlas = True
if mask_dlas:
    maskstr = 'nomask'
else:
    maskstr = 'mask'

# %%
fname = mockdir + f"partially_contaminated/analysis-200/onlydlas_{maskstr}/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"

# speed-up code by only looking at low kpar
kM_max_cut_AA = 0.7
km_max_cut_AA = 1.1*kM_max_cut_AA
theta_min_cut_arcmin = 5.0
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
data_parcon_wmask = DESI_DR2(data_config)
z = data_parcon_wmask.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))


# %%
theory_config = {'verbose': True, 'default_lya_model': default_lya_model,
                 'include_continuum': True, 'include_hcd': True,
                 'include_metal': False, 'include_sky': False, 'bias':uncont_mini.get_best_fit_params()['bias'], 'beta':uncont_mini.get_best_fit_params()['beta'], 'kp_Mpc':uncont_mini.get_best_fit_params()['kp_Mpc']}
theory_parcon_wmask = Theory(z=z, fid_cosmo=cosmo, config=theory_config)


# %%
params_config = {'b_H': {'ini_value':-.04, 'min_value': -0.2, 'max_value': 0, 'delta':.002}, 'L_H_Mpc':{'min_value':0, 'max_value':20.0, 'ini_value':5, 'delta':.5}}
free_params = prepare_free_parameters(['b_H','L_H_Mpc'], theory_parcon_wmask, theory_config, params_config=params_config)

for p in free_params:
    print(p.name, p.true_value, p.ini_value, p.delta, p.min_value, p.max_value)

# %%
like_parcon_wmask = Likelihood(data=data_parcon_wmask, theory=theory_parcon_wmask, iz=iz, config={'verbose':True})
post_parcon_wmask = Posterior(like_parcon_wmask, free_params, config={'verbose':False})
parcon_wmask_mini = Minimizer(post_parcon_wmask, config={'verbose':False})
parcon_wmask_mini.silence()
start = time.time()
parcon_wmask_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
parcon_wmask_mini.print_results()

# %%
parcon_wmask_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
parcon_wmask_mini.plot_ellipses('b_H', 'L_H_Mpc')

# %% [markdown]
# # Now free bias, beta, kp and try to fit HCD params from partially-contaminated mocks

# %%
params_config = {'b_H': {'ini_value':-.04, 'min_value': -0.2, 'max_value': 0, 'delta':.002}, 'L_H_Mpc':{'min_value':0, 'max_value':20.0, 'ini_value':5, 'delta':.5},
                 'bias': {'min_value': -0.3, 'max_value': -0.01}, 'beta':{'min_value':0.1, 'max_value':3.0}, 'kp_Mpc':{'min_value':0.0, 'max_value':5.0}}
free_params = prepare_free_parameters(['bias','beta','kp_Mpc','b_H','L_H_Mpc'], theory_parcon_wmask, theory_config, params_config=params_config)

for p in free_params:
    print(p.name, p.true_value, p.ini_value, p.delta, p.min_value, p.max_value)

# %%
post_parcon_wmask_free = Posterior(like_parcon_wmask, free_params, config={'verbose':False})
parcon_wmask_free_mini = Minimizer(post_parcon_wmask_free, config={'verbose':False})
parcon_wmask_free_mini.silence()
start = time.time()
parcon_wmask_free_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
parcon_wmask_free_mini.print_results()

# %%
parcon_wmask_free_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
parcon_wmask_free_mini.plot_ellipses('b_H', 'L_H_Mpc')

# %% [markdown]
# # The same but with the addition of beta_H

# %%
params_config = {'beta_H': {'ini_value':.4, 'min_value': 0, 'max_value': 5, 'delta':.1},
                 'b_H': {'ini_value':-.04, 'min_value': -0.2, 'max_value': 0, 'delta':.002},
                   'L_H_Mpc':{'min_value':0, 'max_value':20.0, 'ini_value':5, 'delta':.5},
                 'bias': {'min_value': -0.3, 'max_value': -0.01}, 'beta':{'min_value':0.1, 'max_value':3.0},
                   'kp_Mpc':{'min_value':0.0, 'max_value':5.0}}
free_params = prepare_free_parameters(['bias','beta','kp_Mpc','beta_H','b_H','L_H_Mpc'], theory_parcon_wmask, theory_config, params_config=params_config)

for p in free_params:
    print(p.name, p.true_value, p.ini_value, p.delta, p.min_value, p.max_value)

# %%
post_parcon_wmask_freebeta = Posterior(like_parcon_wmask, free_params, config={'verbose':False})
parcon_wmask_freebeta_mini = Minimizer(post_parcon_wmask_freebeta, config={'verbose':False})
parcon_wmask_freebeta_mini.silence()
start = time.time()
parcon_wmask_freebeta_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
parcon_wmask_freebeta_mini.print_results()

# %%
parcon_wmask_freebeta_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
parcon_wmask_freebeta_mini.plot_ellipses('L_H_Mpc', 'beta_H')

# %% [markdown]
# # And the last step as well for unmasked DLAs
#

# %%
mask_dlas = False
if mask_dlas:
    maskstr = 'nomask'
else:
    maskstr = 'mask'

fname = mockdir + f"partially_contaminated/analysis-200/onlydlas_{maskstr}/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"

# speed-up code by only looking at low kpar
kM_max_cut_AA = 0.7
km_max_cut_AA = 1.1*kM_max_cut_AA
theta_min_cut_arcmin = 5.0
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
data_parcon_nomask = DESI_DR2(data_config)
z = data_parcon_nomask.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))

like_parcon_nomask = Likelihood(data=data_parcon_nomask, theory=theory_parcon_wmask, iz=iz, config={'verbose':True})

# %%
params_config = {'b_H': {'ini_value':-.04, 'min_value': -0.2, 'max_value': 0, 'delta':.002}, 'L_H_Mpc':{'min_value':0, 'max_value':20., 'ini_value':10, 'delta':.5},
                 'bias': {'min_value': -0.2, 'max_value': -0.01}, 'beta':{'min_value':0.1, 'max_value':5.0}, 'kp_Mpc':{'min_value':0.0, 'max_value':5.0}}
free_params = prepare_free_parameters(['bias','beta','kp_Mpc','b_H','L_H_Mpc'], theory_parcon_wmask, theory_config, params_config=params_config)

for p in free_params:
    print(p.name, p.true_value, p.ini_value, p.delta, p.min_value, p.max_value)

# %%
post_parcon_nomask = Posterior(like_parcon_nomask, free_params, config={'verbose':False})
parcon_nomask_mini = Minimizer(post_parcon_nomask, config={'verbose':False})
parcon_nomask_mini.silence()
start = time.time()
parcon_nomask_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
parcon_nomask_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
parcon_nomask_mini.print_results()

# %%
xrange = [-0.18, -0.1]
yrange = [1, 2.5]
ax = parcon_nomask_mini.plot_ellipses('bias', 'beta', xrange=xrange, yrange=yrange, extralabel='DLA cont, no mask')
parcon_wmask_free_mini.plot_ellipses('bias', 'beta', ax=ax, color='orange', xrange=xrange, yrange=yrange, extralabel='DLA cont, with mask')
uncont_mini.plot_ellipses('bias', 'beta', ax=ax, color='green', xrange=xrange, yrange=yrange, extralabel='uncont')

# %%
xrange = [-.1,-.15]
yrange = [5,30]
ax = parcon_nomask_mini.plot_ellipses('bias', 'L_H_Mpc', xrange=xrange, yrange=yrange, extralabel='DLA cont, no mask')
parcon_wmask_free_mini.plot_ellipses('bias', 'L_H_Mpc', ax=ax, color='orange', xrange=xrange, yrange=yrange, extralabel='DLA cont, with mask')

# %%
# plot bias and beta together from uncontaminated and partially contaminated fits
# let's focus on the no-DLA-masking and DLA-masking cases

