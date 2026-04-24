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
# # Compare data vs two theories

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import time
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
# ### Setup data

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
rescale_cov = True
Nm = 50
#fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg{}.hdf5".format(Nm)
fname = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg{}.hdf5".format(Nm)

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


# %% [markdown]
# ### Setup theory, likelihood, posterior and minimizer

# %%
def get_free_params(theory):
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
    
    if theory.include_continuum:
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
        free_params.append(par_kC)
        free_params.append(par_pC)
        
    return free_params


# %%
def get_analysis(include_continuum):
    # set theory
    theory_config = {'verbose': False, 'default_lya_model': default_lya_model, 'include_continuum': include_continuum}
    theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
    print(theory.lya_model.default_lya_params)
    if include_continuum:
        print(theory.cont_model.default_continuum_params)

    # set free params
    free_params = get_free_params(theory)
    for par in free_params:
        print(par.name, par.ini_value)

    # set likelihood, posterior and minimizer
    like = Likelihood(data=data, theory=theory, iz=iz, config={'verbose':False})
    post = Posterior(like, free_params, config={'verbose':False})
    mini = Minimizer(post, config={'verbose':False})

    return mini


# %%
mini_with_cont = get_analysis(include_continuum=True)

# %%
mini_no_cont = get_analysis(include_continuum=False)

# %%
for mini in [mini_with_cont, mini_no_cont]:
    mini.silence()
    start = time.time()
    mini.minimize()
    end = time.time()
    print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
for mini in [mini_with_cont, mini_no_cont]:
    mini.print_results()
    mini.plot_best_fit(multiply_by_k=False, every_other_theta=True, residual_to_theory=True)

# %% [markdown]
# ### Plot two alternative theories, one at a time

# %%
best_fit_params = mini_with_cont.get_best_fit_params()
print(best_fit_params)

# %%
best_fit_info = {'label': 'best-fit model', 'params': best_fit_params }
new_params = dict(best_fit_params)
new_params['kC_Mpc'] = 1e-20
new_info = {'label': 'no distortion', 'params': new_params }

# %%
for info in [best_fit_info, new_info]:
    params = info['params']
    theorylabel = info['label']
    like = mini_with_cont.post.like
    chi2 = like.get_chi2(params=params)
    print(theorylabel, params, chi2)
    datalabel = 'uncon mocks (z={}), chi2={:.2f}'.format(z, chi2)
    like.plot_px(multiply_by_k=False, every_other_theta=True, residual_to_theory=True, params=params,
                 theorylabel=theorylabel, datalabel=datalabel)

# %%
theorylabel = 'best-fit model'
datalabel = 'uncontaminated mocks (z={})'.format(z)
extra_params = dict(best_fit_params)
extra_params['kC_Mpc'] = 1e-20
mini_with_cont.plot_best_fit(multiply_by_k=False, every_other_theta=True, 
                             residual_to_theory=True,
                             theorylabel=theorylabel, datalabel=datalabel,
                             extra_params=extra_params, extra_label='no distortion (no new fit)')

# %%
extra_params = mini_no_cont.get_best_fit_params()
extra_params['kC_Mpc'] = 1e-20
mini_with_cont.plot_best_fit(multiply_by_k=False, every_other_theta=True, 
                             residual_to_theory=True,
                             theorylabel=theorylabel, datalabel=datalabel,
                             extra_params=extra_params, extra_label='no distortion (new fit)')

# %%
