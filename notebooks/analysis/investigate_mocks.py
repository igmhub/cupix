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
# # Investigate the impact of contaminated weights on the measurements from uncontaminated mocks

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
from cupix.parameter_inference.posterior import Posterior
from cupix.parameter_inference.minimize_posterior import Minimizer

# %%
# speed-up code by only looking at low kpar (should be enough for theta > 10 arcmin or so)
kM_max_cut_AA=0.5
km_max_cut_AA=1.1*kM_max_cut_AA
theta_min_cut_arcmin=3.0

# %%
# path to mocks
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-49/"
fname_1 = mockdir + "uncontaminated/bf3_binned_out_px-zbins_4-thetabins_20_w_res.hdf5"
data_1 = DESI_DR2(fname_1, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA, theta_min_cut_arcmin=theta_min_cut_arcmin)
fname_2 = mockdir + "uncont_w_contweights/bf3_binned_out_px-zbins_4-thetabins_20_w_res.hdf5"
data_2 = DESI_DR2(fname_2, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA, theta_min_cut_arcmin=theta_min_cut_arcmin)
# dummy data object, only to get the redshift of interest
iz = 1
z = data_1.z[iz]
assert z== data_2.z[iz]
print('analyze zbin {}, at z = {}'.format(iz, z))

# %%
# setup cosmology (should check this is the right cosmology in the mocks)
cosmo = cosmology.Cosmology()
# starting point for Lya bias parameters in mocks
default_lya_model = 'pressure_only_fits_from_colore'
#default_lya_model = 'best_fit_arinyo_from_colore'
theory_config = {'verbose': False, 'default_lya_model': default_lya_model, 'include_continuum': True}
theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
print(theory.lya_model.default_lya_params)
print(theory.cont_model.default_continuum_params)

# %%
# set initial value for bias / beta based on best-fit values from Laura
ini_bias = theory.lya_model.default_lya_params['bias']
ini_beta = theory.lya_model.default_lya_params['beta']
ini_kC = theory.cont_model.default_continuum_params['kC_Mpc']
ini_pC = theory.cont_model.default_continuum_params['pC']
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
par_kC = FreeParameter(
    name='kC_Mpc',
    min_value=1e-4,
    max_value=1e-1,
    ini_value=ini_kC,
    delta=0.001
)
par_pC = FreeParameter(
    name='pC',
    min_value=0.01,
    max_value=2.0,
    ini_value=ini_pC,
    delta=0.01
)
#free_params = [par_bias, par_beta, par_kC, par_pC]
free_params = [par_bias, par_beta]
for par in free_params:
    print(par.name, par.ini_value)

# %%
datas = [data_1, data_2]

# %%
minis = []
for data in datas:
    like = Likelihood(data=data, theory=theory, iz=iz, config={'verbose':False})
    post = Posterior(like, free_params, config={'verbose':False})
    mini = Minimizer(post, config={'verbose':False})
    minis.append(mini)

# %%
for mini in minis:
    print('-----------')
    mini.silence()
    mini.minimize()
    mini.print_results()

# %%
bias = [ mini.get_best_fit_value('bias', return_hesse=False) for mini in minis]
bias_err = [ mini.get_best_fit_value('bias', return_hesse=True)[1] for mini in minis]
beta = [ mini.get_best_fit_value('beta', return_hesse=False) for mini in minis]
beta_err = [ mini.get_best_fit_value('beta', return_hesse=True)[1] for mini in minis]

# %%
plt.errorbar(bias[0], beta[0], beta_err[0], bias_err[0], fmt='None', color='blue', label='original weights')
plt.errorbar(bias[1], beta[1], beta_err[1], bias_err[1], fmt='None', color='red', label='contaminated weights')
plt.legend()
plt.xlabel('bias')
plt.ylabel('beta')
plt.savefig('investigate_mocks.png');

# %%
for mini in minis:
    mini.plot_best_fit(multiply_by_k=False, every_other_theta=True)

# %%

# %%
