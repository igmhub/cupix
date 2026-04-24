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
# # Fitting the stack of uncontaminated mocks
#
# We vary bias / beta / kp_Mpc, but now also pC and kC_Mpc

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

# %%
# path to mocks
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
#fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
fname = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
# dummy data object, only to get the redshift of interest
iz = 1
dummy_data = DESI_DR2(fname)
z = dummy_data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))

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
free_params = [par_bias, par_beta, par_kC, par_pC]
for par in free_params:
    print(par.name, par.ini_value)

# %%
# speed-up code by only looking at low kpar (should be enough for theta > 10 arcmin or so)
kM_max_cut_AA=0.5
km_max_cut_AA=1.1*kM_max_cut_AA

# %%
runs = []
for theta in [0.5, 1.0, 2.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0]:
    run = {}
    run['data'] = DESI_DR2(fname, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA, theta_min_cut_arcmin=theta)
    run['theta_min'] = run['data'].theta_min_a_arcmin[0]
    run['theory'] = theory
    run['like'] = Likelihood(data=run['data'], theory=run['theory'], iz=iz, config={'verbose':False})
    run['post'] = Posterior(run['like'], free_params, config={'verbose':False})
    run['mini'] = Minimizer(run['post'], config={'verbose':False})
    runs.append(run)

# %%
for ii, run in enumerate(runs):
    theta_min = run['theta_min']
    print('-----------------------------')
    print('minimize theta_min = {:.3f} arcmin'.format(theta_min))
    run['mini'].silence()
    run['mini'].minimize()
    run['mini'].print_results()

# %%
theta_min = [ run['theta_min'] for run in runs]
bias = [ run['mini'].get_best_fit_value('bias', return_hesse=False) for run in runs]
bias_err = [ run['mini'].get_best_fit_value('bias', return_hesse=True)[1] for run in runs]
beta = [ run['mini'].get_best_fit_value('beta', return_hesse=False) for run in runs]
beta_err = [ run['mini'].get_best_fit_value('beta', return_hesse=True)[1] for run in runs]
chi2 = [ run['mini'].get_best_fit_chi2() for run in runs]
ndf = [ run['mini'].post.get_ndf() for run in runs]

# %%
plt.plot(theta_min, chi2, label='best-fit chi2')
plt.plot(theta_min, ndf, label='degrees of freedom')
plt.legend()
plt.xlabel(r'$\theta_{\rm min}$ [arcmin]');


# %%
def plot_param(pname, runs):
    theta_min = [ run['theta_min'] for run in runs]
    val = [ run['mini'].get_best_fit_value(pname, return_hesse=False) for run in runs]
    err = [ run['mini'].get_best_fit_value(pname, return_hesse=True)[1] for run in runs]
    plt.figure()
    plt.errorbar(theta_min, val, err)
    plt.ylabel(pname)
    plt.xlabel(r'$\theta_{\rm min}$ [arcmin]');
    # check if prior was set
    post = runs[0]['post']
    ip = post.get_param_index(param_name=pname)
    par = post.free_params[ip]
    if par.gauss_prior_mean is not None:
        mean = par.gauss_prior_mean
        rms = par.gauss_prior_width
        plt.axhspan(mean-rms, mean+rms, alpha=0.2)
    plt.tight_layout()
    if 'pressure_only' in default_lya_model:
        plot_fname = 'pressure_only_uncont'
    else:
        plot_fname = 'dnl_uncont'
    plot_fname += '_{}_{:.2f}.png'.format(pname, z)
    plt.savefig(plot_fname)


# %%
param_names = [par.name for par in free_params]
for pname in param_names:
    plot_param(pname, runs)

# %%
for ii, run in enumerate(runs):
    theta_min = run['theta_min']
    print('-----------------------------')
    print('minimize theta_min = {:.3f} arcmin'.format(theta_min))
    datalabel='True continuum stack (z = {}, theta > {:.2f})'.format(z, theta_min)
    run['mini'].plot_best_fit(multiply_by_k=False, every_other_theta=True, datalabel=datalabel)

# %%
