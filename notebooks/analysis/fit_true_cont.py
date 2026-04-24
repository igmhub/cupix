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
# # Fitting the stack of true-continuum mocks

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
fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
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
theory_config = {'verbose': False, 'default_lya_model': default_lya_model, 'include_continuum': False}
theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
print(theory.lya_model.default_lya_params)

# %%
# set initial value for bias / beta based on best-fit values from Laura
ini_bias = theory.lya_model.default_lya_params['bias']
ini_beta = theory.lya_model.default_lya_params['beta']
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
free_params = [par_bias, par_beta]
for par in free_params:
    print(par.name, par.ini_value)

# %% [markdown]
# ## Step 1: fit bias / beta with different theta cuts

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
plt.errorbar(theta_min, bias, bias_err)
plt.ylabel(r'$b_\alpha$')
plt.xlabel(r'$\theta_{\rm min}$ [arcmin]');

# %%
plt.errorbar(theta_min, beta, beta_err)
plt.ylabel(r'$\beta_\alpha$')
plt.xlabel(r'$\theta_{\rm min}$ [arcmin]');

# %% [markdown]
# ## Fit D_NL parameters using smaller scales 
#
# We will use a prior on bias / beta from theta_min = 10 arcmin

# %%
theta_min = [ run['theta_min'] for run in runs]
i_large = theta_min.index(10.0)
run_large = runs[i_large]

# %%
add_priors = False
if add_priors:
    for par in free_params:
        pname = par.name
        val, err = run_large['mini'].get_best_fit_value(pname, return_hesse=True)
        print(pname, val, err)
        par.gauss_prior_mean = val
        par.gauss_prior_width = err

# %%
# add other free parameters (without prior values for now)
ini_q1 = theory.lya_model.default_lya_params['q1']
par_q1 = FreeParameter(
    name='q1',
    min_value=0.0,
    max_value=5.0,
    ini_value=ini_q1,
    delta=0.01
)
ini_av = theory.lya_model.default_lya_params['av']
par_av = FreeParameter(
    name='av',
    min_value=0.0,
    max_value=5.0,
    ini_value=ini_av,
    delta=0.01
)
ini_bv = theory.lya_model.default_lya_params['bv']
par_bv = FreeParameter(
    name='bv',
    min_value=0.0,
    max_value=5.0,
    ini_value=ini_bv,
    delta=0.01
)
ini_kp = theory.lya_model.default_lya_params['kp_Mpc']
par_kp = FreeParameter(
    name='kp_Mpc',
    min_value=0.0,
    max_value=5.0,
    ini_value=ini_kp,
    delta=0.01
)
ini_kv = theory.lya_model.default_lya_params['kv_Mpc']
par_kv = FreeParameter(
    name='kv_Mpc',
    min_value=0.0,
    max_value=5.0,
    ini_value=ini_kv,
    delta=0.01
)

# %%
free_q1 = False
free_av = False
free_bv = False
free_kp = True
free_kv = False
if free_q1: free_params.append(par_q1)
if free_av: free_params.append(par_av)
if free_bv: free_params.append(par_bv)
if free_kp: free_params.append(par_kp)
if free_kv: free_params.append(par_kv)

# %%
for par in free_params:
    print(par.name, par.ini_value)

# %%
runs_DNL = []
for theta in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    run = {}
    run['data'] = DESI_DR2(fname, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA, theta_min_cut_arcmin=theta)
    run['theta_min'] = run['data'].theta_min_a_arcmin[0]
    run['theory'] = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
    run['like'] = Likelihood(data=run['data'], theory=run['theory'], iz=iz, config={'verbose':False})
    run['post'] = Posterior(run['like'], free_params, config={'verbose':False})
    run['mini'] = Minimizer(run['post'], config={'verbose':False})
    runs_DNL.append(run)

# %%
theory.lya_model.default_lya_params

# %%
for ii, run in enumerate(runs_DNL):
    theta_min = run['theta_min']
    print('-----------------------------')
    print('minimize theta_min = {:.3f} arcmin'.format(theta_min))
    run['mini'].silence()
    run['mini'].minimize()
    run['mini'].print_results()

# %%
theta_min = [ run['theta_min'] for run in runs_DNL]
chi2 = [ run['mini'].get_best_fit_chi2() for run in runs_DNL]
ndf = [ run['mini'].post.get_ndf() for run in runs_DNL]
plt.plot(theta_min, chi2, label='best-fit chi2')
plt.plot(theta_min, ndf, label='degrees of freedom')
plt.legend()
plt.xlabel(r'$\theta_{\rm min}$ [arcmin]');


# %%
def plot_param(pname, runs, label):
    theta_min = [ run['theta_min'] for run in runs]
    val = [ run['mini'].get_best_fit_value(pname, return_hesse=False) for run in runs]
    err = [ run['mini'].get_best_fit_value(pname, return_hesse=True)[1] for run in runs]
    plt.figure()
    plt.errorbar(theta_min, val, err)
    plt.ylabel(pname)
    plt.xlabel(r'$\theta_{\rm min}$ [arcmin]');
    # check if prior was set
    post = runs_DNL[0]['post']
    ip = post.get_param_index(param_name=pname)
    par = post.free_params[ip]
    if par.gauss_prior_mean is not None:
        mean = par.gauss_prior_mean
        rms = par.gauss_prior_width
        plt.axhspan(mean-rms, mean+rms, alpha=0.2)
    plt.tight_layout()
    if 'pressure_only' in default_lya_model:
        plot_fname = 'pressure_only_true_cont'
    else:
        plot_fname = 'dnl_true_cont'
    plot_fname += '_{}_{}_{:.2f}.png'.format(pname, label, z)
    plt.savefig(plot_fname)


# %%
param_names = [par.name for par in free_params]
for pname in param_names:
    plot_param(pname, runs_DNL, label='DNL')

# %%
for ii, run in enumerate(runs_DNL):
    theta_min = run['theta_min']
    print('-----------------------------')
    print('minimize theta_min = {:.3f} arcmin'.format(theta_min))
    theorylabel=""
    datalabel='True continuum stack (z = {}, theta > {:.2f})'.format(z, theta_min)
    for key, par in run['mini'].get_best_fit_params().items():
        theorylabel += "{} = {:.3f}   ".format(key, par)
    run['mini'].plot_best_fit(multiply_by_k=False, every_other_theta=True, 
                              theorylabel=theorylabel, datalabel=datalabel)

# %%
