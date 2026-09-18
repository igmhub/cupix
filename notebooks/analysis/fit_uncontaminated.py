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
from cupix.inference.free_parameter import FreeParameter
from cupix.inference.posterior import Posterior
from cupix.inference.minimize_posterior import Minimizer

# %%
# path to mocks
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
#fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
fname = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
# dummy data object, only to get the redshift of interest
iz = 1
dummy_data = DESI_DR2(config={'data_file':fname})
z = dummy_data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))

# %%
# setup cosmology (should check this is the right cosmology in the mocks)
cosmo = cosmology.Cosmology()
# starting point for Lya bias parameters in mocks
default_lya_model = 'pressure_only_arinyo_from_colore'
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
    run['data'] = DESI_DR2(config={'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta})
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
for ii, run in enumerate(runs):
    run['mini'].save_results(outdir="/pscratch/sd/m/mlokken/desi-lya/px/mocks/minimizer_fits/20260915_uncont_free_cont_thetamin_vars", outfile="minimizer_fit_thetamin_{:.3f}_zbin_{}.hdf5".format(run['theta_min'], iz))

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
plt.plot(theta_min, np.array(chi2)/np.array(ndf), label=r'$\chi^2 / {\rm dof}$')   
plt.xlabel(r'$\theta_{\rm min}$ [arcmin]');
plt.ylabel(r'$\chi^2 / {\rm dof}$');


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

# %% [markdown]
# ## Compare the fits with actual curve

# %%
# path to mocks
mockdir_tru = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
# dummy data object, only to get the redshift of interest
iz = 1
dummy_data = DESI_DR2(config={'data_file':fname})
z = dummy_data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))

# %%
data_tru =  DESI_DR2(config={'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'kM_min_cut_AA':0.0, 'theta_min_cut_arcmin':.63})
data_uncont = DESI_DR2(config={'data_file':mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5", 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'kM_min_cut_AA':0.0, 'theta_min_cut_arcmin':.63})

# %%
theta_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
for itheta in [0,3,6,8]:
    plt.plot(data_uncont.k_M_centers_AA, data_uncont.Px_ZAM[iz, itheta, :]/data_tru.Px_ZAM[iz,itheta,:],  color=theta_colors[int(itheta/2)])
    # plot the model
    kC_Mpc = runs[itheta]['mini'].get_best_fit_value('kC_Mpc')
    pC = runs[itheta]['mini'].get_best_fit_value('pC')
    print(kC_Mpc, pC)
    cont_dist = np.tanh( (runs[0]['data'].k_M_centers_AA/kC_Mpc)**pC )
    plt.plot(runs[0]['data'].k_M_centers_AA, cont_dist, label=r'$\theta_{{min}}$ = {:.2f}'.format(runs[itheta]['data'].theta_min_a_arcmin[0]), linestyle='--', color=theta_colors[int(itheta/2)])
plt.xlim([0,.3])
plt.ylim([0.8, 1.03])


plt.ylabel(r'$P_\times^{\rm cont-fitted} / P_\times^{\rm true-cont} $')
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
# add solid vs dashed to legend
handles, labels = plt.gca().get_legend_handles_labels()
dashed_line = plt.Line2D([], [], color='k', linestyle='--')
solid_line = plt.Line2D([], [], color='k', linestyle='-')
labels_lines = [r'mock data at $\theta=\theta_{{min}}$', r'model fitted to $\theta>\theta_{{min}}$']
plt.legend(handles=[solid_line, dashed_line] + handles, labels=labels_lines + labels, loc='lower center', fontsize=14, ncol=1)
plt.title("$z={:.1f}$".format(z))

# %%
ax

# %%
fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
for iz in range(4):
    mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
    fname_tru = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
    fname_uncont = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
    data_tru =  DESI_DR2(config={'data_file':fname_tru, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'kM_min_cut_AA':0.0, 'theta_min_cut_arcmin':.63})
    data_uncont = DESI_DR2(config={'data_file':fname_uncont, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'kM_min_cut_AA':0.0, 'theta_min_cut_arcmin':.63})
    z = data_tru.z[iz]
    theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
    pC = theory.get_param('pC')
    kC_Mpc = theory.get_param('kC_Mpc')
    print(pC, kC_Mpc)
    if iz==0:
        pC = 0.58
    elif iz==1:
        pC = 0.5
    elif iz==2:
        pC = 0.5
    elif iz==3:
        pC = 0.51
        kC_Mpc = .011
    # plot the ratio and the model
    
    ax[iz//2, iz%2].set_title("$z={:.1f}$".format(z))
    for itheta in np.arange(9):
        ax[iz//2, iz%2].plot(data_uncont.k_M_centers_AA, data_uncont.Px_ZAM[iz, itheta, :]/data_tru.Px_ZAM[iz,itheta,:],  color=theta_colors[int(itheta)], alpha=.3)
    cont_dist = np.tanh((data_uncont.k_M_centers_AA/kC_Mpc)**pC)
    ax[iz//2, iz%2].plot(data_uncont.k_M_centers_AA, cont_dist, label='model', linestyle='--', color='k')
    
plt.xlim([0,.3])
plt.ylim([0.8, 1.03])
plt.legend()
ax[0,0].set_ylabel(r'$P_\times^{\rm cont-fitted} / P_\times^{\rm true-cont} $')
ax[1,0].set_xlabel(r'$k_\parallel$ [1/Mpc]')


# %%
