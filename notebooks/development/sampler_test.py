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
from cupix.likelihood.sampler import Sampler
from cupix.utils.utils import get_path_repo
cupixpath = get_path_repo('cupix')

# %%
forecast_file = f"{cupixpath}/data/px_measurements/forecast/fcast_best_fit_arinyo_from_p1d_real_bf3_binned_out_px-zbins_4-thetabins_10_w_res_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kM_max_cut_AA=0.5, km_max_cut_AA=0.55, theta_min_cut_arcmin=1.0)
iz = 0
z = forecast.z[iz]

# %%
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = forecast.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_m={Nk_m}")
# rebinned values
Nz, Nt_A, Nk_M = forecast.Px_ZAM.shape
print(f"rebinned values: Nz={Nz}, Nt_A={Nt_A}, Nk_M={Nk_M}")
print('theta >', forecast.theta_min_A_arcmin)
print('kpar <', forecast.k_M_edges[0,-1])

# %%
true_cosmo_params = {}
with h5.File(forecast_file) as f:
    for key in f['cosmo_params'].attrs.keys():
        true_cosmo_params[key] = f['cosmo_params'].attrs[key]
print(true_cosmo_params)

# %%
# translate these to our Lya params
with h5.File(forecast_file) as f:

    true_lya_params = {}        
    if 'igm_params' in f['P_Z_AM'][f'z_{iz}'].keys():
        igm_params = f['P_Z_AM'][f'z_{iz}']['igm_params'].attrs
    if 'lya_params' in f['P_Z_AM'][f'z_{iz}'].keys():
        lya_params = f['P_Z_AM'][f'z_{iz}']['lya_params'].attrs
        for par in lya_params:
            true_lya_params[par] = lya_params[par]
    
    elif 'ff_emulated_params' in f['P_Z_AM']['z_0'].keys():
        ff_params = f['P_Z_AM']['z_0']['ff_emulated_params'].attrs
        for par in ff_params:
            true_lya_params[par] = ff_params[par]
    else:
        raise ValueError("No IGM or Lya parameters found in the forecast file.")


# %%
true_lya_params

# %%
# use the true cosmology as fiducial
cosmo = cosmology.Cosmology(cosmo_params_dict=true_cosmo_params)

# %%
# use the true Lya parameters (Arinyo / bias / beta)
config = true_lya_params | {'verbose': False}
# make your life a bit harder by changing a bit the value of beta
wrong_beta = False
if wrong_beta:
    config['beta'] = 1.02*config['beta']
print(config)
theory = Theory(z=z, fid_cosmo=cosmo, config=config)

# %%
like = Likelihood(data=forecast, theory=theory, iz=iz)

# %%
like.get_chi2()

# %%
# start a bit off
ini_bias = 1.05 * true_lya_params['bias']
ini_beta = 0.9 * true_lya_params['beta']

# %% [markdown]
# ### Setup free parameters and posterior

# %%
bias = FreeParameter(
    name='bias',
    min_value=-0.5,
    max_value=-0.01,
    ini_value=ini_bias,
    true_value=true_lya_params['bias'],
    delta=0.01,
    gauss_prior_mean=ini_bias,
    gauss_prior_width=0.05,
    latex_label=r'b_\alpha'
)
beta = FreeParameter(
    name='beta',
    min_value=0.1,
    max_value=5.0,
    ini_value=ini_beta,
    delta=0.1,
    true_value=true_lya_params['beta'],
    gauss_prior_mean=ini_beta,
    gauss_prior_width=0.2,   
    latex_label=r'\beta_\alpha'
)

# %%
#free_params = [bias]
free_params = [bias, beta]
for par in free_params:
    print(par.name, par.ini_value, par.true_value)

# %%
post = Posterior(like, free_params, config={'verbose': False})

# %%
test = post.get_log_posterior()
print(test)

# %%
post.get_log_prior()

# %% [markdown]
# ### Minimize posterior (will use this as starting point of sampler)

# %%
mini = Minimizer(post, config={'verbose':False})

# %%
mini.minimize()

# %%
if len(free_params)==2:
    true_params = {'bias': true_lya_params['bias'], 'beta': true_lya_params['beta']}
    mini.plot_ellipses('bias', 'beta', true_vals=true_params)

# %%
best_fit_params = mini.get_best_fit_params()
print(best_fit_params)

# %% [markdown]
# ### Setup sampler

# %%
Np = len(free_params)
nwalkers = 2*(Np+2)
max_nsteps = 50 + 20 * Np**2
nburnin = 20 + 10 * Np**2
config={'verbose':True, 'nwalkers':nwalkers, 'max_nsteps': max_nsteps, 'nburnin':nburnin}
print(config)

# %%
post.silence()
samp = Sampler(post, config=config)

# %%
ini = samp.get_initial_walkers()
print(ini)

# %%
samp.run_sampler()

# %%
samp.emcee_sampler.acceptance_fraction

# %%
chain = samp.emcee_sampler.get_chain(discard=10, thin=2, flat=True)
print(chain.shape)
mean_bias = np.mean(chain)
print('< bias > =', mean_bias)
print('true bias =', true_lya_params['bias'])

# %%
from getdist import MCSamples, plots

# %%
gdnames = [par.name for par in free_params]
gdlabels = [par.latex_label for par in free_params]
gdsamples = MCSamples(samples=chain, names=gdnames, labels=gdlabels)

# %%
plot_fname = 'bias_beta_{:.2f}.png'.format(forecast.theta_min_A_arcmin[0])
g = plots.get_subplot_plotter()
g.triangle_plot([gdsamples], filled=True)
g.fig.suptitle(r"DR2 forecast ($\theta > {:.2f}^\prime)$".format(forecast.theta_min_A_arcmin[0]))
g.finish_plot()
plt.savefig(plot_fname)

# %%
