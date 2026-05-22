from getdist import MCSamples, plots
import time
from cupix.likelihood.posterior import Posterior
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.free_parameter import FreeParameter
from cupix.utils.utils import get_path_repo
from cupix.sampling.sampler import Sampler

start = time.time()
cupixpath = get_path_repo('cupix')

forecast_file = f"{cupixpath}/data/px_measurements/forecast/fcast_best_fit_arinyo_from_p1d_real_bf3_binned_out_px-zbins_4-thetabins_10_w_res_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kM_max_cut_AA=0.5, km_max_cut_AA=0.55, theta_min_cut_arcmin=1.0)
iz = 0
z = forecast.z[iz]

true_cosmo_params = {}
with h5.File(forecast_file) as f:
    for key in f['cosmo_params'].attrs.keys():
        true_cosmo_params[key] = f['cosmo_params'].attrs[key]
print(true_cosmo_params)

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

# use the true cosmology as fiducial
cosmo = cosmology.Cosmology(cosmo_params_dict=true_cosmo_params)

# use the true Lya parameters (Arinyo / bias / beta)
config = true_lya_params | {'verbose': False}
theory = Theory(z=z, fid_cosmo=cosmo, config=config)
like = Likelihood(data=forecast, theory=theory, iz=iz, config = {'verbose': False})
# start a bit off
ini_bias = 1.05 * true_lya_params['bias']
ini_beta = 0.9 * true_lya_params['beta']
ini_q1   = 0.9 * true_lya_params['q1']
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
q1 = FreeParameter(
    name='q1',
    min_value=0.0,
    max_value=1.0,
    ini_value=ini_q1,
    delta=0.1,
    true_value=true_lya_params['q1'],
    gauss_prior_mean=ini_q1,
    gauss_prior_width=0.2,   
    latex_label=r'q_1'
)

#free_params = [bias]
free_params = [bias, beta, q1]
for par in free_params:
    print(par.name, par.ini_value, par.true_value)

post = Posterior(like, free_params, config={'verbose': False})
print("Trying to run sampler")

Np = len(free_params)
nwalkers = 32 # to match the other run
max_nsteps = 100 + 20 * Np**2
nburnin = 30 + 10 * Np**2
config_samp={'verbose':True, 'nwalkers':nwalkers, 'max_nsteps': max_nsteps, 'nburnin':nburnin}
samp = Sampler(post, config=config_samp)
ini = samp.get_initial_walkers()
print(ini)

samp.post.like.verbose = True
samp.post.like.theory.verbose=True
samp.run_sampler()


chain = samp.emcee_sampler.get_chain(discard=10, thin=2, flat=True)
mean_bias = np.mean(chain)
print('< bias > =', mean_bias)
print('true bias =', true_lya_params['bias'])
gdnames = [par.name for par in free_params]
gdlabels = [par.latex_label for par in free_params]
gdsamples = MCSamples(samples=chain, names=gdnames, labels=gdlabels)
rand_num = np.random.choice(1000)
plot_fname = f'/pscratch/sd/m/mlokken/desi-lya/px/plots/mcmc_bias_beta_{forecast.theta_min_A_arcmin[0]:.2f}_ncores1_{rand_num}.png'.format()
print("Saving to", plot_fname)
g = plots.get_subplot_plotter()
g.triangle_plot([gdsamples], filled=True)
g.fig.suptitle(r"DR2 forecast ($\theta > {:.2f}^\prime)$".format(forecast.theta_min_A_arcmin[0]))
g.finish_plot()
plt.savefig(plot_fname)
print("Sampler finished")
end = time.time()
print("Total runtime: %.2f seconds" % (end - start))