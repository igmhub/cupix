from getdist import MCSamples, plots
import time
from cupix.likelihood.posterior import Posterior
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import multiprocessing as mp
import os
import psutil
import emcee

from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.free_parameter import FreeParameter
from cupix.likelihood.minimize_posterior import Minimizer
from cupix.likelihood.sampler import Sampler
from cupix.utils.utils import get_path_repo

_POST = None

def init_worker(post):
    global _POST
    _POST = post
    _POST.like.get_chi2() # will run camb and store the results in post.like.theory, which will be shared across workers

def log_prob_wrapper(values):
    return _POST.get_log_posterior_from_values(values)

def get_initial_walkers(free_params, nwalkers):
    """Setup initial states of walkers in sensible points """

    ndim = len(free_params)

    # Random values between [0, 1) --> [-0.5, 0.5)
    shifts = -0.5 + np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    ini_walkers = np.empty_like(shifts)
    for ip, par in enumerate(free_params):
        ini_value = par.gauss_prior_mean
        rms = par.gauss_prior_width
        val = ini_value + shifts[:, ip] * rms
        # check that you don't end up outside the bounds
        min_val = par.min_value
        max_val = par.max_value
        _ = val < min_val
        val[_] = min_val + 0.01 * rms
        _ = val > max_val
        val[_] = max_val - 0.01 * rms

        # store into ndarray 
        ini_walkers[:, ip] = val

    return ini_walkers

def main():
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

    #free_params = [bias]
    free_params = [bias, beta]
    for par in free_params:
        print(par.name, par.ini_value, par.true_value)

    post = Posterior(like, free_params, config={'verbose': False})
    print("Trying to run sampler")

    nthreads = psutil.cpu_count(logical=True)
    ncores = psutil.cpu_count(logical=False)
    nthreads_per_core = nthreads // ncores
    nthreads_available = len(os.sched_getaffinity(0))
    ncores_available = nthreads_available // nthreads_per_core
    # let's only use ncores_available to be safe

    print("Starting pool with %d cores available" % ncores_available)

    Np = len(free_params)
    nwalkers = ncores_available # 4*(Np+2) # 2*(Np+2)
    max_nsteps = 50 + 20 * Np**2
    nburnin = 20 + 10 * Np**2
    config={'verbose':True, 'nwalkers':nwalkers, 'max_nsteps': max_nsteps, 'nburnin':nburnin, 'parallel':True}
    print(config)
    with mp.Pool(processes=ncores_available, initializer=init_worker, initargs=(post,)) as pool:
        
        # read emcee configuration
        nwalkers = config.get('nwalkers', 10)
        max_nsteps = config.get('max_nsteps', 1000)
        nburnin = config.get('nburnin', 100)
        verbose = config.get('verbose', False)
        assert nburnin < max_nsteps, 'nburnin >= max_nsteps'
        if verbose:
            print("setting up emcee sampler")
        # create emcee sampler object
        Np = len(free_params)
        emcee_sampler =  emcee.EnsembleSampler(
            nwalkers,
            Np,
            log_prob_wrapper,
            pool=pool
        )
        if verbose:
            print("emcee sampler set up")
        p0 = get_initial_walkers(free_params, nwalkers)
        
        # total number of steps
        ntotal = nburnin + max_nsteps
        for sample in emcee_sampler.sample(p0, iterations=ntotal):
            if verbose:
                it = emcee_sampler.iteration
                if it%10 == 0:
                    print("Step %d out of %d " % (it, ntotal))


        chain = emcee_sampler.get_chain(discard=10, thin=2, flat=True)
        mean_bias = np.mean(chain)
        print('< bias > =', mean_bias)
        print('true bias =', true_lya_params['bias'])
        gdnames = [par.name for par in free_params]
        gdlabels = [par.latex_label for par in free_params]
        gdsamples = MCSamples(samples=chain, names=gdnames, labels=gdlabels)

        plot_fname = f'/pscratch/sd/m/mlokken/desi-lya/px/plots/mcmc_bias_beta_{forecast.theta_min_A_arcmin[0]:.2f}_ncores{ncores_available}.png'.format()
        g = plots.get_subplot_plotter()
        g.triangle_plot([gdsamples], filled=True)
        g.fig.suptitle(r"DR2 forecast ($\theta > {:.2f}^\prime)$".format(forecast.theta_min_A_arcmin[0]))
        g.finish_plot()
        plt.savefig(plot_fname)

if __name__ == "__main__":
    
    start = time.time()
    mp.set_start_method('spawn')
    # mp.set_start_method('fork')
    main()
    print("Sampler finished")
    end = time.time()
    print("Total runtime: %.2f seconds" % (end - start))

