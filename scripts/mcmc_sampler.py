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
from cupix.utils.utils import get_path_repo

_POST = None

def init_worker(post):
    global _POST
    _POST = post
    _POST.like.get_chi2() # will run camb and store the results in post.like.theory, which will be shared across workers
    init_end = time.time()
    

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
    ini_q1   = 0.9 * true_lya_params['q1']
    ini_bv   = 1.05 * true_lya_params['bv']
    ini_kv   = 0.95 * true_lya_params['kv_Mpc']
    ini_av   = 0.5 * true_lya_params['av']
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
        gauss_prior_width=0.1,   
        latex_label=r'q_1'
    )
    bv = FreeParameter(
        name='bv',
        min_value=1.0,
        max_value=2.0,
        ini_value=ini_bv,
        delta=0.1,
        true_value=true_lya_params['bv'],
        gauss_prior_mean=ini_bv,
        gauss_prior_width=0.1,
        latex_label=r'b_v'
    )
    kv = FreeParameter(
        name='kv_Mpc',
        min_value=0.0,
        max_value=1.0,
        ini_value=ini_kv,
        delta=0.1,
        true_value=true_lya_params['kv_Mpc'],
        gauss_prior_mean=ini_kv,
        gauss_prior_width=0.1,
        latex_label=r'k_v'
    )
    av = FreeParameter(
        name='av',
        min_value=0.0,
        max_value=1.0,
        ini_value=ini_av,
        delta=0.1,
        true_value=true_lya_params['av'],
        gauss_prior_mean=ini_av,
        gauss_prior_width=0.1,
        latex_label=r'a_v'
    )

    #free_params = [bias]
    # free_params = [bias, beta]
    free_params = [bias, beta, q1, bv, kv, av]
    for par in free_params:
        print(par.name, par.ini_value, par.true_value)

    post = Posterior(like, free_params, config={'verbose': False})
    print("Trying to run sampler")

    nthreads = psutil.cpu_count(logical=True)
    ncores = psutil.cpu_count(logical=False)
    nthreads_per_core = nthreads // ncores
    nthreads_available = len(os.sched_getaffinity(0))
    ncores_available = nthreads_available // nthreads_per_core
    # let's only use half of ncores_available to be safe
    ncores_use = max(1, ncores_available // 2)
    print("Starting pool with %d cores available" % ncores_use)

    Np = len(free_params)
    nwalkers = 4*ncores_use # 4*(Np+2) # 2*(Np+2)
    max_nsteps = 100 + 10 * Np**3 # muhc longer than before
    nburnin = 50 + 3 * Np**3
    config={'verbose':True, 'nwalkers':nwalkers, 'max_nsteps': max_nsteps, 'nburnin':nburnin, 'parallel':True}
    print(config)
    init_start = time.time()
    with mp.Pool(processes=ncores_use, initializer=init_worker, initargs=(post,)) as pool:
        init_end = time.time()
        print("Time to initialize pool and run CAMB in each worker: %.2f seconds" % (init_end - init_start))
        # read emcee configuration
        nwalkers = config.get('nwalkers', 10)
        max_nsteps = config.get('max_nsteps', 1000)
        nburnin = config.get('nburnin', 100)
        verbose = config.get('verbose', False)
        assert nburnin < max_nsteps, 'nburnin >= max_nsteps'
        if verbose:
            print("setting up emcee sampler")
        # create emcee sampler object
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
        tau_estimates = []
        F = 30
        sampling_start = time.time()
        for sample in emcee_sampler.sample(p0, iterations=ntotal):
            it = emcee_sampler.iteration
            if verbose:
                if it%10 == 0:
                    print("Step %d out of %d " % (it, ntotal))
            if it%10 == 0:
                # estimate the autocorrelation time and check convergence. This might be inaccurate for short chains
                try:
                    tau = emcee_sampler.get_autocorr_time(tol=0, quiet=True) # tol = 0 means not requiring a certain number of autocorr times to trust the estimate.
                except Exception:
                    continue
                mean_tau = np.mean(tau)
                print("mean tau", mean_tau)
                tau_estimates.append(mean_tau)
                # check if tau estimates are stable
                if it>100: # require at least 100 steps to have some estimate of tau
                    if np.std( tau_estimates[-10:] ) / mean_tau < 0.05:
                        print("Tau estimates are stable")
                        # if we have fewer than F * tau samples, do not trust
                        if it > (mean_tau * F):
                            print("Chain has converged after %d steps" % it)
                            break
        sampling_end = time.time()
        print("Time to run sampler: %.2f seconds" % (sampling_end - sampling_start))
        if emcee_sampler.iteration == ntotal:
            print("Warning: chain did not converge after %d steps" % ntotal)
        plt.plot(np.arange(len(tau_estimates))*100, tau_estimates)
        plt.plot(np.arange(len(tau_estimates))*100, np.arange(len(tau_estimates))*100/50, label=r'$\tau = N_{steps}/20$', ls='--')
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("estimated autocorrelation time")
        rand_num = np.random.choice(1000)
        plt.savefig(f'/pscratch/sd/m/mlokken/desi-lya/px/plots/mcmc_Np{Np}_{forecast.theta_min_A_arcmin[0]:.2f}_ncores{ncores_available}_{rand_num}_tau.png'.format())
        plt.clf()
        if nburnin >= emcee_sampler.iteration:
            # reset nburnin to be shorter than the chain
            print("Warning: nburnin longer than chain")
            nburnin = emcee_sampler.iteration // 2
        # first get the full chain and plot one, to understand burnin
        chain_full = emcee_sampler.get_chain(flat=False)
        plt.plot(chain_full[:, :, 0], alpha=0.5)
        plt.ylabel(free_params[0].name)
        plt.xlabel("step")
        plt.title("Full chain for parameter %s" % free_params[0].name)
        plt.savefig(f'/pscratch/sd/m/mlokken/desi-lya/px/plots/mcmc_Np{Np}_{forecast.theta_min_A_arcmin[0]:.2f}_ncores{ncores_available}_{rand_num}_fullchain.png'.format())
        plt.clf()

        chain = emcee_sampler.get_chain(discard=nburnin, thin=2, flat=True)
        gdnames = [par.name for par in free_params]
        gdlabels = [par.latex_label for par in free_params]
        for i in range(Np):
            print("mean", gdnames[i], np.mean(chain[:, i]))
            print("true", gdnames[i], free_params[i].true_value)

        gdsamples = MCSamples(samples=chain, names=gdnames, labels=gdlabels)
        plot_fname = f'/pscratch/sd/m/mlokken/desi-lya/px/plots/mcmc_Np{Np}_{forecast.theta_min_A_arcmin[0]:.2f}_ncores{ncores_available}_{rand_num}.png'.format()
        print("Saving to", plot_fname)
        g = plots.get_subplot_plotter()
        g.triangle_plot([gdsamples], filled=True)
        g.fig.suptitle(r"DR2 forecast ($\theta > {:.2f}^\prime)$".format(forecast.theta_min_A_arcmin[0]))
        g.finish_plot()
        plt.savefig(plot_fname)
        # save the chain
        chain_fname = f'/pscratch/sd/m/mlokken/desi-lya/px/chains/mcmc_chain_Np{Np}_{forecast.theta_min_A_arcmin[0]:.2f}_ncores{ncores_available}_{rand_num}.hdf5'.format()
        print("Saving chain to", chain_fname)
        with h5.File(chain_fname, 'w') as f:
            f.create_dataset('chain', data=chain)
            f.attrs['gdnames'] = gdnames
            f.attrs['gdlabels'] = gdlabels
            f.attrs['free_params'] = gdnames
            for par in free_params:
                f.attrs[f'{par.name}_true_value'] = par.true_value
                f.attrs[f'{par.name}_ini_value'] = par.ini_value
        


if __name__ == "__main__":
    
    start = time.time()
    mp.set_start_method('spawn')
    # mp.set_start_method('fork')
    main()
    print("Sampler finished")
    end = time.time()
    print("Total runtime: %.2f seconds" % (end - start))


