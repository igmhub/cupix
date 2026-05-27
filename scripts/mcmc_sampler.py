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

from cupix.utils.utils import get_path_repo
from cupix.parameter_inference.sampling_funcs import prepare_free_parameters
from cupix.likelihood.config import Config
from cupix.parameter_inference.inference_config import InferenceConfig


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

    
    setup_config = Config(cupixpath+"/example_configs/fcast_best_fit_arinyo_from_p1d_theory_config.yaml")
    inf_config = InferenceConfig(cupixpath+"/example_configs/inference_config_ex.yaml") 
    data = DESI_DR2(setup_config.data_config)
    iz = setup_config.theory_config['iz']
    z = data.z[iz]
    setup_config.print_all()
    inf_config.print_all()

    
    # update config class with a use_truth option that could replace the cosmo and theory params with forecast values if it is a forecast and use_truth is True
    # is_forecast = config.data_config['is_forecast']
    # use_truth = config.post_config.get('use_truth', False)
    # if is_forecast and use_truth:
    # config.update_with_truth()

    cosmo = cosmology.Cosmology(cosmo_params_dict=setup_config.cosmo_config)
    
    theory = Theory(z=z, fid_cosmo=cosmo, config=setup_config.theory_config)
    like = Likelihood(data=data, theory=theory, iz=iz, 
                  config=setup_config.like_config)
    
    
    free_params = prepare_free_parameters(['bias','beta','q1','bv','av'], theory, setup_config.theory_config, params_config=inf_config.params_config)
    
    for par in free_params:
        print("Free parameters are: (name, ini_value, true_value, gauss_prior_mean, gauss_prior_width)", par.name, par.ini_value, par.true_value, par.gauss_prior_mean, par.gauss_prior_width)
    
    post = Posterior(like, free_params, config=inf_config.post_config)
    print("Trying to run sampler")

    nthreads_available = len(os.sched_getaffinity(0))
    ncores_use = nthreads_available-1 # leave 1 to run the figure plotting etc in the end of the main function
    print("Starting pool with %d logical cpus available" % ncores_use)

    Np = len(free_params)
    nwalkers = 2*nthreads_available
    max_nsteps = 100 + 10 * Np**3 # want this to be significantly longer than F*tau
    nburnin = 50 + 3 * Np**3
    init_start = time.time()
    
    with mp.Pool(processes=ncores_use, initializer=init_worker, initargs=(post,)) as pool:
        init_end = time.time()
        print("Time to initialize pool and run CAMB in each worker: %.2f seconds" % (init_end - init_start))
        # read emcee configuration to update if needed
        nwalkers = inf_config.sampler_config.get('nwalkers', nwalkers)
        max_nsteps = inf_config.sampler_config.get('max_nsteps', max_nsteps)
        nburnin = inf_config.sampler_config.get('nburnin', nburnin)
        verbose = inf_config.sampler_config.get('verbose', False)
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
        F = inf_config.sampler_config.get('F_tau', 30) # the number of autocorrelation times required to consider the chain converged
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
        plt.savefig(f'/pscratch/sd/m/mlokken/desi-lya/px/plots/mcmc_Np{Np}_{data.theta_min_A_arcmin[0]:.2f}_ncores{ncores_use}_{rand_num}_tau.png'.format())
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
        plt.savefig(f'/pscratch/sd/m/mlokken/desi-lya/px/plots/mcmc_Np{Np}_{data.theta_min_A_arcmin[0]:.2f}_ncores{ncores_use}_{rand_num}_fullchain.png'.format())
        plt.clf()

        chain = emcee_sampler.get_chain(discard=nburnin, thin=2, flat=True)
        gdnames = [par.name for par in free_params]
        gdlabels = [par.latex_label for par in free_params]
        for i in range(Np):
            print("mean", gdnames[i], np.mean(chain[:, i]))
            print("true", gdnames[i], free_params[i].true_value)

        gdsamples = MCSamples(samples=chain, names=gdnames, labels=gdlabels)
        plot_fname = f'/pscratch/sd/m/mlokken/desi-lya/px/plots/mcmc_Np{Np}_{data.theta_min_A_arcmin[0]:.2f}_ncores{ncores_use}_{rand_num}.png'.format()
        print("Saving to", plot_fname)
        g = plots.get_subplot_plotter()
        g.triangle_plot([gdsamples], filled=True)
        g.fig.suptitle(r"DR2 forecast ($\theta > {:.2f}^\prime)$".format(data.theta_min_A_arcmin[0]))
        g.finish_plot()
        plt.savefig(plot_fname)
        # save the chain
        chain_fname = f'/pscratch/sd/m/mlokken/desi-lya/px/chains/mcmc_chain_Np{Np}_{data.theta_min_A_arcmin[0]:.2f}_ncores{ncores_use}_{rand_num}.hdf5'.format()
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


