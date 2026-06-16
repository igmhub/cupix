import sys
import time
from cupix.inference.posterior import Posterior
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from datetime import datetime
import emcee
import shutil


from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood

from cupix.utils.utils import get_path_repo
from cupix.inference.sampling_funcs import prepare_free_parameters, get_initial_walkers
from cupix.likelihood.config import Config
from cupix.inference.inference_config import InferenceConfig
from cupix.inference.sampling_funcs import plot_tau_estimates, plot_chains, plot_contours, record_mcmc_settings, create_output_directory, save_chain

_POST = None

def init_worker(post):
    global _POST
    _POST = post
    _POST.like.get_chi2() # will run camb and store the results in post.like.theory, which will be shared across workers
    

def log_prob_wrapper(values):
    return _POST.get_log_posterior_from_values(values)



def main():
    # set the directory name
    # YYYYMMDD_HHMM_shorttag/
    if len(sys.argv) < 4:
        sys.exit("Usage: python mcmc_sampler.py [short tag name] [setup_config.yaml] [inference_config.yaml].")
    # example setup config is in: cupixpath + "/example_configs/fcast_best_fit_arinyo_from_p1d_theory_config.yaml"
    # example inference config is in: cupixpath + "/example_configs/inference_config_ex.yaml"
    yyyymmdd = datetime.now().strftime("%Y%m%d")
    shorttag = sys.argv[1]
    runname = f'{yyyymmdd}_{shorttag}'
    
    setup_config_path = sys.argv[2]
    inf_config_path = sys.argv[3]
    setup_config = Config(setup_config_path)
    inf_config = InferenceConfig(inf_config_path)
    
    setup_config.print_all()
    inf_config.print_all()

    outdir = create_output_directory(inf_config, runname)
    print("Outputs will be saved to", outdir)
    # copy the run config into the outdir
    shutil.copy(setup_config_path, os.path.join(outdir, 'setup_config_mcmc.yaml'))
    shutil.copy(inf_config_path, os.path.join(outdir, 'inference_config_mcmc.yaml'))

    data = DESI_DR2(setup_config.data_config)
    iz = setup_config.theory_config['iz']
    z = data.z[iz]
    
    # update config class with a use_truth option that could replace the cosmo and theory params with forecast values if it is a forecast and use_truth is True
    # is_forecast = config.data_config['is_forecast']
    # use_truth = config.post_config.get('use_truth', False)
    # if is_forecast and use_truth:
    # config.update_with_truth()

    cosmo = cosmology.Cosmology(cosmo_params_dict=setup_config.cosmo_config)
    
    theory = Theory(z=z, fid_cosmo=cosmo, config=setup_config.theory_config)
    like = Likelihood(data=data, theory=theory, iz=iz, 
                  config=setup_config.like_config)
    
    free_param_names = list(inf_config.params_config.keys())
    free_params = prepare_free_parameters(free_param_names, theory, setup_config.theory_config, params_config=inf_config.params_config)
    
    for par in free_params:
        print("Free parameters are: (name, ini_value, true_value, gauss_prior_mean, gauss_prior_width)", par.name, par.ini_value, par.true_value, par.gauss_prior_mean, par.gauss_prior_width)
    
    post = Posterior(like, free_params, config=inf_config.post_config)

    # nthreads_available = len(os.sched_getaffinity(0))
    print(os.environ.get("SLURM_CPUS_PER_TASK"))
    nthreads_available = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print("Starting pool with %d logical cpus available" % nthreads_available)
    if nthreads_available > 64:
        print("Warning: using more than 64 cores might cause very long setup time due to initialization of CAMB many times. Consider reducing the request, unless very long chains are expected.")
    Np = len(free_params)
    nwalkers = 2*nthreads_available
    if nwalkers < 2 * Np:
        print("Warning: number of walkers should be at least 2 times the number of parameters. Setting nwalkers to %d. Code may be less efficient." % (2*Np))
        nwalkers = 2*Np
    max_nsteps = 100 + 10 * Np**3 # want this to be significantly longer than F*tau
    nburnin = 50 + 3 * Np**3
    # read emcee configuration to update if needed
    nwalkers = inf_config.samp_config.get('nwalkers', nwalkers)
    print("Will use %d walkers" % nwalkers)
    max_nsteps = inf_config.samp_config.get('max_nsteps', max_nsteps)
    nburnin = inf_config.samp_config.get('nburnin', nburnin)
    verbose = inf_config.samp_config.get('verbose', False)
    tau_stability = inf_config.samp_config.get('tau_stability',.05) # the fractional standard deviation of tau estimates required to consider them stable
    assert nburnin < max_nsteps, 'nburnin >= max_nsteps'
    # record to file all the settings that might have been changed due to ncores available
    record_mcmc_settings(outdir, nwalkers, max_nsteps, nburnin)

    init_start = time.time()
    with mp.Pool(processes=nthreads_available, initializer=init_worker, initargs=(post,)) as pool:
        init_end = time.time()
        print("Time to initialize pool and run CAMB in each worker: %.2f seconds" % (init_end - init_start))    
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
        F = inf_config.samp_config.get('F_tau', 30) # the number of autocorrelation times required to consider the chain converged
        sampling_start = time.time()
        for sample in emcee_sampler.sample(p0, iterations=ntotal):
            it = emcee_sampler.iteration
            if it%10 == 0:
                try:
                    # estimate the autocorrelation time and check convergence. This might be inaccurate for short chains
                    tau = emcee_sampler.get_autocorr_time(tol=0, quiet=True) # tol = 0 means not requiring a certain number of autocorr times to trust the estimate.
                except Exception:
                    continue
                mean_tau = np.mean(tau)
                tau_estimates.append(mean_tau)
                if verbose:
                    print("Step %d out of %d " % (it, ntotal))
                    print("mean tau", mean_tau)
                
                # check if tau estimates are stable
                if it>100: # require at least 100 steps to have some estimate of tau
                    if (np.std( tau_estimates[-10:] ) / mean_tau) < tau_stability:
                        print("Tau estimates are stable")
                        # if we have over F * tau samples, end the chain
                        if it > (mean_tau * F):
                            print("Chain has converged after %d steps" % it)
                            break
        sampling_end = time.time()
        print("Time to run sampler: %.2f seconds" % (sampling_end - sampling_start))
        if emcee_sampler.iteration == ntotal:
            print("Warning: chain did not converge after %d steps" % ntotal)
        if nburnin >= emcee_sampler.iteration:
            # reset nburnin to be shorter than the chain
            print(f"Warning: nburnin longer than chain. Resetting nburnin to Niterations//2 = {emcee_sampler.iteration//2}")
            nburnin = emcee_sampler.iteration // 2

        plot_tau_estimates(tau_estimates, os.path.join(outdir, "tau.png"))
        chain = emcee_sampler.get_chain(discard=nburnin, thin=2, flat=True)
        save_chain(outdir, chain, free_params)
        plot_chains(chain, free_params, 0, save=True, show=False, outdir=outdir)
        plot_contours(chain, free_params, title=runname, save=True, show=False, outdir=outdir)
        


if __name__ == "__main__":
    
    start = time.time()
    mp.set_start_method('spawn')
    # mp.set_start_method('fork')
    main()
    print("Sampler finished")
    end = time.time()
    print("Total runtime: %.2f seconds" % (end - start))


