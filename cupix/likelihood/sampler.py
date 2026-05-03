import numpy as np
import emcee
import os

class Sampler(object):
    """Sampler class, holds posterior"""
    def __init__(
        self,
        free_params,
        config={},
        pool=None,
        log_prob_fn=None
    ):
        """Setup sampler from posterior and config fifle. Inputs:
            - free_params: list of FreeParameter objects
            - config (optional): dictionary with different settings
            - pool: multiprocessing pool for parallelization
            - log_prob_fn: externally defined log_probability function
        """
        self.verbose = config.get('verbose', False)
        self.Np = len(free_params)
        self.setup_emcee_sampler(config, pool=pool, log_prob_fn=log_prob_fn)
        
        self.free_params = free_params
        if self.verbose:
            free_param_names = [par.name for par in self.free_params]
            ini_values = [par.ini_value for par in self.free_params]
            print('Free parameters in sampler')
            print(free_param_names)
            print('Initial values set to', ini_values)


    def setup_emcee_sampler(self, config, pool=None, log_prob_fn =None):

        # read emcee configuration
        nwalkers = config.get('nwalkers', 10)
        self.max_nsteps = config.get('max_nsteps', 1000)
        self.nburnin = config.get('nburnin', 100)
        assert self.nburnin < self.max_nsteps, 'nburnin >= max_nsteps'
        # create emcee sampler object
        self.emcee_sampler =  emcee.EnsembleSampler(
            nwalkers,
            self.Np,
            log_prob_fn,
            pool=pool
        )

    
    def silence(self):
        """set verbose=False in all classes"""
        self.verbose=False
        return
 

    def get_initial_walkers(self):
        """Setup initial states of walkers in sensible points """

        ndim = len(self.free_params)
        nwalkers = self.emcee_sampler.nwalkers

        if self.verbose:
            print("set %d walkers with %d dimensions" % (nwalkers, ndim))

        # Random values between [0, 1) --> [-0.5, 0.5)
        shifts = -0.5 + np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        ini_walkers = np.empty_like(shifts)
        for ip, par in enumerate(self.free_params):
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


    def run_sampler(self):
        """Set up initial points, run burn in, run chains"""

        if self.verbose:
            print('starting run_sampler')

        # set starting point
        p0 = self.get_initial_walkers()
        if self.verbose:
            print('starting points of walkers')
            print(p0)

        # total number of steps
        ntotal = self.nburnin + self.max_nsteps
        for sample in self.emcee_sampler.sample(p0, iterations=ntotal):
            if self.verbose:
                it = self.emcee_sampler.iteration
                if it%10 == 0:
                    print("Step %d out of %d " % (it, ntotal))

        if self.verbose:
            print('finished running sampler')
        return

