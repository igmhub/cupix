import numpy as np
import emcee


class Sampler(object):
    """Sampler class, holds posterior"""
    def __init__(
        self,
        post,
        config={}
    ):
        """Setup sampler from posterior and config file. Inputs:
            - post (required): posterior class 
            - config (optional): dictionary with different settings
        """
        self.verbose = config.get('verbose', False)
        self.post = post
        self.setup_emcee_sampler(config)

        if self.verbose:
            free_param_names = [par.name for par in self.post.free_params]
            ini_values = [par.ini_value for par in self.post.free_params]
            print('Free parameters in sampler')
            print(free_param_names)
            print('Initial values set to', ini_values)


    def setup_emcee_sampler(self, config):

        # read emcee configuration
        nwalkers = config.get('nwalkers', 10)
        self.max_nsteps = config.get('max_nsteps', 1000)
        self.nburnin = config.get('nburnin', 100)
        assert self.nburnin < self.max_nsteps, 'nburnin >= max_nsteps'
        self.parallel = config.get('parallel', False)

        # create emcee sampler object
        Np = len(self.post.free_params)
        self.emcee_sampler =  emcee.EnsembleSampler(
            nwalkers,
            Np,
            self.post.get_log_posterior_from_values
        )


    def silence(self):
        """set verbose=False in all classes"""
        self.verbose=False
        self.post.verbose=False
        self.post.like.verbose=False
        self.post.like.theory.verbose=False
        return
 

    def get_initial_walkers(self):
        """Setup initial states of walkers in sensible points """

        ndim = len(self.post.free_params)
        nwalkers = self.emcee_sampler.nwalkers

        if self.verbose:
            print("set %d walkers with %d dimensions" % (nwalkers, ndim))

        # Normally distributed random values
        shifts = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        ini_walkers = np.empty_like(shifts)
        for ip, par in enumerate(self.post.free_params):
            ini_value = par.ini_value
            rms = par.delta
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
