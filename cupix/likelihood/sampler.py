import numpy as np

class Sampler(object):
    """Sampler class, holds posterior"""
    def __init__(
        self,
        post,
        config={}
    ):
        """Setup posterior from likelihood and free parameters. Inputs:
            - post (required): posterior class 
            - config (optional): dictionary with different settings
        """
        self.verbose = config.get('verbose', False)
        self.post = post
        self.nwalkers = config.get('nwalkers', 100)
        self.nsteps = config.get('nsteps', 10000)
        self.nburnin = config.get('nburnin', 1000)
        assert self.nburnin < self.nsteps, 'nburnin should be smaller than nsteps'
        self.parallel = config.get('parallel', False)
        self.func_to_minimize = self.get_log_posterior_from_cube_values

        # extract information from free_params
        free_params = self.post.free_params
        free_param_names = []
        ini_values = []
        self.Np = len(free_params)
        for ip in range(self.Np):
            free_param_names.append(free_params[ip].name)
            ini_values.append(free_params[ip].ini_value)

        if self.verbose:
            print('Free parameters in sampler')
            print(free_param_names)
            print('Initial values set to', ini_values)

    def silence(self):
        """set verbose=False in all classes"""
        self.verbose=False
        self.post.verbose=False
        self.post.like.verbose=False
        self.post.like.theory.verbose=False
        return
    
    def get_log_posterior_from_cube_values(self, values):
        """Emcee will work better if the parameters are normalized to [0,1],
        so we need to convert back to absolute values before computing log posterior."""
        # convert values array to dictionary of parameters
        true_values = []
        for ip in range(self.Np):
            true_values.append(self.post.free_params[ip].get_true_value_from_cube(values[ip]))
        # compute log posterior using values array
        log_posterior = self.post.get_log_posterior_from_values(true_values)
        return log_posterior

    def get_ini_values_in_cube(self):
        ini_values_in_cube = []
        for param in self.post.free_params:
            ini_value_in_cube = param.get_value_in_cube(param.ini_value)
            ini_values_in_cube.append(ini_value_in_cube)    
        return ini_values_in_cube
    
    def get_initial_walkers(self, pini=None, rms=0.01):
        """Setup initial states of walkers in sensible points
        -- initial will set a range within unit volume around the
           fiducial values to initialise walkers (if no prior is used)"""

        ndim = self.Np
        nwalkers = self.nwalkers

        print("set %d walkers with %d dimensions" % (nwalkers, ndim))

        p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        for ii in range(ndim):
            if pini is None:
                p0[:, ii] = 0.5 + p0[:, ii] * rms
            else:
                p0[:, ii] = pini[ii] + p0[:, ii] * rms
        _ = p0 >= 1.0
        p0[_] = 0.95
        _ = p0 <= 0.0
        p0[_] = 0.05

        return p0
    
    def run_sampler(self, sampler_type='emcee'):
        """Set up sampler, run burn in, run chains,
        return chains"""
        if sampler_type=='emcee':
            import emcee
            ensemble_sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.Np,
                self.func_to_minimize
                )
        else:
            raise NotImplementedError('Sampler type not implemented yet')
        p0 = self.get_initial_walkers(pini=self.get_ini_values_in_cube())
        for sample in ensemble_sampler.sample(
                p0, iterations=self.nburnin + self.nsteps
            ):
                if ensemble_sampler.iteration % 100 == 0:
                    print(
                        "Step %d out of %d "
                        % (ensemble_sampler.iteration, self.nsteps + self.nburnin)
                    )
        return ensemble_sampler
