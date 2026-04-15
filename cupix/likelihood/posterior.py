import numpy as np


class Posterior(object):
    """Posterior class, holds likelihood and priors, knows about free parameters"""

    def __init__(
        self,
        like,
        free_params,
        config={}
    ):
        """Setup posterior from likelihood and free parameters. Inputs:
        - like (required): Likelihood class 
        - free_params (required): list LikelihoodParameter to vary
        - config (optional): dictionary with different settings
        """

        self.verbose = config.get('verbose', False)
        self.like = like
        self.free_params = free_params
 

    def silence(self):
        """set verbose=False in all classes"""
        self.verbose=False
        self.like.verbose=False
        self.like.theory.verbose=False
        return


    def get_log_posterior(self, params={}):

        # ask likelihood to evalute log-like
        log_like = self.like.get_log_like(params=params)

        # ask for prior
        log_prior = self.get_log_prior(params=params)
        
        # add both
        log_posterior = log_like + log_prior

        if self.verbose:
            print(params, log_like, log_prior, log_posterior)

        return log_posterior


    def get_log_posterior_from_values(self, values):

        # convert values array to dictionary of parameters
        params = self.get_params_from_values(values)

        # compute log posterior using values array
        log_posterior = self.get_log_posterior(params)

        return log_posterior


    def get_log_prior(self, params={}):

        # collect priors from each parameter (no correlation)
        chi2_prior = 0.0

        for name,value in params.items():
            ip = self.get_param_index(param_name=name)
            chi2_prior += self.free_params[ip].get_prior_chi2(value)

        log_prior = -0.5 * chi2_prior

        return log_prior


    def get_param_index(self, param_name):

        for ip, par in enumerate(self.free_params):
            if par.name == param_name:
                ipar = ip
        return ipar


    def get_params_from_values(self, values):

        Np = len(self.free_params)
        assert len(values) == Np, "Inconsistent number of free parameters"

        params = {}
        for ip in range(Np):
            name = self.free_params[ip].name
            params[name] = values[ip]

        return params


    def get_values_from_params(self, params):

        Np = len(self.free_params)
        assert len(params) == Np, "Inconsistent number of free parameters"

        values = np.empty(Np)
        for ip in range(Np):
            name = self.free_params[ip].name
            values[ip] = params[name]

        return values
