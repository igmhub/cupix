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
        - free_params (required): list of FreeParameter objects to vary
        - config (optional): dictionary with different settings
        """

        self.verbose = config.get('verbose', False)
        self.like = like

        # this is a list of FreeParameter objects
        self.free_params = free_params

        # this (if provided) is a dictionary
        self.fixed_params = config.get('fixed_params', {})
        if self.fixed_params:
            free_par_names = [
            for key, value in self.fixed_params.items():
                assert key not in self.free_params, key + ' both free and fixed'
                if self.verbose:
                    print('{} parameter fixed to {:.4f}'.format(key, value))


    def silence(self):
        """set verbose=False in all classes"""
        self.verbose=False
        self.like.verbose=False
        self.like.theory.verbose=False
        return


    def get_ndf(self):
        """Number of degrees of freedom"""
        ndata = self.like.get_ndata()
        npar = len(self.free_params)
        return ndata-npar


    def get_log_like_from_values(self, values):

        # get dictionary with free parameters
        params = self.get_params_from_values(values)

        # include also fixed parameters
        params.update(self.fixed_params)

        # ask for likelihood (including also fixed_params)
        log_like = self.like.get_log_like(params=params)

        if self.verbose:
            print('params =', params)
            print('log like =', log_like)

        return log_like


    def get_log_posterior_from_values(self, values):

        # start by computing the prior contribution
        log_prior = self.get_log_prior_from_values(values)

        # compute the likelihood (including fixed params)
        log_like = self.get_log_like_from_values(values)

        # add both
        log_posterior = log_like + log_prior

        if self.verbose:
            print('values =', values)
            print('log_like, prior, post =', log_like, log_prior, log_posterior)

        return log_posterior


    def get_log_prior_from_values(self, values):

        Np = len(self.free_params)
        assert len(values) == Np, "Inconsistent number of free parameters"

        # collect priors from each parameter (no correlation)
        chi2_prior = 0.0

        for ip, par in enumerate(self.free_params):
            # check parameter bounds
            if par.out_of_bounds(values[ip]):
                return -np.inf
            # if within bounds, add prior chi2
            chi2_prior += par.get_prior_chi2(values[ip])

        log_prior = -0.5 * chi2_prior

        return log_prior


    def get_param_index(self, param_name):

        for ip, par in enumerate(self.free_params):
            if par.name == param_name:
                ipar = ip
        return ipar


    def get_params_from_values(self, values):
        """Collect dictionary of parameters using input values"""

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
