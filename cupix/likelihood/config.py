import yaml

# Read yaml config files and create dictionaries
class Config(object):
    " Config class, reads yaml files and creates dictionaries "
    def __init__(
        self,
        yaml_file
    ):
        with open(yaml_file) as stream:
            try:
                self.data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.verbose_all = self.data.get('verbose_all', None)
        self._theory_params = self.data.get('theory_params', {})
        self._like_params = self.data.get('likelihood_params', {})
        self._post_params = self.data.get('posterior_params', {})
        self._mini_params = self.data.get('minimizer_params', {})
        self._samp_params = self.data.get('sampler_params', {})
        
        if self._theory_params['default_lya_model'] is None:
            self._theory_params['default_lya_model'] = ''
        # check that all the parameters are valid entries and create a single dictionary with all parameters
        
        self.check_params()
        self.create_single_dictionary()
        self.regulate_params()

    def check_contaminant_params(self):
        " Check that the contaminant_params have the right names"
        allowed_hcd_params = ['b_H', 'beta_H', 'L_H_Mpc']
        allowed_metal_params = ['b_X', 'beta_X']
        allowed_sky_params = ['b_noise_Mpc']
        allowed_continuum_params = ['kC_Mpc', 'pC']
        for par in self._theory_params['contaminant_params']['hcd_params']:
            assert par in allowed_hcd_params, f"hcd_param {par} not recognized, allowed parameters are {allowed_hcd_params}"
        for par in self._theory_params['contaminant_params']['metal_params']:
            assert par in allowed_metal_params, f"metal_param {par} not recognized, allowed parameters are {allowed_metal_params}"
        for par in self._theory_params['contaminant_params']['sky_params']:
            assert par in allowed_sky_params, f"sky_param {par} not recognized, allowed parameters are {allowed_sky_params}"
        for par in self._theory_params['contaminant_params']['continuum_params']:
            assert par in allowed_continuum_params, f"continuum_param {par} not recognized, allowed parameters are {allowed_continuum_params}"

    def check_igm_params(self):
        " Check that the igm_params have the right names"
        allowed_params = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
        for par in self._theory_params['igm_params']:
            assert par in allowed_params, f"igm_param {par} not recognized, allowed parameters are {allowed_params}"
    
    # assuming LaCE already has checks for allowed cosmo param names

    def check_lya_params(self):
        " Check that the lya_params have the right names"
        allowed_params = ['bias', 'beta', 'q1', 'kv_Mpc', 'av', 'bv', 'kp_Mpc', 'q2']
        for par in self._theory_params['lya_params']:
            assert par in allowed_params, f"lya_param {par} not recognized, allowed parameters are {allowed_params}"
    
    def check_params(self):
        " Check that the parameter names are all allowed"
        assert self._theory_params['default_lya_model'] in ['', 'best_fit_arinyo_from_p1d', 'best_fit_arinyo_from_colore', 'best_fit_igm_from_p1d', 'gadget_igm_central', 'gadget_arinyo_central'], f"default_lya_model {self.default_lya_model} not recognized. The options are None, 'best_fit_arinyo_from_p1d', 'best_fit_arinyo_from_colore', 'best_fit_igm_from_p1d', 'gadget_igm_central', 'gadget_arinyo_central'"
        
        if self._theory_params['default_lya_model'] == '':
            assert (len(self._theory_params['lya_params']) > 0) or (len(self._theory_params['igm_params']) > 0), "If default_lya_model is empty, you must provide some lya_params or igm_params"
        
        if len(self._theory_params['default_lya_model']) > 0 and len(self._theory_params['default_lya_model']) > 0:
            raise ValueError("You cannot provide both igm_params and lya_params, choose one or the other")
        
        if 'igm' in self._theory_params['default_lya_model'] and len(self._theory_params['default_lya_model'])>0:
            raise ValueError("You cannot provide lya_params if default_lya_model is an igm model")
        
        if 'arinyo' in self._theory_params['default_lya_model'] and len(self._theory_params['default_lya_model'])>0:
            raise ValueError("You cannot provide igm_params if default_lya_model is an arinyo model")
        self.check_lya_params()
        self.check_igm_params()
        self.check_contaminant_params()
    
    def create_single_dictionary(self):
        " Create a single dictionary with all parameters, for easy access"
        self.all_params = {'theory_params': self._theory_params, 'like_params': self._like_params, 'post_params': self._post_params, 'mini_params': self._mini_params, 'samp_params': self._samp_params}
        self.verbose_update()

    def regulate_params(self):

        for param_type in self.all_params: # e.g., 'verbose_all', 'theory_params', 'like_params', etc.
            if type(self.all_params[param_type]) == dict:
                self.all_params[param_type] = {k: v for k, v in self.all_params[param_type].items() if v is not None}
                for subparam_type in self.all_params[param_type]: # e.g., 'igm_params', 'lya_params', etc.
                    if type(self.all_params[param_type][subparam_type]) == dict: # e.g., 'igm_params', 'lya_params', etc.
                        self.all_params[param_type][subparam_type] = {k: v for k, v in self.all_params[param_type][subparam_type].items() if v is not None}

    def verbose_update(self):
        if self.verbose_all is not None:
            if self.verbose_all:
                for param_type in self.all_params:
                    self.all_params[param_type]['verbose'] = True
            else:
                for param_type in self.all_params:
                    self.all_params[param_type]['verbose'] = False