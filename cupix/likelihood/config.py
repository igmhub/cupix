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

        self.cosmo_params = self.data.get('cosmo_params', {})
        self.igm_params = self.data.get('igm_params', {})
        self.lya_params = self.data.get('lya_params', {})
        self.contaminant_params = self.data.get('contaminant_params', {})
        self.default_lya_model = self.data.get('default_lya_model', '')
        if self.default_lya_model is None:
            self.default_lya_model = ''
        self.regulate_params()
        self.check_params()
        self.create_single_dictionary()

    def check_params(self):
        " Check that the parameter names are all allowed"
        assert self.default_lya_model in ['', 'best_fit_arinyo_from_p1d', 'best_fit_arinyo_from_colore', 'best_fit_igm_from_p1d', 'gadget_igm_central', 'gadget_arinyo_central'], f"default_lya_model {self.default_lya_model} not recognized. The options are None, 'best_fit_arinyo_from_p1d', 'best_fit_arinyo_from_colore', 'best_fit_igm_from_p1d', 'gadget_igm_central', 'gadget_arinyo_central'"
        
        if self.default_lya_model == '':
            assert (len(self.lya_params) > 0) or (len(self.igm_params) > 0), "If default_lya_model is empty, you must provide some lya_params or igm_params"
        
        if len(self.igm_params) > 0 and len(self.lya_params) > 0:
            raise ValueError("You cannot provide both igm_params and lya_params, choose one or the other")
        
        if 'igm' in self.default_lya_model and len(self.lya_params)>0:
            raise ValueError("You cannot provide lya_params if default_lya_model is an igm model")
        
        if 'arinyo' in self.default_lya_model and len(self.igm_params)>0:
            raise ValueError("You cannot provide igm_params if default_lya_model is an arinyo model")
        self.check_lya_params()
        self.check_igm_params()
        self.check_contaminant_params()


    def check_lya_params(self):
        " Check that the lya_params have the right names"
        allowed_params = ['bias', 'beta', 'q1', 'kv_Mpc', 'av', 'bv', 'kp_Mpc', 'q2']
        for par in self.lya_params:
            assert par in allowed_params, f"lya_param {par} not recognized, allowed parameters are {allowed_params}"
    

    def check_igm_params(self):
        " Check that the igm_params have the right names"
        allowed_params = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
        for par in self.igm_params:
            assert par in allowed_params, f"igm_param {par} not recognized, allowed parameters are {allowed_params}"
    
    # assuming LaCE already has checks for allowed cosmo param names
    
    def check_contaminant_params(self):
        " Check that the contaminant_params have the right names"
        allowed_hcd_params = ['b_H', 'beta_H', 'L_H_Mpc']
        allowed_metal_params = ['b_X', 'beta_X']
        allowed_sky_params = ['b_noise_Mpc']
        allowed_continuum_params = ['kC_Mpc', 'pC']
        for par in self.contaminant_params['hcd_params']:
            assert par in allowed_hcd_params, f"hcd_param {par} not recognized, allowed parameters are {allowed_hcd_params}"
        for par in self.contaminant_params['metal_params']:
            assert par in allowed_metal_params, f"metal_param {par} not recognized, allowed parameters are {allowed_metal_params}"
        for par in self.contaminant_params['sky_params']:
            assert par in allowed_sky_params, f"sky_param {par} not recognized, allowed parameters are {allowed_sky_params}"
        for par in self.contaminant_params['continuum_params']:
            assert par in allowed_continuum_params, f"continuum_param {par} not recognized, allowed parameters are {allowed_continuum_params}"

    def create_single_dictionary(self):
        " Create a single dictionary with all parameters, for easy access"
        self.all_params = {}
        if self.default_lya_model != '':
            self.all_params.update({'default_lya_model': self.default_lya_model})
        self.all_params.update(self.cosmo_params)
        self.all_params.update(self.igm_params)
        self.all_params.update(self.lya_params)
        self.all_params.update(self.contaminant_params['hcd_params'])
        self.all_params.update(self.contaminant_params['metal_params'])
        self.all_params.update(self.contaminant_params['sky_params'])
        self.all_params.update(self.contaminant_params['continuum_params'])
        

    def regulate_params(self):
        # remove any parameters that are None
        self.cosmo_params = {k: v for k, v in self.cosmo_params.items() if v is not None}
        self.igm_params = {k: v for k, v in self.igm_params.items() if v is not None}
        self.lya_params = {k: v for k, v in self.lya_params.items() if v is not None}
        self.contaminant_params['hcd_params'] = {k: v for k, v in self.contaminant_params['hcd_params'].items() if v is not None}
        self.contaminant_params['metal_params'] = {k: v for k, v in self.contaminant_params['metal_params'].items() if v is not None}
        self.contaminant_params['sky_params'] = {k: v for k, v in self.contaminant_params['sky_params'].items() if v is not None}
        self.contaminant_params['continuum_params'] = {k: v for k, v in self.contaminant_params['continuum_params'].items() if v is not None}
