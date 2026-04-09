import copy
import numpy as np



class ContaminantsModel(object):
    """Help the theory with contaminants (HCDs, metals, sky, continuum...)"""

    def __init__(self, z, config={'verbose':False}):
        """Create object from a dictionary"""
        self.z_lya = z
        self.lr_metal = 1206.52 # SiIII for now
        self.setup_from_config(config)
        return 


    def setup_from_config(self, config):
        """Setup from a dictionary"""
        self.verbose = config.get('verbose', False)
        if self.verbose: print('ContaminantsModel::setup_from_config')

        # set meaningful default values for all parameters
        self.default_hcd_params = self.get_default_hcd_params(config)
        self.default_metal_params = self.get_default_metal_params(config)
        self.default_sky_params = self.get_default_sky_params(config)
        self.default_continuum_params = self.get_default_continuum_params(config)

        # read the function xi_noise(theta) needed for the sky contamination
        self.xi_noise = self.read_xi_noise(config)

        return


    def get_default_hcd_params(self, config):
        # here we should get the default values based on the config and z
        # L_H in Mpc, not Mpc/h
        hcd_params = {'b_H': -0.02, 'beta_H': 0.5, 'L_H_Mpc': 5.0/0.67}

        # update parameters if present in config
        for par in hcd_params:
            if par in config:
                hcd_params[par] = config[par]

        return hcd_params


    def get_default_metal_params(self, config):
        # here we should get the default values based on the config and z
        metal_params = {'b_X': -0.005, 'beta_X': 0.5}

        # update parameters if present in config
        for par in metal_params:
            if par in config:
                metal_params[par] = config[par]

        return metal_params


    def get_default_sky_params(self, config):
        # here we should get the default values based on the config and z
        a_noise = 4e-4
        # b_noise here in Mpc, not Mpc/h
        Delta_rp = 4 / 0.67
        sky_params = {'b_noise_Mpc': a_noise * Delta_rp}

        # update parameters if present in config
        for par in sky_params:
            if par in config:
                sky_params[par] = config[par]

        return sky_params


    def get_default_continuum_params(self, config):
        # here we should get the default values based on the config and z
        continuum_params = {'kC_Mpc': 0.02 * 0.67, 'pC': 1}

        # update parameters if present in config
        for par in continuum_params:
            if par in config:
                continuum_params[par] = config[par]

        return continuum_params


    def get_hcd_params(self, params):
        """Get the complete list of HCD parameters from defaults and input"""

        hcd_params = copy.deepcopy(self.default_hcd_params)
        for key in hcd_params:
            if key in params:
                hcd_params[key] = params[key]

        return hcd_params


    def get_metal_params(self, params):
        """Get the complete list of metal parameters from defaults and input"""

        metal_params = copy.deepcopy(self.default_metal_params)
        for key in metal_params:
            if key in params:
                metal_params[key] = params[key]

        return metal_params


    def get_sky_params(self, params):
        """Get the complete list of sky parameters from defaults and input"""

        sky_params = copy.deepcopy(self.default_sky_params)
        for key in sky_params:
            if key in params:
                sky_params[key] = params[key]

        return sky_params


    def get_continuum_params(self, params):
        """Get the complete list of continuum parameters from defaults and input"""

        continuum_params = copy.deepcopy(self.default_continuum_params)
        for key in continuum_params:
            if key in params:
                continuum_params[key] = params[key]

        return continuum_params


    def read_xi_noise(self, config):
        """Read function xi_noise(theta) for sky contamination"""

        from scipy.interpolate import interp1d
        from astropy.table import Table
        import cupix

        cupixpath = cupix.__path__[0].rsplit('/', 1)[0]
        fname=cupixpath+"/data/desi_instrument/desi-instrument-syst-for-forest-auto-correlation_arcmin.csv"
        syst_table = Table.read(fname)
        syst_interp = interp1d(syst_table["theta_arc"], syst_table["xi_noise"], kind='linear')

        return syst_interp


    def get_xi_noise(self, theta_arc):
        """Evaluate xi_noise at theta_arc"""

        return self.xi_noise(theta_arc)
