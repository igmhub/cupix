import copy
import numpy as np
import os
from astropy.io import fits
# our modules below
from lace.cosmo.thermal_broadening import thermal_broadening_kms
import forestflow
from forestflow import priors
from forestflow.P3D_cINN import P3DEmulator
import cupix



def lya_params_from_forestflow_params(ff_params):
    lya_params = ff_params.copy()
    lya_params['kp_Mpc'] = lya_params.pop('kp')
    kvav = lya_params.pop('kvav')
    lya_params['kv_Mpc'] = np.exp( np.log(kvav) / lya_params['av'] )
    return lya_params



class LyaModel(object):
    """Help the theory with pure / clean Lya P3D. 
       It can work with IGM parameters, or directly with Lya parameters.
    """

    def __init__(self, z, config={'verbose':False}):
        """Create object from dictionary"""
        self.z = z
        self.lr_lya = 1215.67 # could be read from elsewhere
        self._setup_from_config(config)
        return 


    def _setup_from_config(self, config):
        """Setup from a dictionary"""

        self.verbose = config.get('verbose', False)
        if self.verbose: print('LyaModel::setup_from_config')

        # setup default values for parameters
        self.default_lya_model = config.get('default_lya_model', 'best_fit_arinyo_from_p1d')
        if 'igm' in self.default_lya_model:
            # default values of mF, T0, gamma and kF_kms
            self.default_igm_params = self.get_default_igm_params(config)
            self.default_lya_params = None
            # setup emulator
            emulator_label = config.get('emulator_label', 'forest_mpg')
            Nrealizations = config.get('Nrealizations', 3000)
            self.emulator = self.get_emulator(emulator_label, Nrealizations)
        else:
            # default values of Lya params (bias, beta, arinyo)
            self.default_lya_params = self.get_default_lya_params(config)
            self.default_igm_params = None
            self.emulator = None

        return


    def get_default_igm_params(self, config):
        # here we get the default values based on default_lya_model string and z
        if self.verbose: print('LyaModel::get_default_igm_params')
        prior_info = priors.get_IGM_priors(z=self.z, tag='DESI_DR1_P1D')
        igm_params = prior_info['mean']
        # update parameters if present in config
        for par in igm_params:
            if par in config:
                igm_params[par] = config[par]
        if self.verbose: print('default values', igm_params)
        return igm_params


    def get_default_lya_params(self, config):
        if self.verbose: print('LyaModel::get_default_lya_params')

        if 'colore' in self.default_lya_model.lower():
            assert self.z in [2.2, 2.4, 2.6, 2.8], "For tag 'best_fit_arinyo_from_colore', redshifts must be in [2.2, 2.4, 2.6, 2.8]"
            # Load Laura's CF fits for all redshifts
            cupixpath = os.path.dirname(cupix.__path__[0])
            ff_params = {}
            with fits.open(cupixpath+f"/data/colore_xi/bin_{self.z:.1f}/lyaxlya.fits") as zbin_cf_file:
                zbin_cf_fit = zbin_cf_file[1].header
                ff_params['bias'] = zbin_cf_fit['bias_LYA']
                ff_params['beta'] = zbin_cf_fit['beta_LYA']
                ff_params['q1'] = zbin_cf_fit['dnl_arinyo_q1']
                ff_params['kvav'] = zbin_cf_fit['dnl_arinyo_kv']**zbin_cf_fit['dnl_arinyo_av']
                ff_params['av'] = zbin_cf_fit['dnl_arinyo_av']
                ff_params['bv'] = zbin_cf_fit['dnl_arinyo_bv']
                ff_params['kp'] = zbin_cf_fit['dnl_arinyo_kp']
                if 'dnl_arinyo_q2' in zbin_cf_fit:
                    ff_params['q2'] = zbin_cf_fit['dnl_arinyo_q2']
                else:
                    ff_params['q2'] = 0
        elif 'p1d' in self.default_lya_model.lower():
            prior_info = priors.get_arinyo_priors(z=self.z, tag='DESI_DR1_P1D')
            ff_params = prior_info['mean']
        else:
            print("unknown default_lya_model", self.default_lya_model)
            #raise ValueError("unknown default_lya_model", self.default_lya_model)

            # ANDREU: we should clarify the if / elif / else above
            prior_info = priors.get_arinyo_priors(z=self.z, tag='DESI_DR1_P1D')
            ff_params = prior_info['mean']

        if self.verbose: print('initial values', ff_params)
        lya_params = lya_params_from_forestflow_params(ff_params)

        # update parameters if present in config
        for par in lya_params:
            if par in config:
                lya_params[par] = config[par]

        if self.verbose: print('final values', lya_params)

        return lya_params



    def get_emulator(self, emulator_label, Nrealizations):
        """Setup the ForestFlow emulator"""

        if emulator_label == "forest_mpg":
            path_program = forestflow.__path__[0][:-10]
            emulator = P3DEmulator(
                model_path=path_program+"/data/emulator_models/forest_mpg", #new_emu
                Nrealizations=Nrealizations
            )
        else:
            raise ValueError("implement emulator_label", emulator_label)

        return emulator


    def get_lya_params(self, cosmo, params):
        """Get the complete list of lya parameters (bias, beta, arinyo)
        from defaults and input params, potentially using the emulator"""

        # check whether you are working with IGM parameters (and emulator)
        if self.default_igm_params is not None:
            assert self.emulator is not None, "need emulator for IGM params"
            # updated IGM params
            igm_params = copy.deepcopy(self.default_igm_params)
            for key in igm_params:
                if key in params:
                    igm_params[key] = params[key]
            # get Lya params from IGM params with emulator
            lya_params = self.emulate_lya_params(cosmo, igm_params)
            # here we could look for Lya params also in input params... not sure

        else:
            lya_params = copy.deepcopy(self.default_lya_params)
            # update Lya params
            for key in lya_params:
                if key in params:
                    lya_params[key] = params[key]

        return lya_params
    
    def emulate_lya_params(self, cosmo, igm_params):
        """Use emulator to translate IGM params and cosmo to Lya params"""

        # emu params include igm and cosmo params
        emu_params = {}
        emu_params['mF'] = igm_params['mF']
        emu_params['gamma'] = igm_params['gamma']
        emu_params['sigT_Mpc'] = igm_params['sigT_Mpc']
        emu_params['kF_Mpc'] = igm_params['kF_Mpc']
        ### MARTINE: these lines are no longer needed if our defualts are sigT_Mpc already. commenting them in case we want in the future again. ###
        # these igm params are not in the correct units for the emulator
        # dkms_dMpc = cosmo.get_dkms_dMpc(self.z)
        # sigT_kms = thermal_broadening_kms(igm_params['T0'])
        # emu_params['sigT_Mpc'] = sigT_kms / dkms_dMpc
        # emu_params['kF_Mpc'] = igm_params['kF_kms'] * dkms_dMpc
        #####

        # amplitude and slope of linear power at kp = 0.7 1/Mpc
        #kp_Mpc = self.emulator.kp_Mpc
        kp_Mpc = 0.7
        linP_params = cosmo.get_linP_Mpc_params(z=self.z, kp_Mpc=kp_Mpc)
        emu_params['Delta2_p'] = linP_params['Delta2_p']
        emu_params['n_p'] = linP_params['n_p']
        # use emulator to estimate Lya params
        ff_params = self.emulator.predict_Arinyos(emu_params=emu_params)
        if self.verbose:
            print('igm params', igm_params)
            print('emu params', emu_params)
            print('ff params', ff_params)

        # we use slightly different names in cupix
        lya_params = lya_params_from_forestflow_params(ff_params)
        if self.verbose:
            print('lya params', lya_params)

        return lya_params


