import copy
import numpy as np
from astropy.io import fits
import pandas as pd
# our modules below
import forestflow
from forestflow import priors
from forestflow.P3D_cINN import P3DEmulator
from cupix.utils.utils import get_path_repo


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
        if 'p1d' in self.default_lya_model.lower():
            prior_info = priors.get_IGM_priors(z=self.z, tag='DESI_DR1_P1D')
            igm_params = prior_info['mean']
        elif 'gadget' in self.default_lya_model.lower():
            prior_info = get_priors_gadget(z=self.z, model='igm', verbose=self.verbose)
            igm_params = {par: prior_info[par]['mean'] for par in prior_info}
        else:
            raise ValueError("unknown default_lya_model", self.default_lya_model)
        # update parameters if present in config
        for par in igm_params:
            if par in config:
                igm_params[par] = config[par]
        if self.verbose: print('default values', igm_params)
        return igm_params


    def get_default_lya_params(self, config):
        if self.verbose: print('LyaModel::get_default_lya_params')

        if 'colore' in self.default_lya_model.lower():
            prior_info = get_priors_colore(self.z)
            ff_params = {par: prior_info[par]['mean'] for par in prior_info}
            if 'pressure_only' in self.default_lya_model.lower():
                ff_params['q1'] = 0.0
                ff_params['q2'] = 0.0
                if self.z == 2.2:
                    ff_params['kp'] = 0.325
                elif self.z == 2.4:
                    ff_params['kp'] = 0.315
                else:
                    ff_params['kp'] = 0.300

        elif 'p1d' in self.default_lya_model.lower():
            prior_info = priors.get_arinyo_priors(z=self.z, tag='DESI_DR1_P1D')
            ff_params = prior_info['mean']

        elif 'gadget' in self.default_lya_model.lower():
            # apply the best-fit model from the central gadget sims
            prior_info = get_priors_gadget(z=self.z, model='arinyo', verbose=self.verbose)
            ff_params = {par: prior_info[par]['mean'] for par in prior_info}

        else:
            raise ValueError("unknown default_lya_model", self.default_lya_model)

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

def get_priors_gadget(z, model, verbose=False):
    igm_parnames = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
    ff_parnames = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']
    if 'igm' in model:
        parnames = igm_parnames
    elif 'arinyo' in model:
        parnames = ff_parnames
    else:
        raise ValueError("Error in get_priors_gadget: model should include either 'igm' or 'arinyo'")
    # apply the model from the central gadget sims
    gadget_short_info_file = get_path_repo('cupix') + '/data/emulator/ff_training_info.csv'
    train_test_info = pd.read_csv(gadget_short_info_file)
    # set up a smooth function of z for the mean, min, and max values of desired params
    z_all = train_test_info['z']
    priors_dict = {}
    for par in parnames:
        if par+"_central" in train_test_info.columns:
            par_central_interp = np.interp(z, z_all, train_test_info[par+"_central"])
            par_min_interp = np.interp(z, z_all, train_test_info[par+"_min"])
            par_max_interp = np.interp(z, z_all, train_test_info[par+"_max"])
            if verbose: print(f"for parameter {par}, interpolated central value is {par_central_interp}, min is {par_min_interp}, max is {par_max_interp}")
            priors_dict[par] = {
                "mean": par_central_interp,
                "std": 0.5*(par_max_interp - par_min_interp), # approximation
                "max": par_min_interp,
                "min": par_max_interp
            }
        else:
            print("Parameter", par, "not found in training info file for redshift", z)
    return priors_dict

def get_priors_colore(z):
    assert z in [2.2, 2.4, 2.6, 2.8], "We only have CoLoRe fits for redshifts in [2.2, 2.4, 2.6, 2.8]"
    ff_parnames = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']
    # Load Laura's CF fits for all redshifts
    priors_dict = {}
    
    
    with fits.open(get_path_repo('cupix')+f"/data/colore_xi/bin_{z:.1f}/lyaxlya.fits") as zbin_cf_file:
        for par in ff_parnames:
            if par == "bias":
                val = zbin_cf_file[1].header['bias_LYA']
            elif par == "beta":
                val = zbin_cf_file[1].header['beta_LYA']
            elif par == "q1":
                val = zbin_cf_file[1].header['dnl_arinyo_q1']
            elif par == "kvav":
                val = zbin_cf_file[1].header['dnl_arinyo_kv']**zbin_cf_file[1].header['dnl_arinyo_av']
            elif par == "av":
                val = zbin_cf_file[1].header['dnl_arinyo_av']
            elif par == "bv":
                val = zbin_cf_file[1].header['dnl_arinyo_bv']
            elif par == "kp":
                val = zbin_cf_file[1].header['dnl_arinyo_kp']
            elif par == "q2":
                if 'dnl_arinyo_q2' in zbin_cf_file[1].header:
                    val = zbin_cf_file[1].header['dnl_arinyo_q2']
                else:
                    val = 0
            priors_dict[par] = {
                "mean": val,
                "std": 0.5*np.abs(val), # arbitrary
                "max": val + 5 * 0.5*np.abs(val), # arbitrary
                "min": val - 5 * 0.5*np.abs(val), # arbitrary

            }
    return priors_dict
        
    

def no_unrecognized_lya_params(config):
    " Check that the config does not contain any unrecognized parameters "
    allowed_lya_params = ['bias', 'beta', 'q1', 'kv_Mpc', 'av', 'bv', 'kp_Mpc', 'q2']
    for par in config:
        assert par in allowed_lya_params, f"lya_param {par} not recognized, allowed parameters are {allowed_lya_params}"
    
def no_unrecognized_igm_params(config):
    " Check that the config does not contain any unrecognized parameters "
    allowed_igm_params = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
    if config is not None:
        for par in config:
            assert par in allowed_igm_params, f"igm_param {par} not recognized, allowed parameters are {allowed_igm_params}"

def no_conflicting_params(config):
    """ Accepts a dictionary of theory config parameters,
    checks that the parameter names are all allowed and that the
    user is not trying to pass parameters for multiple model types at once """
    allowed_lya_params = ['bias', 'beta', 'q1', 'kv_Mpc', 'av', 'bv', 'kp_Mpc', 'q2']
    allowed_igm_params = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']

    if 'default_lya_model' in config:
        assert config['default_lya_model'] in ['best_fit_arinyo_from_p1d', 'best_fit_arinyo_from_colore', 'best_fit_igm_from_p1d', 'gadget_igm_central', 'gadget_arinyo_central'], f"default_lya_model {config['default_lya_model']} not recognized. The options are None, 'best_fit_arinyo_from_p1d', 'best_fit_arinyo_from_colore', 'best_fit_igm_from_p1d', 'gadget_igm_central', 'gadget_arinyo_central'"
        # make sure the default lya model does not conflict with input parameters
        if 'igm' in config['default_lya_model']:
            for par in allowed_lya_params:
                assert par not in config, f"you cannot provide lya parameter {par} if default_lya_model is an igm model"
        elif 'arinyo' in config['default_lya_model']:
            for par in allowed_igm_params:
                assert par not in config, f"you cannot provide igm parameter {par} if default_lya_model is an arinyo model"
    # make sure the input parameters do not conflict with each other
    for par in allowed_lya_params:
        if par in config:
            for igm_par in allowed_igm_params:
                assert igm_par not in config, f"you cannot provide both lya parameter {par} and igm parameter {igm_par}"