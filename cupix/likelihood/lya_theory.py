import numpy as np
import copy
from lace.cosmo import cosmology
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.lyaP3D import LyaP3D
from cupix.likelihood.likelihood_parameter import likeparam_from_dict, LikelihoodParameter, dict_from_likeparam, format_like_params_dict
import sys
from lace.cosmo.thermal_broadening import thermal_broadening_kms
from forestflow import priors
from astropy.io import fits


class Theory(object):
    """Translator between the likelihood object and the emulator. This object
    will map from a set of CAMB parameters directly to emulator calls, without
    going through our Delta^2_\star parametrisation"""

    def __init__(
        self,
        zs,
        fid_cosmo=None,
        default_lya_theory='best_fit_arinyo_from_p1d',
        p3d_label='arinyo',
        emulator_label=None,
        verbose=False,
        k_unit='iAA' # either iAA or ikms
    ):
        """Setup object to compute predictions for the 1D power spectrum.
        Inputs:
            - zs: redshifts that will be evaluated
            - fid_cosmo: fiducial Cosmology object, used for unit conversions, setting the linear power spectrum, etc
            - default_lya_theory: string tag to specify which default set of likelihood parameters to use. Options are 'best_fit_arinyo_from_p1d' to use the best-fit Arinyo coefficients from the p1d paper, 'best_fit_igm_from_p1d' to use the best-fit IGM parameters from the p1d paper, and 'best_fit_arinyo_from_colore' to use the best-fit Arinyo coefficients from Laura's CF fits.
            - p3d_label: string tag to specify which P3D model to use. Current option is 'arinyo' to use the Arinyo model.
            - emulator_label: string tag to specify which emulator to use for predicting the Arinyo coefficients. Current option is 'forestflow_emu' to use the forestflow emulator.
        """

        self.verbose = verbose
        self.zs = zs
        # specify pivot point used in compressed parameters
        self.k_unit = k_unit
        if fid_cosmo is None:
            self.fid_cosmo = cosmology.Cosmology()
        else:
            self.fid_cosmo = fid_cosmo
        self._load_P3D_model(p3d_label=p3d_label)
        self.default_lya_theory = default_lya_theory
        self.set_default_param_values(tag=default_lya_theory)
        self.arinyo_par_names = ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]
        self.igm_par_names = ["Delta2_p", "n_p", "mF", "gamma", "sigT_Mpc", "kF_Mpc", "lambda_P"]

        if emulator_label == "forestflow_emu":
            self.emulator = FF_emulator(zs)
        elif emulator_label is None:
            assert "igm" not in self.default_lya_theory, "If default_lya_theory includes 'igm', an emulator must be specified to predict the Arinyo parameters. Please specify an emulator_label, for example 'forestflow_emu'."
        else:
            raise ValueError("Emulator label not recognized. Current only option is 'forestflow_emu'.")


    def get_unit_conversions(self, zs, like_params):
        """return conversion from Mpc to km/s or Mpc to inv Angstroms,
            and from Mpc to deg for transverse separations"""
 
        # convert list of LikelihoodParameters to dictionary
        params_dict = dict_from_likeparam(like_params)
        print('params_dict', params_dict)

        if self.fid_cosmo.same_background(cosmo_params=params_dict):
            # use background and transfer functions from initial cosmology
            cosmo = self.fid_cosmo
            if self.verbose:
                print("recycle transfer function")
        else:
            cosmo = cosmology.Cosmology(cosmo_params_dict=params_dict)

        dkms_dMpc_zs  = np.array([cosmo.get_dkms_dMpc(z) for z in zs])
        ddeg_dMpc_zs = np.array([cosmo.get_ddeg_dMpc(z) for z in zs])
        dAA_dMpc_zs   = np.array([cosmo.get_dAA_dMpc(z) for z in zs])

        if self.k_unit == 'ikms':
            return_conv_k_of_zs = dkms_dMpc_zs
        else:
            return_conv_k_of_zs = dAA_dMpc_zs
        return return_conv_k_of_zs, ddeg_dMpc_zs


    def get_emulator_calls(
        self, iz_choice, theory_inputs
    ):
        """Compute models that will be emulated, one per redshift bin.
        - like_params identify likelihood parameters to use."""

        # convert any different units to emulator-accepted units
        for key in theory_inputs.keys():
            if 'T0' in key:
                sigT_kms = thermal_broadening_kms(theory_inputs[key])
                z_int = int(key.split('_')[-1])
                print('WARNING: check this code (Andreu)')
                z=self.zs[z_int]
                # I don't this can assume fiducial cosmo...
                sigT_Mpc = sigT_kms / self.fid_cosmo.get_dkms_dMpc(z=z)
                theory_inputs[f'sigT_Mpc_{z_int}'] = sigT_Mpc
            elif 'kF_kms' in key:
                z_int = int(key.split('_')[-1])
                print('WARNING: check this code (Andreu)')
                z=self.zs[z_int]
                # I don't this can assume fiducial cosmo...
                kF_Mpc = theory_inputs[key] / self.fid_cosmo.get_dkms_dMpc(z=z)
                theory_inputs[f'kF_Mpc_{z_int}'] = kF_Mpc
            # later, when varying cosmology, can add here the conversions for Delta* and n* to Deltap and np

        # store emulator calls
        emu_call = {}
        # reformat redshifts
        for key in self.emulator.emu_params:
            emu_call[key] = np.zeros(len(iz_choice))
            for iiz, iz in enumerate(iz_choice):
                key_iz = key + f"_{iz}"
                if key_iz in theory_inputs.keys():
                    if theory_inputs[key_iz] == np.nan or theory_inputs[key_iz] is None:
                        raise ValueError(
                            f"Parameter {key} not found in theory inputs"
                        )
                    else:
                        emu_call[key][iiz] = theory_inputs[key_iz]
                else:
                    raise ValueError(
                        "Missing information for", key_iz
                    )
        return emu_call


    def set_default_param_values(self, tag='best_fit_arinyo_from_p1d'):

        """Set default set of likelihood parameters.
        Param_format options are 'arinyo' or 'cosmo_igm'.
        Tag options are 'p1d' to input priors from the p1d paper,
        'colore' to get the Laura's best-fit parameters (arinyo only)."""

        self.default_param_dict = {}
        
        for iz, z in enumerate(self.zs):
            # record the index-z relationship
            self.default_param_dict[f"z_{iz}"] = z
            if tag == 'best_fit_arinyo_from_p1d' or tag == 'best_fit_igm_from_p1d':
                if tag == 'best_fit_arinyo_from_p1d':
                    # import priors using forestflow functions
                    prior_info = priors.get_arinyo_priors(z, tag='DESI_DR1_P1D')
                elif tag == 'best_fit_igm_from_p1d':
                    prior_info = priors.get_IGM_priors(z, tag='DESI_DR1_P1D')
                for par in prior_info["mean"]:
                    self.default_param_dict[par+f"_{iz}"] = prior_info["mean"][par]
                        
            elif tag == 'best_fit_arinyo_from_colore':
                assert z in [2.2, 2.4, 2.6, 2.8], "For tag 'best_fit_arinyo_from_colore', redshifts must be in [2.2, 2.4, 2.6, 2.8]"
                # Load Laura's CF fits for all redshifts
                with fits.open(f"/global/cfs/cdirs/desicollab/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-0/jura-124/raw_bao_unblinding/fits/output_fitter-z-bins/bin_{z:.1f}/lyaxlya.fits") as zbin_cf_file:
                    zbin_cf_fit = zbin_cf_file[1].header
                    self.default_param_dict[f'bias_{iz}'] = zbin_cf_fit['bias_LYA']
                    self.default_param_dict[f'beta_{iz}'] = zbin_cf_fit['beta_LYA']
                    self.default_param_dict[f'q1_{iz}'] = zbin_cf_fit['dnl_arinyo_q1']
                    self.default_param_dict[f'kvav_{iz}'] = zbin_cf_fit['dnl_arinyo_kv']**zbin_cf_fit['dnl_arinyo_av']
                    self.default_param_dict[f'av_{iz}'] = zbin_cf_fit['dnl_arinyo_av']
                    self.default_param_dict[f'bv_{iz}'] = zbin_cf_fit['dnl_arinyo_bv']
                    self.default_param_dict[f'kp_{iz}'] = zbin_cf_fit['dnl_arinyo_kp']
                    if 'dnl_arinyo_q2' in zbin_cf_fit:
                        self.default_param_dict[f'q2_{iz}'] = zbin_cf_fit['dnl_arinyo_q2']
            else:
                raise ValueError("Tag not recognized, choose from 'best_fit_arinyo_from_p1d', 'best_fit_igm_from_p1d', or 'best_fit_arinyo_from_colore'")
            

    def get_px_AA(
        self,
        k_AA,
        theta_arcmin,
        zs=None,
        like_params={},
        add_silicon=False,
        verbose=None,
        return_arinyo_coeffs=False
    ):
        """Emulate Px in velocity units, for all redshift bins,
        as a function of input likelihood parameters.
        theta_arcmin is a list of theta bins."""

        if verbose is None:
            verbose = self.verbose
        if zs is None:
            zs = self.zs
            iz_choice = np.arange(len(self.zs))
        else:
            if type(zs) == float or type(zs) == int:
                zs = [zs]
            assert [z in self.zs for z in zs], "Input redshifts not all found in theory zs"
            iz_choice = [np.argwhere(self.zs == z)[0, 0] for z in zs]
        zs = np.atleast_1d(zs)
        iz_choice = np.atleast_1d(iz_choice)
        if verbose:
            print("Evaluating theory at redshifts", zs)
            print("This will correspond to redshift bins with indices", iz_choice)
        Nz = len(zs)
        if Nz > 1 and k_AA.ndim == 1:
            k_AA = np.array([k_AA] * Nz) # if only one k_AA array provided, assume it's the same for all redshift bins
        if Nz > 1 and theta_arcmin.ndim == 1:
            theta_arcmin = np.array([theta_arcmin] * Nz) # if only one theta array provided, assume it's the same for all redshift bins

        theta_deg = np.atleast_1d(theta_arcmin) / 60.0
        
        # no matter the inputs, make sure like_params_obj is a list of LikelihoodParameter objects, and like_params is a dictionary of parameter values for easier handling below
        if like_params:  # should evaluate to false if empty list, None, or empty dict
            if type(like_params) == dict:
                likeparam_obj_list = likeparam_from_dict(like_params)
            else:
                assert type(like_params[0]) == LikelihoodParameter, "If like_params is not a dictionary, it must be a list of LikelihoodParameter objects"
                likeparam_obj_list = like_params # rename
                like_params = dict_from_likeparam(likeparam_obj_list)
            # make sure like_params (dict) is formatted correctly
            like_params = format_like_params_dict(iz_choice, like_params)
        else:
            likeparam_obj_list = None
        
        # set up input parameters. First, set all the defaults
        theory_inputs = copy.deepcopy(self.default_param_dict.copy())
        # replace with any user-provided values
        if like_params:
            for par in like_params:
                theory_inputs[par] = like_params[par]
        if self.verbose:
            print("Theory inputs for this redshift evaluation:", {key: theory_inputs[key] for key in theory_inputs if any([f"_{iz}" in key for iz in iz_choice])})

        # First, find out if theory inputs contain all the necessary arinyo coefficients. If so, we will avoid using the emulator.
        if self.has_all_arinyo_coeffs(iz_choice, theory_inputs):
            if verbose:
                print("All Arinyo coefficients found in likelihood parameters, using them")
            arinyo_coeffs = {}
            for par in self.arinyo_par_names:
                try:
                    arinyo_coeffs[par] = np.zeros(len(iz_choice))
                    for iiz, iz in enumerate(iz_choice):
                        par_iz = par + f"_{iz}"
                        arinyo_coeffs[par][iiz] = theory_inputs[par_iz] # this should never crash
                except:
                    print(par, "not found")

        # Use the emulator to predict the Arinyo coeffs if they're not already being fed in
        else:
            # figure out emulator calls
            emu_call = self.get_emulator_calls(
                iz_choice=iz_choice,
                theory_inputs=theory_inputs
            )
            if verbose:
                print("Using emulator to get the Arinyo coefficients")
            arinyo_coeffs = self.emulator.emulate_P3D_params(emu_call, zs)
            # check if the user provided any Arinyo coefficients, if so, overwrite the emulated ones with the user-provided ones
            if like_params is not None:
                for iiz, iz in enumerate(iz_choice):
                    for par in arinyo_coeffs.keys():
                        par_iz = par + f"_{iz}"
                        if par_iz in theory_inputs and theory_inputs[par_iz] is not None and not np.isnan(theory_inputs[par_iz]):
                            arinyo_coeffs[par][iiz] = theory_inputs[par_iz]
                            print("Warning! Overwriting emulated coefficient", par, "with user-provided value", theory_inputs[par_iz])
            
        
        # activate the arinyo model
        p3d_fun = self.p3d_model.P3D_Mpc_k_mu
    
        si_coeffs = {}
        if add_silicon:
            if verbose:
                print("Adding silicon contamination")
            # add silicon contamination
            for par in ["bias_SiIII", "beta_SiIII", "k_p_SiIII"]:
                for iz in iz_choice:
                    si_coeffs[par] = np.zeros(len(iz_choice))
                    if par+f"_{iz}" not in theory_inputs.keys() or theory_inputs[par+f"_{iz}"] is None or np.isnan(theory_inputs[par+f"_{iz}"]):
                        raise ValueError(
                            f"Parameter {par} not found in likelihood parameters, but add_silicon is True. Please provide values for bias_SiIII, beta_SiIII, and k_p_SiIII for all redshift bins, or set default values for all redshift bins using set_default_param_values()"
                        )
                    else:
                        si_coeffs[par][iz] = theory_inputs[par+f"_{iz}"]
        lyap3d = LyaP3D(zs, P3D_model=self.p3d_model, P3D_fun=p3d_fun, P3D_coeffs=arinyo_coeffs, Si_contam=add_silicon, contam_coeffs=si_coeffs, verbose=verbose)
        
        # get unit conversions
        dAA_dMpc_z, ddeg_dMpc_z = self.get_unit_conversions(zs, like_params=likeparam_obj_list)

        # compute input k, theta to Pcross computation in Mpc
        Nk = 0
        Ntheta = 0
        if Nz > 1:
            print("check 0")
            for iz in range(Nz):
                if len(k_AA[iz]) > Nk: # find the max length of k_AA across redshift bins
                    Nk = len(k_AA[iz])
                if len(theta_deg[iz]) > Ntheta:
                    Ntheta = len(theta_deg[iz])
            print("check 1")
        else:
            if len(k_AA) == 1:
                k_AA = k_AA[0]
            if len(theta_deg) == 1:
                theta_deg = theta_deg[0]
            Nk = len(k_AA)
            k_AA = [k_AA]
            # if theta is just 1 float
            if np.isscalar(theta_deg):
                Ntheta = 1
            else:
                Ntheta = len(theta_deg)
            theta_deg = [theta_deg]
        kin_Mpc = np.zeros((Nz, Nk))
        
        theta_in_Mpc = np.zeros((Nz, Ntheta))
        for iz in range(Nz):
            kin_Mpc[iz, : Nk] = k_AA[iz] * dAA_dMpc_z[iz]
            theta_in_Mpc[iz, : Ntheta] = theta_deg[iz] / ddeg_dMpc_z[iz]
        
        # predict Px
        px_pred_Mpc = lyap3d.model_Px(kin_Mpc, theta_in_Mpc)
        
        # move from Mpc to AA
        px_AA = np.zeros((Nz, Ntheta, Nk))
        for iz in range(Nz):
            px_AA[iz, :, :] = px_pred_Mpc[iz, :, : len(k_AA[iz])] * dAA_dMpc_z[iz]

        if return_arinyo_coeffs:
            return px_AA, arinyo_coeffs
        else:
            return px_AA
        
    def has_all_arinyo_coeffs(self, iz_choice, input_dict):
        """Check if the likelihood parameters have all the Arinyo coefficients"""
        arinyo_coeffs = ["bias", "beta", "q1", "kvav", "av", "bv", "kp"] # q2 is optional
        for coeff in arinyo_coeffs:
            for iz in iz_choice:
                coeff_iz = coeff+f"_{iz}"
                if coeff_iz not in input_dict.keys() or input_dict[coeff_iz] is None or np.isnan(input_dict[coeff_iz]):
                    return False
        return True
                

    def _load_P3D_model(self, p3d_label='arinyo'):
        
        """ This function reads cosmo paremeters dictionary and loads a P3D model, then sets it
        """
        if p3d_label == 'arinyo':
            from forestflow.model_p3d_arinyo import ArinyoModel
            arinyo = ArinyoModel(fid_cosmo=self.fid_cosmo)
        else:
            sys.exit("Error: no P3D model specified. Please choose a valid P3D model. Current option are 'arinyo'.")
        
        self.p3d_model = arinyo
    
    def get_param(self, param_name, iz_choice=None):
        """Utility function to get the value of a parameter from the default_param_dict, for a given redshift bin (iz_choice)"""
        if iz_choice is not None:
            key = f"{param_name}_{iz_choice}"
            if key in self.default_param_dict:
                return self.default_param_dict[key]
            else:
                raise ValueError(f"Parameter {key} not found in default_param_dict")
        elif ("_" not in param_name):
            # if no iz_choice provided, return a list of values for all redshift bins
            values = []
            for iz in range(len(self.zs)):
                key = f"{param_name}_{iz}"
                if key in self.default_param_dict:
                    values.append(self.default_param_dict[key])
                else:
                    raise ValueError(f"Parameter {key} not found in default_param_dict")
            return values
        else:
            return self.default_param_dict[param_name]

    
