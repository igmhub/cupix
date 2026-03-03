import numpy as np
import copy
import matplotlib.pyplot as plt
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.emulator import gp_emulator
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood import CAMB_model
from cupix.likelihood.model_contaminants import Contaminants
from cupix.likelihood.model_systematics import Systematics
from cupix.likelihood.model_igm import IGM
from cupix.likelihood.cosmologies import set_cosmo
from cupix.utils.utils_sims import get_training_hc
from cupix.utils.hull import Hull
from cupix.utils.utils import is_number_string
from cupix.likelihood.window_and_rebin import convolve_window
from cupix.likelihood.lyaP3D import LyaP3D
from cupix.likelihood.likelihood_parameter import likeparam_from_dict, LikelihoodParameter, dict_from_likeparam, format_like_params_dict
import sys
from forestflow import priors
from astropy.io import fits

def set_theory(
    zs, bkgd_cosmo, default_theory, p3d_label, emulator_label, k_unit='iAA', verbose=False
):
    """Set theory"""


    # set theory
    theory = Theory(
        zs = zs,
        bkgd_cosmo=bkgd_cosmo,
        default_lya_theory=default_theory,
        p3d_label=p3d_label,
        emulator_label = emulator_label,
        k_unit = k_unit,
        verbose=verbose
    )

    return theory


class Theory(object):
    """Translator between the likelihood object and the emulator. This object
    will map from a set of CAMB parameters directly to emulator calls, without
    going through our Delta^2_\star parametrisation"""

    def __init__(
        self,
        zs,
        bkgd_cosmo=None,
        default_lya_theory='best_fit_arinyo_from_p1d',
        p3d_label='arinyo',
        emulator_label=None,
        verbose=False,
        z_star=3.0,
        kp_kms=0.009,
        use_star_priors=None,
        k_unit='iAA' # either iAA or ikms
    ):
        """Setup object to compute predictions for the 1D power spectrum.
        Inputs:
            - zs: redshifts that will be evaluated
            - emulator: object to interpolate simulated p1d
            - verbose: print information, useful to debug
            - F_model: mean flux model
            - T_model: thermal model
            - P_model: pressure model
            - metal_models: list of metal models to include
            - hcd_model: model for HCD contamination
            - bkgd_cosmo: dictionary of background cosmology parameters used for unit conversions, etc
            - fid_sim_igm: IGM model assumed
            - true_sim_igm: if not None, true IGM model of the mock
        """

        self.verbose = verbose

        # specify pivot point used in compressed parameters
        self.z_star = z_star
        self.kp_kms = kp_kms
        self.use_star_priors = use_star_priors
        self.k_unit = k_unit
        self.set_cosmo_dict(bkgd_cosmo)
        laceCosmo = camb_cosmo.get_cosmology_from_dictionary(bkgd_cosmo)
       
        if emulator_label == "forestflow_emu": # not really sure what this does
            self.emu_kp_Mpc = 0.7  # not really sure what this does

        # I don't know what this part does either but it is necessary for set_fid_cosmo
        res = get_training_hc("mpg")
        self.emu_pars = res[0]
        self.hc_points = res[1]
        self.emu_cosmo_all = res[2]
        self.emu_igm_all = res[3]
        self._load_P3D_model(p3d_label=p3d_label)
        self.set_fid_cosmo(zs, input_cosmo=laceCosmo)
        self.default_lya_theory = default_lya_theory
        self.set_default_param_values(tag=default_lya_theory)
        if 'arinyo' in default_lya_theory:
            self.param_names = ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]
        elif 'igm' in default_lya_theory:
            self.param_names = ["Delta2_p", "n_p", "mF", "gamma", "sigT_Mpc", "kF_Mpc", "lambda_P"]
        else:
            raise ValueError("default_lya_theory tag not recognized, parameters not implemented")
        if emulator_label == "forestflow_emu":
            self.emulator = FF_emulator(zs, self.bkgd_cosmo)
        else:
            print("Warning: no emulator specified, theory will not be able to make predictions")

    def set_cosmo_dict(self, bkgd_cosmo):
        omnuh2 = 0.0006
        mnu = omnuh2 * 93.14
        H0 = 67.36
        omch2 = 0.12
        ombh2 = 0.02237
        As = 2.1e-9
        ns = 0.9649
        nrun = 0.0
        w = -1.0
        omk = 0
        cosmo_dict = {
            'H0': H0,
            'omch2': omch2,
            'ombh2': ombh2,
            'mnu': mnu,
            'omk': omk,
            'As': As,
            'ns': ns,
            'nrun': nrun,
            'w': w
        }
        if bkgd_cosmo is not None:
            for key in bkgd_cosmo:
                if key in cosmo_dict:
                    cosmo_dict[key] = bkgd_cosmo[key] # update default value with user-provided value
                else:
                    print(f"Warning: {key} not recognized as a cosmological parameter, ignoring it.")
        self.bkgd_cosmo = cosmo_dict

    def set_fid_cosmo(
        self, zs, zs_hires=None, input_cosmo=None, extra_factor=1.15
    ):
        """Setup fiducial cosmology"""

        self.zs = zs
        self.zs_hires = zs_hires


        # setup fiducial cosmology (used for fitting)
        if input_cosmo is None:
            input_cosmo = camb_cosmo.get_cosmology()
    
        # setup CAMB object for the fiducial cosmology and precompute some things
        if self.zs_hires is not None:
            _zs = np.concatenate([self.zs, self.zs_hires, [self.z_star]])
        else:
            _zs = np.concatenate([self.zs, [self.z_star]])
        _zs = np.unique(_zs)

        self.fid_cosmo = {}
        self.fid_cosmo["zs"] = _zs
        self.fid_cosmo["cosmo"] = CAMB_model.CAMBModel(
            zs=_zs,
            cosmo=input_cosmo,
            z_star=self.z_star,
            kp_kms=self.kp_kms,
        )
        # self.fid_cosmo["linP_Mpc_params"] = self.fid_cosmo[
        #     "cosmo"
        # ].get_linP_Mpc_params(kp_Mpc=self.emu_kp_Mpc) # this part is calling camb
        self.fid_cosmo["dkms_dMpc_zs"] = self.fid_cosmo["cosmo"].get_dkms_dMpc()
        self.fid_cosmo["dAA_dMpc_zs"] = self.fid_cosmo["cosmo"].get_dAA_dMpc()
        self.fid_cosmo["ddeg_dMpc_zs"] = self.fid_cosmo["cosmo"].get_ddeg_dMpc()
        # self.fid_cosmo["linP_params"] = self.fid_cosmo[
            # "cosmo"
        # ].get_linP_params()

        # when using a fiducial cosmology, easy to change in other cases (TODO)
        # self.set_cosmo_priors()

    def set_cosmo_priors(self, extra_factor=1.25):
        """Set priors for cosmological parameters

        We get the priors on As, ns, and nrun from differences in star parameters in the training set
        Only works when using a fiducial cosmology
        """

        # pivot scale of primordial power
        ks_Mpc = self.fid_cosmo["cosmo"].cosmo.InitPower.pivot_scalar

        # likelihood pivot point, in velocity units
        dkms_dMpc = self.fid_cosmo["cosmo"].dkms_dMpc(self.z_star)
        kp_Mpc = self.kp_kms * dkms_dMpc

        # logarithm of ratio of pivot points
        ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

        fid_As = self.fid_cosmo["cosmo"].cosmo.InitPower.As
        fid_ns = self.fid_cosmo["cosmo"].cosmo.InitPower.ns
        fid_nrun = self.fid_cosmo["cosmo"].cosmo.InitPower.nrun

        fid_Astar = self.fid_cosmo["linP_params"]["Delta2_star"]
        fid_nstar = self.fid_cosmo["linP_params"]["n_star"]
        fid_alphastar = self.fid_cosmo["linP_params"]["alpha_star"]

        if self.use_star_priors is not None:
            self.star_priors = {}
            for key in self.use_star_priors:
                self.star_priors[key] = self.use_star_priors[key]
        else:
            self.star_priors = None

        hc_fid = {}
        hc_fid["As"] = []
        hc_fid["ns"] = []
        hc_fid["nrun"] = []

        for key in self.emu_cosmo_all:
            cos = self.emu_cosmo_all[key]
            if is_number_string(cos["sim_label"][-1]) == False:
                continue
            test_Astar = cos["star_params"]["Delta2_star"]
            test_nstar = cos["star_params"]["n_star"]
            test_alphastar = cos["star_params"]["alpha_star"]

            ln_ratio_Astar = np.log(test_Astar / fid_Astar)
            delta_nstar = test_nstar - fid_nstar
            delta_alphastar = test_alphastar - fid_alphastar

            delta_nrun = delta_alphastar
            delta_ns = delta_nstar - delta_nrun * ln_kp_ks
            ln_ratio_As = (
                ln_ratio_Astar
                - (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
            )
            hc_fid["nrun"].append(fid_nrun + delta_nrun)
            hc_fid["ns"].append(fid_ns + delta_ns)
            hc_fid["As"].append(fid_As * np.exp(ln_ratio_As))

        hc_fid["As"] = np.array(hc_fid["As"])
        hc_fid["ns"] = np.array(hc_fid["ns"])
        hc_fid["nrun"] = np.array(hc_fid["nrun"])

        self.cosmo_priors = {
            "As": np.array([hc_fid["As"].min(), hc_fid["As"].max()]),
            "ns": np.array([hc_fid["ns"].min(), hc_fid["ns"].max()]),
            "nrun": np.array([hc_fid["nrun"].min(), hc_fid["nrun"].max()]),
        }

        for par in self.cosmo_priors:
            for ii in range(2):
                if (ii == 0) and (self.cosmo_priors[par][ii] < 0):
                    self.cosmo_priors[par][ii] *= extra_factor
                elif (ii == 0) and (self.cosmo_priors[par][ii] >= 0):
                    self.cosmo_priors[par][ii] *= 1 - (extra_factor - 1)
                elif (ii == 1) and (self.cosmo_priors[par][ii] < 0):
                    self.cosmo_priors[par][ii] *= 1 - (extra_factor - 1)
                elif (ii == 1) and (self.cosmo_priors[par][ii] >= 0):
                    self.cosmo_priors[par][ii] *= extra_factor

    def fixed_background(self, like_params):
        """Check if any of the input likelihood parameters would change
        the background expansion of the fiducial cosmology"""
        if like_params is None:
            return True
        # look for parameters that would change background
        for par in like_params:
            if par.name in ["ombh2", "omch2", "H0", "mnu", "cosmomc_theta"]:
                return False

        return True

    def get_linP_Mpc_params_from_fiducial(
        self, zs, like_params, return_derivs=False
    ):
        """Recycle linP_Mpc_params from fiducial model, when only varying
        primordial power spectrum (As, ns, nrun)"""

        # make sure you are not changing the background expansion
        assert self.fixed_background(like_params)

        zs = np.atleast_1d(zs)

        # differences in primordial power (at CMB pivot point)
        ratio_As = 1.0
        delta_ns = 0.0
        delta_nrun = 0.0
        for par in like_params:
            if par.name == "As":
                fid_As = self.fid_cosmo["cosmo"].cosmo.InitPower.As
                ratio_As = par.value / fid_As
            if par.name == "ns":
                fid_ns = self.fid_cosmo["cosmo"].cosmo.InitPower.ns
                delta_ns = par.value - fid_ns
            if par.name == "nrun":
                fid_nrun = self.fid_cosmo["cosmo"].cosmo.InitPower.nrun
                delta_nrun = par.value - fid_nrun

        # pivot scale in primordial power
        ks_Mpc = self.fid_cosmo["cosmo"].cosmo.InitPower.pivot_scalar
        # logarithm of ratio of pivot points
        ln_kp_ks = np.log(self.emu_kp_Mpc / ks_Mpc)

        # compute scalings
        delta_alpha_p = delta_nrun
        delta_n_p = delta_ns + delta_nrun * ln_kp_ks
        ln_ratio_A_p = (
            np.log(ratio_As)
            + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
        )

        # update values of linP_params at emulator pivot point, at each z
        linP_Mpc_params = []
        for z in zs:
            _ = np.argwhere(self.fid_cosmo["zs"] == z)[0, 0]
            zlinP = self.fid_cosmo["linP_Mpc_params"][_]
            linP_Mpc_params.append(
                {
                    "Delta2_p": zlinP["Delta2_p"] * np.exp(ln_ratio_A_p),
                    "n_p": zlinP["n_p"] + delta_n_p,
                    "alpha_p": zlinP["alpha_p"] + delta_alpha_p,
                }
            )

        if return_derivs:
            val_derivs = {}
            _ = np.argwhere(self.fid_cosmo["zs"] == self.z_star)[0, 0]
            zlinP = self.fid_cosmo["linP_Mpc_params"][_]

            val_derivs["Delta2star"] = zlinP["Delta2_p"] * np.exp(ln_ratio_A_p)
            val_derivs["nstar"] = zlinP["n_p"] + delta_n_p
            val_derivs["alphastar"] = zlinP["alpha_p"] + delta_alpha_p

            val_derivs["der_alphastar_nrun"] = 1
            val_derivs["der_alphastar_ns"] = 0
            val_derivs["der_alphastar_As"] = 0

            val_derivs["der_nstar_nrun"] = ln_kp_ks
            val_derivs["der_nstar_ns"] = 1
            val_derivs["der_nstar_As"] = 0

            val_derivs["der_Delta2star_nrun"] = (
                0.5 * val_derivs["Delta2star"] * ln_kp_ks**2
            )
            val_derivs["der_Delta2star_ns"] = (
                val_derivs["Delta2star"] * ln_kp_ks
            )
            val_derivs["der_Delta2star_As"] = val_derivs["Delta2star"] / (
                ratio_As * fid_As
            )

            return linP_Mpc_params, val_derivs
        else:
            return linP_Mpc_params


    def get_unit_conversions(self, zs, like_params, return_blob=False):
        """return conversion from Mpc to km/s or Mpc to inv Angstroms,
            and from Mpc to deg for transverse separations"""
        
        # compute linear power parameters at all redshifts, and H(z) / (1+z)
        if self.fixed_background(like_params):
            # use background and transfer functions from fiducial cosmology
            if self.verbose:
                print("recycle transfer function")
            dkms_dMpc_zs  = []
            ddeg_dMpc_zs = []
            dAA_dMpc_zs   = []
            for z in zs:
                _ = np.argwhere(self.fid_cosmo["zs"] == z)[0, 0]
                dkms_dMpc_zs.append(self.fid_cosmo["dkms_dMpc_zs"][_])
                ddeg_dMpc_zs.append(self.fid_cosmo["ddeg_dMpc_zs"][_])
                dAA_dMpc_zs.append(self.fid_cosmo["dAA_dMpc_zs"][_])
            dkms_dMpc_zs  = np.array(dkms_dMpc_zs)
            ddeg_dMpc_zs = np.array(ddeg_dMpc_zs)
            dAA_dMpc_zs   = np.array(dAA_dMpc_zs)
            if return_blob:
                blob = self.get_blob_fixed_background(like_params)
        else:
            # setup a new CAMB_model from like_params
            if self.verbose:
                print("create new CAMB_model")
            camb_model = self.fid_cosmo["cosmo"].get_new_model(zs, like_params)
            dkms_dMpc_zs = camb_model.get_dkms_dMpc()
            dAA_dMpc_zs = camb_model.get_dAA_dMpc()
            ddeg_dMpc_zs = camb_model.get_ddeg_dMpc()
            if return_blob:
                blob = self.get_blob(camb_model=camb_model)
        if self.k_unit == 'ikms':
            return_conv_k_of_zs = dkms_dMpc_zs
        else:
            return_conv_k_of_zs = dAA_dMpc_zs
        if return_blob:
            return return_conv_k_of_zs, ddeg_dMpc_zs, blob
        else:
            return return_conv_k_of_zs, ddeg_dMpc_zs
            
    def get_emulator_calls(
        self, iz_choice, theory_inputs
    ):
        """Compute models that will be emulated, one per redshift bin.
        - like_params identify likelihood parameters to use."""

        # store emulator calls
        emu_call = {}
        # check if using the base parameters
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

    def get_blobs_dtype(self):
        """Return the format of the extra information (blobs) returned
        by get_p1d_kms and used in the fitter."""

        blobs_dtype = [
            ("Delta2_star", float),
            ("n_star", float),
            ("alpha_star", float),
            ("f_star", float),
            ("g_star", float),
            ("H0", float),
        ]
        return blobs_dtype

    def get_blob(self, camb_model=None):
        """Return extra information (blob) for the fitter."""

        if camb_model is None:
            Nblob = len(self.get_blobs_dtype())
            if Nblob == 1:
                return np.nan
            else:
                out = np.nan, *([np.nan] * (Nblob - 1))
                return out
        else:
            # compute linear power parameters for input cosmology
            params = self.fid_cosmo["cosmo"].get_linP_params()
            return (
                params["Delta2_star"],
                params["n_star"],
                params["alpha_star"],
                params["f_star"],
                params["g_star"],
                camb_model.cosmo.H0,
            )

    def get_blob_fixed_background(self, like_params, return_derivs=False):
        """Fast computation of blob when running with fixed background"""

        # make sure you are not changing the background expansion
        assert self.fixed_background(like_params)

        # differences in primordial power (at CMB pivot point)
        ratio_As = 1.0
        delta_ns = 0.0
        delta_nrun = 0.0
        for par in like_params:
            if par.name == "As":
                fid_As = self.fid_cosmo["cosmo"].cosmo.InitPower.As
                ratio_As = par.value / fid_As
            if par.name == "ns":
                fid_ns = self.fid_cosmo["cosmo"].cosmo.InitPower.ns
                delta_ns = par.value - fid_ns
            if par.name == "nrun":
                fid_nrun = self.fid_cosmo["cosmo"].cosmo.InitPower.nrun
                delta_nrun = par.value - fid_nrun

        # pivot scale of primordial power
        ks_Mpc = self.fid_cosmo["cosmo"].cosmo.InitPower.pivot_scalar

        # likelihood pivot point, in velocity units
        dkms_dMpc = self.fid_cosmo["cosmo"].dkms_dMpc(self.z_star)
        kp_Mpc = self.kp_kms * dkms_dMpc

        # logarithm of ratio of pivot points
        ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

        # get blob for fiducial cosmo
        ### TODO: make this more efficient! Maybe directly storing the params?
        fid_blob = self.get_blob(self.fid_cosmo["cosmo"])

        # rescale blobs
        delta_alpha_star = delta_nrun
        delta_n_star = delta_ns + delta_nrun * ln_kp_ks
        ln_ratio_A_star = (
            np.log(ratio_As)
            + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
        )

        alpha_star = fid_blob[2] + delta_alpha_star
        n_star = fid_blob[1] + delta_n_star
        Delta2_star = fid_blob[0] * np.exp(ln_ratio_A_star)

        linP_Mpc_params = (Delta2_star, n_star, alpha_star) + fid_blob[3:]

        if return_derivs:
            val_derivs = {}

            val_derivs["Delta2star"] = Delta2_star
            val_derivs["nstar"] = n_star
            val_derivs["alphastar"] = alpha_star

            val_derivs["der_alphastar_nrun"] = 1
            val_derivs["der_alphastar_ns"] = 0
            val_derivs["der_alphastar_As"] = 0

            val_derivs["der_nstar_nrun"] = ln_kp_ks
            val_derivs["der_nstar_ns"] = 1
            val_derivs["der_nstar_As"] = 0

            val_derivs["der_Delta2star_nrun"] = (
                0.5 * val_derivs["Delta2star"] * ln_kp_ks**2
            )
            val_derivs["der_Delta2star_ns"] = (
                val_derivs["Delta2star"] * ln_kp_ks
            )
            val_derivs["der_Delta2star_As"] = val_derivs["Delta2star"] / (
                ratio_As * fid_As
            )

            return linP_Mpc_params, val_derivs
        else:
            return linP_Mpc_params

    def set_default_param_values(self, tag='best_fit_arinyo_from_p1d'):

        """Set default set of likelihood parameters.
        Param_format options are 'arinyo' or 'cosmo_igm'.
        Tag options are 'p1d' to input priors from the p1d paper,
        'colore' to get the Laura's best-fit parameters (arinyo only)."""

        if tag=='colore':
            param_format = 'arinyo'
            # warn
            print("Warning: tag 'colore' is only compatible with param_format 'arinyo', setting param_format to 'arinyo'")
            ari_pp_names = ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]
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
        verbose=None
    ):
        """Emulate Px in velocity units, for all redshift bins,
        as a function of input likelihood parameters.
        theta_arcmin is a list of theta bins.
        It might also return a covariance from the emulator,
        or a blob with extra information for the fitter."""

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

        
        # Use the emulator to predict the Arinyo coeffs if they're not already being fed in
        if "igm" in self.default_lya_theory:
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
                            if verbose:
                                print("Overwriting emulated coefficient", par, "with user-provided value", theory_inputs[par_iz])
        
        elif "arinyo" in self.default_lya_theory:
            if not self.has_all_arinyo_coeffs(iz_choice, theory_inputs):
                raise ValueError(
                "Not all Arinyo coefficients found in likelihood parameters, and/or default values not set for all redshift bins. Please provide values for bias, beta, q1, kvav, av, bv, and kp for all redshift bins, or set default values for all redshift bins using set_default_param_values()"
            )
            else:
                if verbose:
                    print("All Arinyo coefficients found in likelihood parameters, using them")
                arinyo_coeffs = {}
                for par in self.param_names:
                    arinyo_coeffs[par] = np.zeros(len(iz_choice))
                    for iz in iz_choice:
                        par_iz = par + f"_{iz}"
                        if par_iz in theory_inputs.keys():
                            arinyo_coeffs[par][iz] = theory_inputs[par_iz]
                        else:
                            print("Warning: parameter", par_iz, "not found in theory inputs, setting to 0.")
        
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

        return px_AA
        
    def has_all_arinyo_coeffs(self, iz_choice, input_dict):
        """Check if the likelihood parameters have all the Arinyo coefficients"""
        arinyo_coeffs = ["bias", "beta", "q1", "kvav", "av", "bv", "kp"] # q2 is optional
        for coeff in arinyo_coeffs:
            for iz in iz_choice:
                coeff_iz = coeff+f"_{iz}"
                print(coeff_iz, input_dict[coeff_iz])
                if coeff_iz not in input_dict.keys() or input_dict[coeff_iz] is None or np.isnan(input_dict[coeff_iz]):
                    return False
        return True
                

    def _load_P3D_model(self, p3d_label='arinyo'):
        
        """ This function reads cosmo paremeters dictionary and loads a P3D model, then sets it
        """
        if p3d_label == 'arinyo':
            from forestflow.model_p3d_arinyo import ArinyoModel
            arinyo = ArinyoModel(cosmo=self.bkgd_cosmo) # set model
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

    