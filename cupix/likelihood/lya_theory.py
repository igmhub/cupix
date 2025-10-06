import numpy as np
import copy
import matplotlib.pyplot as plt
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.emulator import gp_emulator

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

def set_theory(
    args, emulator, free_parameters=None, use_hull=True, fid_or_true="fid", k_unit='iAA'
):
    """Set theory"""


    if fid_or_true == "fid":
        pars_igm = args.fid_igm
        pars_cont = args.fid_cont
        pars_syst = args.fid_syst
    elif fid_or_true == "true":
        pars_igm = args.true_igm
        pars_cont = args.true_cont
        pars_syst = args.true_syst
    else:
        raise ValueError("fid_or_true must be 'fid' or 'true'")
    args.pars_igm = pars_igm

    # set igm model
    model_igm = IGM(free_param_names=free_parameters, pars_igm=pars_igm)
    
    # set contaminants
    model_cont = Contaminants(
        free_param_names=free_parameters,
        pars_cont=pars_cont,
        ic_correction=args.ic_correction,
    )
    # set systematics
    model_syst = Systematics(
        free_param_names=free_parameters, pars_syst=pars_syst
    )

    

    # set theory
    theory = Theory(
        emulator=emulator,
        model_igm=model_igm,
        model_cont=model_cont,
        model_syst=model_syst,
        use_hull=use_hull,
        use_star_priors=args.use_star_priors,
        z_star=args.z_star,
        kp_kms=args.kp_kms,
        k_unit = k_unit
    )

    return theory


class Theory(object):
    """Translator between the likelihood object and the emulator. This object
    will map from a set of CAMB parameters directly to emulator calls, without
    going through our Delta^2_\star parametrisation"""

    def __init__(
        self,
        emulator=None,
        model_igm=None,
        model_cont=None,
        model_syst=None,
        use_hull=True,
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
            - fid_cosmo: fiducial cosmology used for fixed parameters
            - fid_sim_igm: IGM model assumed
            - true_sim_igm: if not None, true IGM model of the mock
        """

        self.verbose = verbose

        # specify pivot point used in compressed parameters
        self.z_star = z_star
        self.kp_kms = kp_kms
        self.use_hull = use_hull
        self.use_star_priors = use_star_priors
        self.k_unit = k_unit
        # setup emulator
        if emulator is None:
            raise ValueError("Emulator not specified")
        else:
            self.emulator = emulator
        self.emu_kp_Mpc = self.emulator.kp_Mpc
        res = get_training_hc("mpg")
        self.emu_pars = res[0]
        self.hc_points = res[1]
        self.emu_cosmo_all = res[2]
        self.emu_igm_all = res[3]

        # setup model_igm
        if model_igm is None:
            print("Model_IGM is None, this isn't implemented yet.")
            self.model_igm = IGM() # should add some default behavior here
        else: 
            self.model_igm = model_igm
            print("Using input IGM model.")

        # setup model_cont
        if model_cont is None:
            self.model_cont = Contaminants()
        else:
            self.model_cont = model_cont

        # setup model_syst
        if model_syst is None:
            self.model_syst = Systematics()
        else:
            self.model_syst = model_syst

    def set_fid_cosmo(
        self, zs, zs_hires=None, input_cosmo=None, extra_factor=1.15
    ):
        """Setup fiducial cosmology"""

        self.zs = zs
        self.zs_hires = zs_hires

        if self.use_hull:
            self.hull = Hull(
                zs=zs,
                data_hull=self.hc_points,
                suite="mpg",
                extra_factor=extra_factor,
            )
            if zs_hires is not None:
                if len(zs) == len(zs_hires):
                    self.hull_hires = self.hull
                else:
                    self.hull_hires = Hull(
                        zs=zs_hires,
                        data_hull=self.hc_points,
                        suite=self.emulator.list_sim_cube[0][:3],
                        extra_factor=extra_factor,
                    )

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
        self.fid_cosmo["linP_Mpc_params"] = self.fid_cosmo[
            "cosmo"
        ].get_linP_Mpc_params(kp_Mpc=self.emu_kp_Mpc)
        self.fid_cosmo["M_kms_of_zs"] = self.fid_cosmo["cosmo"].get_M_kms_of_zs()
        self.fid_cosmo["M_AA_of_zs"] = self.fid_cosmo["cosmo"].get_M_AA_of_zs()
        self.fid_cosmo["M_tdeg_of_zs"] = self.fid_cosmo["cosmo"].get_M_tdeg_of_zs()
        self.fid_cosmo["linP_params"] = self.fid_cosmo[
            "cosmo"
        ].get_linP_params()

        # when using a fiducial cosmology, easy to change in other cases (TODO)
        self.set_cosmo_priors()

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

    def get_emulator_calls(
        self, zs, like_params=[], return_M_of_z=True, return_blob=False
    ):
        """Compute models that will be emulated, one per redshift bin.
        - like_params identify likelihood parameters to use.
        - return_M_of_z will also return conversion from Mpc to km/s
            and from Mpc to deg for transverse separations
        - return_blob will return extra information about the call."""

        # compute linear power parameters at all redshifts, and H(z) / (1+z)
        if self.fixed_background(like_params):
            # use background and transfer functions from fiducial cosmology
            if self.verbose:
                print("recycle transfer function")
            linP_Mpc_params = self.get_linP_Mpc_params_from_fiducial(
                zs, like_params
            )
            M_kms_of_zs  = []
            M_tdeg_of_zs = []
            M_AA_of_zs   = []
            for z in zs:
                _ = np.argwhere(self.fid_cosmo["zs"] == z)[0, 0]
                M_kms_of_zs.append(self.fid_cosmo["M_kms_of_zs"][_])
                M_tdeg_of_zs.append(self.fid_cosmo["M_tdeg_of_zs"][_])
                M_AA_of_zs.append(self.fid_cosmo["M_AA_of_zs"][_])
            M_kms_of_zs  = np.array(M_kms_of_zs)
            M_tdeg_of_zs = np.array(M_tdeg_of_zs)
            M_AA_of_zs   = np.array(M_AA_of_zs)
            if return_blob:
                blob = self.get_blob_fixed_background(like_params)
        else:
            # setup a new CAMB_model from like_params
            if self.verbose:
                print("create new CAMB_model")
            camb_model = self.fid_cosmo["cosmo"].get_new_model(zs, like_params)
            linP_Mpc_params = camb_model.get_linP_Mpc_params(
                kp_Mpc=self.emu_kp_Mpc
            )
            M_kms_of_zs = camb_model.get_M_kms_of_zs()
            M_AA_of_zs = camb_model.get_M_AA_of_zs()
            M_tdeg_of_zs = camb_model.get_M_tdeg_of_zs()
            if return_blob:
                blob = self.get_blob(camb_model=camb_model)

        # store emulator calls
        emu_call = {}
        # check if using the base parameters

        for key in self.emulator.emu_params:
            if key in [par.name for par in like_params]: # if this parameter will be used in the likelihood
                print("Found parameter", key, "in likelihood parameters")

                if (key == "Delta2_p") | (key == "n_p") | (key == "alpha_p"):
                    emu_call[key] = np.zeros(len(zs))
                    for ii in range(len(linP_Mpc_params)):
                        emu_call[key][ii] = linP_Mpc_params[ii][key]
                elif key == "mF":
                    emu_call[key] = self.model_igm.models["F_model"].get_mean_flux(
                        zs, like_params=like_params
                    )
                elif key == "gamma":
                    emu_call[key] = self.model_igm.models["T_model"].get_gamma(
                        zs, like_params=like_params
                    )
                elif key == "sigT_Mpc":
                    emu_call[key] = (
                        self.model_igm.models["T_model"].get_sigT_kms(
                            zs, like_params=like_params
                        )
                        / M_kms_of_zs
                    )
                elif key == "kF_Mpc":
                    emu_call[key] = (
                        self.model_igm.models["P_model"].get_kF_kms(
                            zs, like_params=like_params
                        )
                        * M_kms_of_zs
                    )
                elif key == "lambda_P":
                    emu_call[key] = 1000 / (
                        self.model_igm.models["P_model"].get_kF_kms(
                            zs, like_params=like_params
                        )
                        * M_kms_of_zs
                    )
                elif key in ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]:
                    # feed in the direct Arinyo model input parameters
                    for par in like_params:
                        if par.name==key:
                            # make sure a value was provided; currently no default model implemented
                            if par.value is None:
                                
                                raise ValueError(
                                    f"Parameter {key} not found in likelihood parameters"
                                )
                            else:
                                emu_call[key] = np.atleast_1d(par.value) # if the likelihood params are stored as lists per z (emu_call[key] should be a list /array per z (shape 0 is Nz))
                    # print("Found parameter", key, "in likelihood parameters, setting it to", emu_call[key])
                else:
                    raise ValueError(
                        "Not a theory model for emulator parameter", key
                    )

        if return_M_of_z:
            if self.k_unit == 'ikms':
                    return_M_k_of_zs = M_kms_of_zs
            else:
                return_M_k_of_zs = M_AA_of_zs
            if return_blob:
                return emu_call, return_M_k_of_zs, M_tdeg_of_zs, blob
            else:
                return emu_call, return_M_k_of_zs, M_tdeg_of_zs,
        else:
            if return_blob:
                return emu_call, blob
            else:
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

    
    def get_px_AA(
        self,
        zs,
        k_AA,
        theta_arcmin,
        like_params=[],
        return_covar=False,
        return_blob=True,
        return_emu_params=False,
        add_silicon=False
    ):
        """Emulate Px in velocity units, for all redshift bins,
        as a function of input likelihood parameters.
        theta_arcmin is a list of theta bins.
        It might also return a covariance from the emulator,
        or a blob with extra information for the fitter."""
        

        zs = np.atleast_1d(zs)
        Nz = len(zs)
        theta_deg = np.atleast_1d(theta_arcmin) / 60.0
        # figure out emulator calls
        emu_call, M_AA_of_z, M_tdeg_of_z, blob = self.get_emulator_calls(
            zs,
            like_params=like_params,
            return_M_of_z=True,
            return_blob=True,
        )
        

        # also apply priors on compressed parameters
        # temporary hack
        dict_trans = {
            "Delta2_star": 0,
            "n_star": 1,
            "alpha_star": 2,
        }
        if self.star_priors is not None:
            for key in self.star_priors:
                _ = np.argwhere(
                    (blob[dict_trans[key]] > self.star_priors[key][1])
                    | (blob[dict_trans[key]] < self.star_priors[key][0])
                )
                if len(_) > 0:
                    print("Returning none because star")
                    return None


    
        # compute input k, theta to emulator in Mpc
        
        Nk = 0
        Ntheta = 0
        if Nz > 1:
            for iz in range(Nz):
                if len(k_AA[iz]) > Nk:
                    Nk = len(k_AA[iz])
                if len(theta_deg[iz]) > Ntheta:
                    Ntheta = len(theta_deg[iz])
        else:
            if len(k_AA) == 1:
                k_AA = k_AA[0]
            if len(theta_deg) == 1:
                theta_deg = theta_deg[0]

            Nk = len(k_AA)
            k_AA = [k_AA]
            Ntheta = len(theta_deg)
            theta_deg = [theta_deg]
        
        kin_Mpc = np.zeros((Nz, Nk))
        theta_in_Mpc = np.zeros((Nz, Ntheta))
        
        for iz in range(Nz):
            kin_Mpc[iz, : len(k_AA[iz])] = k_AA[iz] * M_AA_of_z[iz]
            theta_in_Mpc[iz, : len(theta_deg[iz])] = theta_deg[iz] / M_tdeg_of_z[iz]
        
        # get the Arinyo coeffs if they're not already being fed in (e.g. if the likelihood parameters don't include them)
        if not self.has_all_arinyo_coeffs(like_params):
            print("Arinyo coefficients not found in likelihood parameters, using emulator to get them")
            arinyo_coeffs = self.emulator.emulate_P3D_params(emu_call)
        else:
            print("Arinyo coefficients found in likelihood parameters, using them")
            arinyo_coeffs = {}
            for par in like_params:
                if par.name in ["bias", "beta", "q1", "kvav", "av", "bv", "kp"]:
                    if par.value is None:
                        raise ValueError(
                            f"Parameter `{par.name}` was not assigned a value, but is required."
                        )
                    arinyo_coeffs[par.name] = np.atleast_1d(par.value)
        # activate the arinyo model
        p3d_model = self.emulator.arinyo.P3D_Mpc
        si_coeffs = {}
        if add_silicon:
            print("Adding silicon contamination")
            # add silicon contamination
            for par in like_params:
                if par.name in ["bias_SiIII", "beta_SiIII", "k_p_SiIII"]:
                    si_coeffs[par.name] = par.value

        lyap3d = LyaP3D(zs, p3d_model, arinyo_coeffs, Si_contam=add_silicon, contam_coeffs=si_coeffs, Arinyo=self.emulator.arinyo)
        px_pred_Mpc = lyap3d.model_Px(kin_Mpc, theta_in_Mpc)
        # move from Mpc to AA
        px_AA = np.zeros((Nz, Ntheta, Nk))
        covars = []
        for iz in range(Nz):
            px_AA[iz, :, :] = px_pred_Mpc[iz, :, : len(k_AA[iz])] * M_AA_of_z[iz]

        # decide what to return, and return it
        out = [px_AA] # np.asarray([px_kms])
        if return_covar:
            out.append(covars)
        if return_blob:
            out.append(blob)
        if return_emu_params:
            out.append(emu_call)

        if len(out) == 1:
            return out[0]
        else:
            return out
        
    def has_all_arinyo_coeffs(self, likelihood_params):
        """Check if the likelihood parameters have all the Arinyo coefficients"""
        arinyo_coeffs = ["bias", "beta", "q1", "kvav", "av", "bv", "kp"] # q2 is optional
        for coeff in arinyo_coeffs:
            if coeff not in [par.name for par in likelihood_params]:
                return False
        return True
