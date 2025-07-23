import sys, os, configargparse
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from cupix.utils.utils import (
    create_print_function,
    mpi_hello_world,
    get_path_repo,
)

def parse_args():
    """
    Parse input arguments

    Returns
    -------
    args : Namespace
        Namespace of input arguments
    """
    parser = configargparse.ArgumentParser(
        description="Passing options to sampler"
    )

    # emulator
    parser.add_argument(
        "--emulator_label",
        default=None,
        choices=[
            "Pedersen21",
            "Pedersen21_ext",
            "Pedersen21_ext8",
            "Pedersen23",
            "Pedersen23_ext",
            "Pedersen23_ext8",
            "CH24",
            "Cabayol23",
            "Cabayol23_extended",
            "Cabayol23+",  # recommended for mpg
            "Cabayol23+_extended",  # recommended for mpg small scales
            "Nyx_v0",
            # "Nyx_v0_extended",
            "Nyx_alphap",  # recommended for nyx
        ],
        required=True,
        help="Type of emulator to be used",
    )
    parser.add_argument(
        "--data_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to create mock P1Ds",
    )
    parser.add_argument(
        "--data_label_hires",
        default=None,
        type=str,
        help="Input simulation to create mock P1Ds",
    )
    parser.add_argument(
        "--z_min",
        type=float,
        default=2,
        help="Minimum redshift of P1D measurements to be analyzed",
    )
    parser.add_argument(
        "--z_max",
        type=float,
        default=4.5,
        help="Maximum redshift of P1D measurements to be analyzed",
    )
    parser.add_argument(
        "--fid_igm_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set fiducial IGM model",
    )
    parser.add_argument(
        "--true_igm_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set IGM model to create mock",
    )
    parser.add_argument(
        "--n_igm",
        type=int,
        default=2,
        help="Number of free parameters for IGM model",
    )
    parser.add_argument(
        "--n_metals",
        type=int,
        default=0,
        help="Number of free parameters for SiIII metal contamination",
    )
    parser.add_argument(
        "--true_SiIII",
        type=float,
        default=-10,
        help="Metal contamination to create mock",
    )
    parser.add_argument(
        "--fid_SiIII",
        type=float,
        default=-10,
        help="Metal contamination to set fiducial",
    )
    parser.add_argument(
        "--true_SiII",
        type=float,
        default=-10,
        help="Metal contamination to create mock",
    )
    parser.add_argument(
        "--fid_SiII",
        type=float,
        default=-10,
        help="Metal contamination to set fiducial",
    )
    parser.add_argument(
        "--true_HCD",
        type=float,
        default=-10,
        help="HCD contamination to create mock",
    )
    parser.add_argument(
        "--fid_HCD",
        type=float,
        default=-6,
        help="HCD contamination to set fiducial",
    )

    parser.add_argument(
        "--n_dla",
        type=int,
        default=0,
        help="Number of free parameters for DLA contamination",
    )

    parser.add_argument(
        "--fid_cosmo_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set fiducial cosmology",
    )
    parser.add_argument(
        "--true_cosmo_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set true cosmology for mock",
    )

    parser.add_argument(
        "--drop_sim",
        action="store_true",
        help="Drop data_label simulation from the training set",
    )

    # P1D
    parser.add_argument(
        "--apply_smoothing",
        default=None,
        required=False,
        help="Apply smoothing to data, None for whatever is best for selected emulator",
    )

    # likelihood
    parser.add_argument(
        "--cov_label",
        type=str,
        default="Chabanier2019",
        choices=["Chabanier2019", "QMLE_Ohio"],
        required=False,
        help="Data covariance",
    )
    parser.add_argument(
        "--cov_label_hires",
        type=str,
        default="Karacayli2022",
        choices=["Karacayli2022"],
        required=False,
        help="Data covariance for high-res data",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Add noise to P1D mock according to covariance matrix",
    )
    parser.add_argument(
        "--seed_noise",
        type=int,
        default=0,
        help="Seed for noise",
    )
    parser.add_argument(
        "--fix_cosmo",
        action="store_true",
        help="Fix cosmological parameters while sampling",
    )
    parser.add_argument(
        "--vary_alphas",
        action="store_true",
        help="Fit running power spectrum",
    )

    parser.add_argument(
        "--version",
        default="v3",
        help="Version of the pipeline",
    )

    parser.add_argument(
        "--prior_Gauss_rms",
        default=None,
        help="Width of Gaussian prior",
    )
    parser.add_argument(
        "--emu_cov_factor",
        type=float,
        default=0,
        help="scale contribution of emulator covariance",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print information",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test job",
    )

    parser.add_argument(
        "--explore",
        action="store_true",
        help="Save all chains",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Parallelize",
    )

    parser.add_argument(
        "--n_burn_in",
        type=int,
        default=0,
        help="For emcee, n_burn_in",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=0,
        help="For emcee, n_steps",
    )

    #######################
    # print args

    args = parser.parse_args()
    mpi_hello_world()

    fprint = create_print_function(verbose=args.verbose)
    fprint("--- print options from parser ---")
    fprint(args)
    fprint("----------")
    fprint(parser.format_values())
    fprint("----------")

    args.archive = None
    dict_training_set = {
        "Pedersen21": "Pedersen21",
        "Pedersen21_ext": "Cabayol23",
        "Pedersen21_ext8": "Cabayol23",
        "Pedersen23": "Pedersen21",
        "Pedersen23_ext": "Cabayol23",
        "Pedersen23_ext8": "Cabayol23",
        "CH24": "Cabayol23",
        "Cabayol23": "Cabayol23",
        "Cabayol23_extended": "Cabayol23",
        "Cabayol23+": "Cabayol23",
        "Cabayol23+_extended": "Cabayol23",
        "Nyx_v0": "Nyx23_Oct2023",
        # "Nyx_v0_extended": "Nyx23_Oct2023",
        "Nyx_alphap": "Nyx23_Oct2023",
    }
    dict_apply_smoothing = {
        "Pedersen21": False,
        "Pedersen21_ext": False,
        "Pedersen21_ext8": False,
        "Pedersen23": True,
        "Pedersen23_ext": True,
        "Pedersen23_ext8": True,
        "CH24": True,
        "Cabayol23": True,
        "Cabayol23_extended": True,
        "Cabayol23+": True,
        "Cabayol23+_extended": True,
        "Nyx_v0": True,
        # "Nyx_v0_extended": True,
        "Nyx_alphap": True,
    }

    args.training_set = dict_training_set[args.emulator_label]
    if args.apply_smoothing is None:
        args.apply_smoothing = dict_apply_smoothing[args.emulator_label]
    else:
        if args.apply_smoothing == "True":
            args.apply_smoothing = True
        else:
            args.apply_smoothing = False

    if args.test:
        args.explore = True

    return args


# from dataclasses import dataclass, field
# from typing import List

# @dataclass
# class Parameter:
#     name: str = field(default="hello", metadata={"possible_values": ["hello", "world", "parameter"]})

#     def is_valid(self) -> bool:
#         """
#         Check if the current value of `name` is within the allowed possible values.
#         """
#         possible_values = self.__dataclass_fields__["name"].metadata["possible_values"]
#         return self.name in possible_values


@dataclass
class Args:
    """
    Class to store input arguments
    """

    archive: str | None = None
    emulator: str | None = None
    training_set: str = "Pedersen21"
    nyx_training_set: str = "models_Nyx_Mar2025_with_CGAN_val_3axes"
    emulator_label: str = "CH24_nyxcen_gpr"
    data_label: str = "mpg_central"
    p1d_fname: str | None = None
    cov_syst_type: str = "red"
    data_label_hires: str | None = None
    z_min: float = 0
    z_max: float = 10
    fid_label_mF: str = "mpg_central"
    fid_label_T: str = "mpg_central"
    fid_label_kF: str = "mpg_central"
    true_label_mF: str = "mpg_central"
    true_label_T: str = "mpg_central"
    true_label_kF: str = "mpg_central"
    n_tau: int = 2
    n_sigT: int = 2
    n_gamma: int = 2
    n_kF: int = 2
    n_d_dla: int = 0
    n_s_dla: int = 0
    n_sn: int = 0
    n_agn: int = 0
    n_res: int = 0
    ic_correction: bool = False
    igm_priors: str = "hc"
    fid_cosmo_label: str = "mpg_central"
    true_cosmo_label: str | None = None
    fid_val_mF: list[float] = field(default_factory=lambda: [0, 0])
    true_val_mF: list[float] = field(default_factory=lambda: [0, 0])
    fid_val_sigT: list[float] = field(default_factory=lambda: [0, 0])
    true_val_sigT: list[float] = field(default_factory=lambda: [0, 0])
    fid_val_gamma: list[float] = field(default_factory=lambda: [0, 0])
    true_val_gamma: list[float] = field(default_factory=lambda: [0, 0])
    fid_val_kF: list[float] = field(default_factory=lambda: [0, 0])
    true_val_kF: list[float] = field(default_factory=lambda: [0, 0])
    fid_A_damp: list[float] = field(default_factory=lambda: [0, -9])
    true_A_damp: list[float] = field(default_factory=lambda: [0, -9])
    fid_A_scale: list[float] = field(default_factory=lambda: [0, 5])
    true_A_scale: list[float] = field(default_factory=lambda: [0, 5])
    fid_SN: list[float] = field(default_factory=lambda: [0, -4])
    true_SN: list[float] = field(default_factory=lambda: [0, -4])
    fid_AGN: list[float] = field(default_factory=lambda: [0, -5])
    true_AGN: list[float] = field(default_factory=lambda: [0, -5])
    fid_R_coeff: list[float] = field(default_factory=lambda: [0, 0])
    true_R_coeff: list[float] = field(default_factory=lambda: [0, 0])
    drop_sim: bool = False
    apply_smoothing: bool = False
    cov_label: str = "Chabanier2019"
    cov_label_hires: str = "Karacayli2022"
    hcd_model_type: str = "Rogers2017"
    mF_model_type: str = "pivot"
    use_star_priors: Optional[dict] = None
    add_noise: bool = False
    seed_noise: int = 0
    fix_cosmo: bool = False
    vary_alphas: bool = False
    prior_Gauss_rms: float | None = None
    emu_cov_factor: int | None = 1
    cov_factor: int = 1
    emu_cov_type: str = "full"
    verbose: bool = True
    test: bool = False
    explore: bool = False
    parallel: bool = True
    n_burn_in: int = 0
    n_steps: int = 0
    z_star: float = 3
    kp_kms: float = 0.009
    fid_igm: dict = field(default_factory=lambda: {})
    fid_cont: dict = field(default_factory=lambda: {})
    true_cont: dict = field(default_factory=lambda: {})
    true_igm: dict = field(default_factory=lambda: {})
    fid_syst: dict = field(default_factory=lambda: {})
    true_syst: dict = field(default_factory=lambda: {})
    n_metals: dict = field(default_factory=lambda: {})
    Gauss_priors: dict | None = None
    metal_lines: list[str] = field(
        default_factory=lambda: [
            "Lya_SiIII",
            "Lya_SiIIa",
            "Lya_SiIIb",
            "SiIIa_SiIIb",
            "SiIIa_SiIII",
            "SiIIb_SiIII",
        ]
    )

    def __post_init__(self):
        # Setting up fiducial and true values of IGM parameters
        igms = [self.fid_igm, self.true_igm]
        for ii in range(2):
            igms[ii]["priors"] = "hc"
            igms[ii]["label_mF"] = "mpg_central"
            igms[ii]["label_T"] = "mpg_central"
            igms[ii]["label_kF"] = "mpg_central"
            igms[ii]["n_tau"] = 1
            igms[ii]["n_sigT"] = 1
            igms[ii]["n_gamma"] = 1
            igms[ii]["n_kF"] = 1

        # Setting up fiducial and true values of nuisance parameters
        conts = [self.fid_cont, self.true_cont]
        for ii in range(2):
            # each metal line
            for metal_line in self.metal_lines:
                conts[ii]["f_" + metal_line] = [0, -11.5]
                conts[ii]["s_" + metal_line] = [0, -9]
                conts[ii]["p_" + metal_line] = [0, 1]
                conts[ii]["n_f_" + metal_line] = 0
                conts[ii]["n_s_" + metal_line] = 0
                conts[ii]["n_p_" + metal_line] = 0

            # same for dlas
            conts[ii]["hcd_model_type"] = "new"
            conts[ii]["n_d_dla1"] = 0
            conts[ii]["HCD_damp1"] = [0, 0]

            conts[ii]["n_s_dla1"] = 0
            conts[ii]["HCD_scale1"] = [0, 0]

            conts[ii]["n_d_dla2"] = 0
            conts[ii]["HCD_damp2"] = [0, 0]

            conts[ii]["n_s_dla2"] = 0
            conts[ii]["HCD_scale2"] = [0, 0]

            conts[ii]["n_c_dla"] = 0
            conts[ii]["HCD_const"] = [0, 0]

            # and others
            conts[ii]["n_sn"] = 0
            conts[ii]["SN"] = [0, -4]

            conts[ii]["n_agn"] = 0
            conts[ii]["AGN"] = [0, -5.5]

        # Setting up fiducial and true values of systematic parameters
        systs = [self.fid_syst, self.true_syst]
        for ii in range(2):
            systs[ii]["res_model_type"] = "pivot"
            systs[ii]["n_res"] = 0
            systs[ii]["R_coeff"] = [0, 0]

    def check_emulator_label(self):
        avail_emulator_label = [
            "Pedersen21",
            "Pedersen21_ext",
            "Pedersen21_ext8",
            "Pedersen23",
            "Pedersen23_ext",
            "Pedersen23_ext8",
            "Cabayol23",
            "Cabayol23_extended",
            "Cabayol23+",
            "Cabayol23+_extended",
            "Nyx_v0",
            # "Nyx_v0_extended",
            "Nyx_alphap",
            "Nyx_alphap_cov",
            "CH24",
            "CH24_Nyx",
            "CH24_mpgcen_gpr",
            "CH24_nyxcen_gpr",
        ]
        if self.emulator_label not in avail_emulator_label:
            raise ValueError(
                "emulator_label " + self.emulator_label + " not implemented"
            )

    def set_baseline(
        self,
        fix_cosmo=True,
        fit_type="at_a_time",
        ztar=0,
        zmax=4.2,
        P1D_type="DESIY1_QMLE3",
    ):
        """
        Set baseline parameters
        """
        self.ic_from_file = None
        self.fit_type = fit_type
        if fit_type not in [
            "andreu2",
            "global",
            "at_a_time",
            "at_at_time_igm",
            "at_a_time_orig",
            "wip",
            "andreu",
        ]:
            raise ValueError("fit_type " + fit_type + " not implemented")
        if P1D_type.startswith("DESIY1"):
            self.P1D_type = P1D_type
            self.true_igm_label = None
            self.data_label = "lyacolore"
            self.cov_syst_type = "red"
            self.z_min = 2.1
            self.z_max = 4.3
            self.true_cosmo_label = None

            path_in_challenge = os.path.join(
                os.path.dirname(get_path_repo("cupix")), "data", "in_DESI_DR1"
            )
            path_out_challenge = os.path.join(
                os.path.dirname(get_path_repo("cupix")),
                "data",
                "out_DESI_DR1",
            )
            if P1D_type.endswith("QMLE3"):
                self.p1d_fname = os.path.join(
                    path_in_challenge,
                    "qmle_measurement",
                    "DataProducts",
                    "v3",
                    "desi_y1_snr3_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits",
                )
            elif P1D_type.endswith("QMLE"):
                self.p1d_fname = os.path.join(
                    path_in_challenge,
                    "qmle_measurement",
                    "DataProducts",
                    "v3",
                    "desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits",
                )
            elif P1D_type.endswith("FFT_dir"):
                self.p1d_fname = os.path.join(
                    path_in_challenge,
                    "fft_measurement",
                    "p1d_fft_y1_measurement_kms_v7_direct_metal_subtraction.fits",
                )
            elif P1D_type.endswith("FFT"):
                self.p1d_fname = os.path.join(
                    path_in_challenge,
                    "fft_measurement",
                    "p1d_fft_y1_measurement_kms_v7.fits",
                )
            else:
                raise ValueError(
                    "P1D_type " + P1D_type + " not implemented for DESI_DR1"
                )

            self.ic_from_file = os.path.join(
                path_out_challenge,
                "ic",
                "allz_snr3_nocosmo_" + fit_type,
                "res.npy",
            )

        # set all cont to zero
        cont_props = [
            "n_HCD_damp1",
            "n_HCD_damp2",
            "n_HCD_damp3",
            "n_HCD_damp4",
            "n_HCD_const",
            "n_f_Lya_SiIII",
            "n_s_Lya_SiIII",
            "n_f_Lya_SiII",
            "n_s_Lya_SiII",
            "n_f_SiIIa_SiIIb",
            "n_s_SiIIa_SiIIb",
            "n_f_SiIIa_SiIII",
            "n_f_SiIIb_SiIII",
        ]
        self.fid_cont["HCD_const_otype"] = "const"
        for prop in cont_props:
            self.fid_cont[prop] = 0

        # set igm stuff
        igm_props = ["n_tau", "n_sigT", "n_gamma", "n_kF"]

        self.fid_igm["tau_eff_otype"] = "exp"
        self.fid_igm["gamma_otype"] = "const"
        self.fid_igm["sigT_kms_otype"] = "const"
        self.fid_igm["kF_kms_otype"] = "const"
        for prop in igm_props:
            self.fid_igm[prop] = 0

        self.null_vals = {}

        self.cov_factor = 1
        self.cov_syst_type = "red"
        self.emu_cov_factor = 1
        self.emu_cov_type = "block"
        self.prior_Gauss_rms = None
        self.rebin_k = 8  # good enough to 0.06 chi2
        if fix_cosmo:
            self.fix_cosmo = True
        else:
            self.fix_cosmo = False
        self.vary_alphas = False
        self.ic_correction = False
        self.fid_cosmo_label = "Planck18"

        sim_fid = "mpg_central"
        self.fid_igm["label_mF"] = sim_fid
        self.fid_igm["label_T"] = sim_fid
        self.fid_igm["label_kF"] = sim_fid

        self.fid_cont["hcd_model_type"] = "new_rogers"

        props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]
        props_cont = [
            "f_Lya_SiIII",
            "s_Lya_SiIII",
            "f_Lya_SiII",
            "s_Lya_SiII",
            "f_SiIIa_SiIIb",
            "s_SiIIa_SiIIb",
            "f_SiIIa_SiIII",
            "f_SiIIb_SiIII",
            "HCD_damp1",
            "HCD_damp2",
            "HCD_damp3",
            "HCD_damp4",
        ]

        self.fid_cont["z_max"] = {}

        # z at a time
        if fit_type == "at_a_time":
            baseline_prop = [
                "tau_eff",
                "sigT_kms",
                # "gamma",
                # "kF_kms",
                "f_Lya_SiIII",
                "s_Lya_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
                "f_SiIIa_SiIIb",
                "s_SiIIa_SiIIb",
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "HCD_damp1",
                # "HCD_damp2",
                # "HCD_damp3",
                "HCD_damp4",
                # "HCD_const",
            ]

            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5

            if ztar > 3.5:
                baseline_prop.remove("f_SiIIa_SiIII")
                baseline_prop.remove("f_SiIIb_SiIII")
                self.fid_cont["z_max"]["f_SiIIa_SiIII"] = 3.5
                self.fid_cont["z_max"]["f_SiIIb_SiIII"] = 3.5
                baseline_prop.remove("f_Lya_SiII")
                baseline_prop.remove("s_Lya_SiII")
                self.fid_cont["z_max"]["f_Lya_SiII"] = 3.5
                self.fid_cont["z_max"]["s_Lya_SiII"] = 3.5

            if ztar > 3.7:
                baseline_prop.remove("f_SiIIa_SiIIb")
                baseline_prop.remove("s_SiIIa_SiIIb")
                self.fid_cont["z_max"]["f_SiIIa_SiIIb"] = 3.7
                self.fid_cont["z_max"]["s_SiIIa_SiIIb"] = 3.7

            print("baseline_prop", baseline_prop)

            self.fid_syst["res_model_type"] = "pivot"
            self.fid_syst["n_res"] = 0
            self.fid_syst["R_coeff"] = [0, 0]

            props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]
            props_cont = [
                "f_Lya_SiIII",
                "s_Lya_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
                "f_SiIIa_SiIIb",
                "s_SiIIa_SiIIb",
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "HCD_damp1",
                "HCD_damp2",
                # "HCD_damp3",
                "HCD_damp4",
                # "HCD_const",
            ]
            # self.fid_igm["tau_eff_znodes"] = []
            # self.fid_igm["sigT_kms_znodes"] = []
            # self.fid_igm["gamma_znodes"] = []
            # self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                if prop in baseline_prop:
                    self.fid_igm["n_" + prop] = 1
                else:
                    self.fid_igm["n_" + prop] = 0
                self.fid_igm[prop + "_ztype"] = "pivot"

            for prop in props_cont:
                if prop in baseline_prop:
                    self.fid_cont["n_" + prop] = 1
                else:
                    self.fid_cont["n_" + prop] = 0
                self.fid_cont[prop + "_ztype"] = "pivot"
                if prop != "HCD_const":
                    self.fid_cont[prop] = [0, -11.5]
                else:
                    self.fid_cont[prop] = [0, 0]

            print(self.fid_cont["n_HCD_const"])

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

        elif fit_type == "at_a_time_igm":
            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5

            for prop in props_igm:
                self.fid_igm["n_" + prop] = 1
                self.fid_igm[prop + "_ztype"] = "pivot"

            for prop in props_cont:
                self.fid_cont["n_" + prop] = 0
                self.fid_cont[prop + "_ztype"] = "pivot"
                if prop != "HCD_const":
                    self.fid_cont[prop] = [0, -11.5]
                else:
                    self.fid_cont[prop] = [0, 0]

            self.fid_syst["res_model_type"] = "pivot"
            self.fid_syst["n_res"] = 0
            self.fid_syst["R_coeff"] = [0, 0]

        elif fit_type == "at_a_time_orig":
            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5

            for prop in props_igm:
                self.fid_igm["n_" + prop] = 1
                self.fid_igm[prop + "_ztype"] = "pivot"

            for prop in props_cont:
                self.fid_cont["n_" + prop] = 1
                self.fid_cont[prop + "_ztype"] = "pivot"
                if prop != "HCD_const":
                    self.fid_cont[prop] = [0, -11.5]
                else:
                    self.fid_cont[prop] = [0, 0]

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

            self.fid_syst["res_model_type"] = "pivot"
            self.fid_syst["n_res"] = 1
            self.fid_syst["R_coeff"] = [0, 0]

        elif fit_type == "wip":
            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5
            props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]
            props_cont = [
                "f_Lya_SiIII",
                "s_Lya_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
                "f_SiIIa_SiIIb",
                "s_SiIIa_SiIIb",
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "HCD_damp1",
                "HCD_damp4",
            ]

            z_max_props = [
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
            ]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.5
            z_max_props = ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.7

            self.opt_props = props_igm + props_cont

            self.fid_igm["tau_eff_znodes"] = np.linspace(2.2, 4.2, 8)
            self.fid_igm["sigT_kms_znodes"] = np.linspace(2.2, 4.2, 10)
            self.fid_igm["gamma_znodes"] = []
            self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_spl"

            # do not gain anything by using splines?
            # self.fid_igm["n_tau_eff"] = 3
            # self.fid_igm["n_sigT_kms"] = 3  # what about 2?
            # self.fid_igm["n_gamma"] = 0
            # self.fid_igm["n_kF_kms"] = 0

            # znodes:
            # tau: 2.2, 2.4, 3.2, 3.6, 3.8, 4.2
            # sigT: 2.2, 2.6, 4.2
            # fLya-SiIII: 2.2, 2.8, 3.2, 4.2 as a function of tau-prior? check
            # sLya-SiIII: 2.2, 4.2
            # fLya-SiII: 2.2, 2.6, 3.4
            # sLya-SiII: 2.2, 2.6, 3.4
            # fSiIIa-SiIIb: 2.2, 3.6
            # sSiIIa-SiIIb: 2.2, 2.8, 3.6
            # fSiIIa-SiIII: 2.2, 3.4
            # fSiIIb-SiIII: 2.2, 3.4
            # HCD-damp1: All?
            # HCD-damp4: 2.2, 3.2, 4.2

            # 2, 3, 4, 6, 5, 41 (3.2), 4, 10 (3.6), 10 (3.8), 8 (4.0), 4

            for prop in props_cont:
                if prop in self.opt_props:
                    _znodes = 0

                if prop in ["f_SiIIa_SiIII", "f_SiIIb_SiIII"]:
                    _znodes = np.linspace(2.2, self.fid_cont["z_max"][prop], 2)

                if prop in ["f_Lya_SiII"]:
                    _znodes = np.linspace(2.2, self.fid_cont["z_max"][prop], 3)

                if prop in ["s_Lya_SiII"]:
                    _znodes = np.linspace(2.2, self.fid_cont["z_max"][prop], 5)

                if prop in ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]:
                    _znodes = np.linspace(2.2, self.fid_cont["z_max"][prop], 6)

                if prop in ["f_Lya_SiIII", "s_Lya_SiIII"]:
                    _znodes = np.linspace(2.2, 4.2, 3)

                if prop in ["HCD_damp1"]:
                    _znodes = np.linspace(2.2, 4.2, 11)

                if prop in ["HCD_damp4"]:
                    _znodes = np.linspace(2.2, 4.2, 8)

                self.fid_cont[prop + "_znodes"] = _znodes
                self.fid_cont["n_" + prop] = len(
                    self.fid_cont[prop + "_znodes"]
                )
                if self.fid_cont["n_" + prop] <= 1:
                    self.fid_cont[prop + "_ztype"] = "pivot"
                    if prop != "HCD_const":
                        self.fid_cont[prop] = [0, -11.5]
                    else:
                        self.fid_cont[prop] = [0, 0]
                else:
                    self.fid_cont[prop + "_ztype"] = "interp_spl"

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

        elif fit_type == "global":
            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5
            props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]
            props_cont = [
                "f_Lya_SiIII",
                "s_Lya_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
                "f_SiIIa_SiIIb",
                "s_SiIIa_SiIIb",
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "HCD_damp1",
                "HCD_damp4",
            ]

            z_max_props = [
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
            ]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.5
            z_max_props = ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.3
            z_max_props = ["f_Lya_SiIII", "s_Lya_SiIII"]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.9

            self.opt_props = props_igm + props_cont

            self.fid_igm["tau_eff_znodes"] = np.array(
                [2.2, 2.4, 2.6, 2.8, 3.2, 3.6, 3.8, 4.2]
            )
            self.fid_igm["sigT_kms_znodes"] = np.array([2.2, 4.2])
            self.fid_igm["gamma_znodes"] = np.array([2.2, 2.6, 4.2])
            self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_spl"

            # do not gain anything by using splines?
            # self.fid_igm["n_tau_eff"] = 3
            # self.fid_igm["n_sigT_kms"] = 3  # what about 2?
            # self.fid_igm["n_gamma"] = 0
            self.fid_igm["n_kF_kms"] = 0

            # znodes:
            # tau: 2.2, 2.4, 2.6, 3.2, 3.6, 3.8, 4.2
            # sigT: 2.2, 2.6, 4.2
            # fLya-SiIII: 2.2, 2.8, 3.2, 4.2 as a function of tau-prior? check
            # sLya-SiIII: 2.2, 4.2
            # fLya-SiII: 2.2, 2.6, 3.4
            # sLya-SiII: 2.2, 2.6, 3.4
            # fSiIIa-SiIIb: 2.2, 3.6
            # sSiIIa-SiIIb: 2.2, 2.8, 3.6
            # fSiIIa-SiIII: 2.2, 3.4
            # fSiIIb-SiIII: 2.2, 3.4
            # HCD-damp1: All?
            # HCD-damp4: 2.2, 3.2, 4.2

            # Lya-SiII: 2.2, 2.4, 2.8, DONE
            # SiII-SiII: 3.2 DONE
            # DLA: 3.4, 3.6, 3.8 DONE
            # Lya-SiIII: 3.6, 3.8 DONE

            for prop in props_cont:
                if prop in self.opt_props:
                    _znodes = 0

                if prop in ["f_Lya_SiIII", "s_Lya_SiIII"]:
                    _znodes = np.array([2.2, 2.8, 3.4, 3.6, 3.8])

                if prop in ["f_SiIIa_SiIII", "f_SiIIb_SiIII"]:
                    _znodes = np.array([2.2, 3.4])

                if prop in ["f_Lya_SiII"]:
                    _znodes = np.array([2.2, 2.4, 2.6, 2.8, 3.4])
                if prop in ["s_Lya_SiII"]:
                    _znodes = np.array([2.2, 2.4, 2.8, 3.4])

                if prop in ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]:
                    _znodes = np.array([2.2, 2.8, 3.2])

                if prop in ["HCD_damp1"]:
                    _znodes = np.array([2.2, 2.8, 3.4, 3.6, 3.9, 4.2])

                if prop in ["HCD_damp4"]:
                    _znodes = np.array([2.2, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])

                self.fid_cont[prop + "_znodes"] = _znodes
                self.fid_cont["n_" + prop] = len(
                    self.fid_cont[prop + "_znodes"]
                )
                if self.fid_cont["n_" + prop] <= 1:
                    self.fid_cont[prop + "_ztype"] = "pivot"
                    if prop != "HCD_const":
                        self.fid_cont[prop] = [0, -11.5]
                    else:
                        self.fid_cont[prop] = [0, 0]
                else:
                    self.fid_cont[prop + "_ztype"] = "interp_spl"

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

        elif fit_type == "andreu":
            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5
            props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]
            props_cont = [
                "f_Lya_SiIII",
                "s_Lya_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
                "f_SiIIa_SiIIb",
                "s_SiIIa_SiIIb",
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "HCD_damp1",
                "HCD_damp4",
            ]

            z_max_props = [
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
            ]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.5
            z_max_props = ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.7

            self.opt_props = props_igm + props_cont

            self.fid_igm["tau_eff_znodes"] = np.linspace(2.2, 4.2, 11)
            self.fid_igm["sigT_kms_znodes"] = np.linspace(2.2, 4.2, 6)
            self.fid_igm["gamma_znodes"] = []
            self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_spl"

            # do not gain anything by using splines?
            # self.fid_igm["n_tau_eff"] = 3
            # self.fid_igm["n_sigT_kms"] = 3  # what about 2?
            # self.fid_igm["n_gamma"] = 0
            # self.fid_igm["n_kF_kms"] = 0

            # znodes = [0, 1]

            for prop in props_cont:
                if prop in self.opt_props:
                    _znodes = 0

                if prop in [
                    "f_SiIIa_SiIII",
                    "f_SiIIb_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                ]:
                    _znodes = np.linspace(2.2, self.fid_cont["z_max"][prop], 3)

                if prop in ["f_Lya_SiIII", "s_Lya_SiIII"]:
                    _znodes = np.linspace(2.2, 4.2, 6)

                if prop in ["HCD_damp1", "HCD_damp4"]:
                    _znodes = np.linspace(2.2, 4.2, 6)

                self.fid_cont[prop + "_znodes"] = _znodes
                self.fid_cont["n_" + prop] = len(
                    self.fid_cont[prop + "_znodes"]
                )
                if self.fid_cont["n_" + prop] <= 1:
                    self.fid_cont[prop + "_ztype"] = "pivot"
                    if prop != "HCD_const":
                        self.fid_cont[prop] = [0, -11.5]
                    else:
                        self.fid_cont[prop] = [0, 0]
                else:
                    self.fid_cont[prop + "_ztype"] = "interp_spl"

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

        elif fit_type == "andreu2":
            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5
            props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]
            props_cont = [
                "f_Lya_SiIII",
                "s_Lya_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
                "f_SiIIa_SiIIb",
                "s_SiIIa_SiIIb",
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "HCD_damp1",
                "HCD_damp4",
            ]

            z_max_props = [
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
            ]
            # for prop in z_max_props:
            #     self.fid_cont["z_max"][prop] = 3.5
            # z_max_props = ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]
            # for prop in z_max_props:
            #     self.fid_cont["z_max"][prop] = 3.3
            # z_max_props = ["f_Lya_SiIII", "s_Lya_SiIII"]
            # for prop in z_max_props:
            #     self.fid_cont["z_max"][prop] = 3.9

            self.opt_props = props_igm + props_cont

            self.fid_igm["tau_eff_znodes"] = np.linspace(2.2, 4.2, 6)
            self.fid_igm["sigT_kms_znodes"] = np.array([2.2, 3.0, 4.2])
            self.fid_igm["gamma_znodes"] = np.array([2.2, 3.0, 4.2])
            self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_spl"

            # do not gain anything by using splines?
            # self.fid_igm["n_tau_eff"] = 3
            # self.fid_igm["n_sigT_kms"] = 3  # what about 2?
            # self.fid_igm["n_gamma"] = 0
            self.fid_igm["n_kF_kms"] = 0

            # znodes = [0, 1]

            for prop in props_cont:
                if prop in self.opt_props:
                    _znodes = 0

                if prop in [
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                ]:
                    _znodes = np.linspace(2.2, 3.6, 3)

                if prop in ["f_SiIIa_SiIII", "f_SiIIb_SiIII"]:
                    _znodes = np.array([2.2, 3.6])

                if prop in ["HCD_damp1", "HCD_damp4"]:
                    _znodes = np.linspace(2.2, 4.2, 4)

                self.fid_cont[prop + "_znodes"] = _znodes
                self.fid_cont["n_" + prop] = len(
                    self.fid_cont[prop + "_znodes"]
                )
                if self.fid_cont["n_" + prop] <= 1:
                    self.fid_cont[prop + "_ztype"] = "pivot"
                    if prop != "HCD_const":
                        self.fid_cont[prop] = [0, -11.5]
                    else:
                        self.fid_cont[prop] = [0, 0]
                else:
                    self.fid_cont[prop + "_ztype"] = "interp_spl"

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

        else:
            self.fid_igm["tau_eff_ztype"] = "interp_spl"
            self.fid_igm["sigT_kms_ztype"] = "interp_spl"
            self.fid_igm["gamma_ztype"] = "pivot"
            self.fid_igm["kF_kms_ztype"] = "pivot"
            self.fid_igm["tau_eff_znodes"] = np.arange(2.2, 4.4, 0.2)
            self.fid_igm["sigT_kms_znodes"] = np.linspace(2.2, 4.2, 6)
            # self.fid_igm["gamma_znodes"] = np.linspace(2.2, 4.2, 4)
            # self.fid_igm["kF_kms_znodes"] = np.linspace(2.2, 4.2, 4)
            if self.fid_igm["tau_eff_ztype"].startswith("interp"):
                self.fid_igm["n_tau"] = len(self.fid_igm["tau_eff_znodes"])
            else:
                self.fid_igm["n_tau"] = 1

            if self.fid_igm["sigT_kms_ztype"].startswith("interp"):
                self.fid_igm["n_sigT"] = len(self.fid_igm["sigT_kms_znodes"])
            else:
                self.fid_igm["n_sigT"] = 1

            if self.fid_igm["gamma_ztype"].startswith("interp"):
                self.fid_igm["n_gamma"] = len(self.fid_igm["gamma_znodes"])
            else:
                self.fid_igm["n_gamma"] = 1

            if self.fid_igm["kF_kms_ztype"].startswith("interp"):
                self.fid_igm["n_kF"] = len(self.fid_igm["kF_kms_znodes"])
            else:
                self.fid_igm["n_kF"] = 1
            self.fid_igm["n_gamma"] = 0
            self.fid_igm["n_kF"] = 0

            self.fid_syst["res_model_type"] = "pivot"
            self.fid_syst["n_res"] = 0
            self.fid_syst["R_coeff"] = [0, 0]

            # model like lya-metal
            self.fid_cont["n_f_Lya_SiIII"] = 0
            self.fid_cont["n_s_Lya_SiIII"] = 0

            self.fid_cont["n_f_Lya_SiII"] = 0
            self.fid_cont["n_s_Lya_SiII"] = 0

            self.fid_cont["n_f_SiIIa_SiIIb"] = 0
            self.fid_cont["n_s_SiIIa_SiIIb"] = 0

            self.fid_cont["n_f_SiIIa_SiIII"] = 0
            self.fid_cont["n_f_SiIIb_SiIII"] = 0

            self.fid_cont["n_f_CIVa_CIVb"] = 0
            self.fid_cont["n_s_CIVa_CIVb"] = 0

            self.fid_cont["n_f_MgIIa_MgIIb"] = 0
            self.fid_cont["n_s_MgIIa_MgIIb"] = 0

            self.fid_cont["n_d_dla1"] = 0
            self.fid_cont["n_d_dla2"] = 0
            self.fid_cont["n_d_dla3"] = 0
            self.fid_cont["n_d_dla4"] = 0
            self.fid_cont["n_c_dla"] = 0

        fid_vals_metals = {
            "f_Lya_SiIII": -4.25,
            "f_Lya_SiII": -4.5,
            "f_SiIIa_SiIIb": -0.5,
            "f_SiIIa_SiIII": 1,
            "f_SiIIb_SiIII": 1,
        }
        add_lines = [
            "SiIIa_SiIIb",
            "CIVa_CIVb",
            "MgIIa_MgIIb",
        ]

        for metal_label in self.metal_lines:
            if self.fid_cont["n_f_" + metal_label] == 0:
                self.fid_cont["f_" + metal_label] = [0, -11.5]
            else:
                self.fid_cont["f_" + metal_label] = [
                    0,
                    fid_vals_metals["f_" + metal_label],
                ]

            if self.fid_cont["n_s_" + metal_label] == 0:
                self.fid_cont["s_" + metal_label] = [0, -11.5]
            else:
                self.fid_cont["s_" + metal_label] = [0, 4.75]

            # priors
            self.fid_cont["flat_priors"] = {}
            if metal_label not in add_lines:
                self.fid_cont["flat_priors"]["s_" + metal_label] = [
                    [-2, 2],
                    [-11, 8],
                ]

            self.fid_cont["flat_priors"]["f_Lya_SiIII"] = [
                [-1, 1],
                [-11, -2.5],
            ]
            self.fid_cont["flat_priors"]["s_Lya_SiIII"] = [
                [-1, 1],
                [2, 5.75],
            ]

            self.fid_cont["flat_priors"]["f_Lya_SiII"] = [
                [-1, 1],
                [-11, -3],
            ]
            self.fid_cont["flat_priors"]["s_Lya_SiII"] = [
                [-1, 1],
                [2, 6.25],
            ]

            self.fid_cont["flat_priors"]["f_SiIIa_SiIIb"] = [
                [-1, 4],
                [-11, 3],
            ]
            self.fid_cont["flat_priors"]["s_SiIIa_SiIIb"] = [
                [-1, 4],
                [2, 6],
            ]

            self.fid_cont["flat_priors"]["f_SiIIa_SiIII"] = [
                [-1, 2],
                [-1, 4],
            ]
            self.fid_cont["flat_priors"]["f_SiIIb_SiIII"] = [
                [-1, 1],
                [-1, 3],
            ]

        # priors
        self.fid_cont["flat_priors"]["HCD_damp"] = [[-0.5, 0.5], [-11.5, -0.05]]
        self.fid_cont["flat_priors"]["HCD_const"] = [[-1, 1], [-0.2, 1e-6]]


# Set Gaussian priors
# # self.prior_Gauss_rms = 0.1
# self.prior_Gauss_rms = None

# self.Gauss_priors = {}
# self.Gauss_priors["ln_tau_0"] = [10]
# # self.Gauss_priors["ln_sigT_kms_0"] = [0.02]
# # self.Gauss_priors["ln_gamma_0"] = [0.08]
# # self.Gauss_priors["ln_kF_0"] = [0.003]

# f_Gprior = {
#     "Lya_SiIII": 1,
#     "Lya_SiIIa": 1,
#     "Lya_SiIIb": 1,
#     "SiIIa_SiIIb": 3,
#     "SiIIa_SiIII": 4,
#     "SiIIb_SiIII": 3,
# }

# d_Gprior = {
#     "Lya_SiIII": 1.5,
#     "Lya_SiIIa": 0.05,
#     "Lya_SiIIb": 0.05,
#     "SiIIa_SiIIb": 1,
#     "SiIIa_SiIII": 0.03,
#     "SiIIb_SiIII": 1,
# }

# a_Gprior = {
#     "Lya_SiIII": 10,
#     "Lya_SiIIa": 10,
#     "Lya_SiIIb": 10,
#     "SiIIa_SiIIb": 2,
#     "SiIIa_SiIII": 0.05,
#     "SiIIb_SiIII": 0.01,
# }

# for metal_line in lines_use:
#     self.Gauss_priors["ln_x_"+metal_line+"_0"] = [f_Gprior[metal_line]]
#     self.Gauss_priors["d_"+metal_line+"_0"] = [d_Gprior[metal_line]]
#     self.Gauss_priors["a_"+metal_line+"_0"] = [a_Gprior[metal_line]]
# self.Gauss_priors["ln_A_damp_0"] = [0.3]
# self.Gauss_priors["ln_A_scale_0"] = [1]
# self.Gauss_priors["R_coeff_0"] = [2]

# self.Gauss_priors = {}
