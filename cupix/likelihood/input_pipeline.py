import sys, os, configargparse
from dataclasses import dataclass, field
from typing import Optional

from cup1d.utils.utils import create_print_function, mpi_hello_world


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
    fid_metals: dict = field(default_factory=lambda: {})
    true_metals: dict = field(default_factory=lambda: {})
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
        for metal_line in self.metal_lines:
            # setting fiducial and true values of parameters for each metal line
            self.fid_metals[metal_line + "_X"] = [0, -10.5]
            self.true_metals[metal_line + "_X"] = [0, -10.5]
            self.fid_metals[metal_line + "_D"] = [0, 5]
            self.true_metals[metal_line + "_D"] = [0, 5]
            self.fid_metals[metal_line + "_L"] = [0, 0]
            self.true_metals[metal_line + "_L"] = [0, 0]
            self.fid_metals[metal_line + "_A"] = [0, 0]
            self.true_metals[metal_line + "_A"] = [0, 0]

            # setting number of X, D, L, A parameters for each metal line
            self.n_metals["n_x_" + metal_line] = 0
            self.n_metals["n_d_" + metal_line] = 0
            self.n_metals["n_l_" + metal_line] = 0
            self.n_metals["n_a_" + metal_line] = 0

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

    def set_baseline(self):
        self.emu_cov_factor = 1
        self.emu_cov_type = "block"
        self.rebin_k = 6
        self.cov_factor = 1
        self.fix_cosmo = True
        self.vary_alphas = False
        self.ic_correction = False
        self.fid_cosmo_label = "Planck18"
        sim_fid = "mpg_central"
        self.fid_label_mF = sim_fid
        self.fid_label_T = sim_fid
        self.fid_label_kF = sim_fid

        # z at a time
        self.mF_model_type = "pivot"
        self.hcd_model_type = "new"
        self.resolution_model_type = "pivot"

        self.n_tau = 1
        self.n_gamma = 1
        self.n_sigT = 1
        self.n_kF = 1
        self.n_res = 1

        lines_use = [
            "Lya_SiIII",
            "Lya_SiIIa",
            "Lya_SiIIb",
            "SiIIa_SiIIb",
            "SiIIa_SiIII",
            "SiIIb_SiIII",
        ]

        f_prior = {
            "Lya_SiIII": -4.5,
            "Lya_SiIIa": -9.5,
            "Lya_SiIIb": -4.5,
            "SiIIa_SiIIb": -5.5,
            "SiIIa_SiIII": -6.0,
            "SiIIb_SiIII": -6.5,
        }

        # at a time
        d_prior = {
            "Lya_SiIII": 0.4,
            "Lya_SiIIa": -0.9,
            "Lya_SiIIb": -0.9,
            "SiIIa_SiIIb": 0.8,
            "SiIIa_SiIII": 1.6,
            "SiIIb_SiIII": 2.7,
        }

        # at a time
        a_prior = {
            "Lya_SiIII": 1.5,
            "Lya_SiIIa": 4.0,
            "Lya_SiIIb": 4.0,
            "SiIIa_SiIIb": 4.0,
            "SiIIa_SiIII": 0.5,
            "SiIIb_SiIII": 2.5,
        }

        # n_metals = {
        #     "Lya_SiIII": [1, 1, 1],
        #     "Lya_SiIIa": [0, 0, 0],
        #     "Lya_SiIIb": [1, 0, 1],
        #     "SiIIa_SiIII": [1, 0, 0],
        #     "SiIIb_SiIII": [1, 1, 0],
        #     "SiIIa_SiIIb": [1, 1, 1],
        # }
        n_metals = {
            "Lya_SiIII": [1, 0, 1],
            "Lya_SiIIa": [0, 0, 0],
            "Lya_SiIIb": [1, 0, 1],
            "SiIIa_SiIII": [1, 0, 1],
            "SiIIb_SiIII": [1, 0, 1],
            "SiIIa_SiIIb": [1, 0, 1],
        }
        # nf, nd, na
        # n_metals = {
        #     "Lya_SiIII": [1, 0, 1],
        #     "Lya_SiIIa": [0, 0, 0],
        #     "Lya_SiIIb": [1, 0, 1],
        #     "SiIIa_SiIII": [0, 0, 0],
        #     "SiIIb_SiIII": [0, 0, 0],
        #     "SiIIa_SiIIb": [1, 0, 1],
        # }

        for metal_line in lines_use:
            self.n_metals["n_x_" + metal_line] = n_metals[metal_line][0]
            if self.n_metals["n_x_" + metal_line] == 0:
                self.fid_metals[metal_line + "_X"] = [0, -10.5]
            else:
                self.fid_metals[metal_line + "_X"] = [0, f_prior[metal_line]]

            self.n_metals["n_d_" + metal_line] = n_metals[metal_line][1]
            if self.n_metals["n_d_" + metal_line] == 0:
                self.fid_metals[metal_line + "_D"] = [0, 0]
            else:
                self.fid_metals[metal_line + "_D"] = [0, d_prior[metal_line]]

            self.n_metals["n_a_" + metal_line] = n_metals[metal_line][2]
            if self.n_metals["n_a_" + metal_line] == 0:
                self.fid_metals[metal_line + "_A"] = [0, 1]
            else:
                self.fid_metals[metal_line + "_A"] = [0, a_prior[metal_line]]

            self.n_metals["n_l_" + metal_line] = 0
            self.fid_metals[metal_line + "_L"] = [0, 0]

        self.n_d_dla = 1
        self.fid_A_damp = [0, -1.4]
        self.n_s_dla = 1
        self.fid_A_scale = [0, 5.2]

        self.fid_AGN = [0, -5.5]

        self.prior_Gauss_rms = None
        self.Gauss_priors = {}
