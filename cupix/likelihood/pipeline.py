from cup1d.utils.utils import create_print_function

import os, sys, time
import numpy as np
from mpi4py import MPI

# our own modules
import lace
from lace.archive import gadget_archive, nyx_archive
from lace.cosmo import camb_cosmo
from cup1d.likelihood.cosmologies import set_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.p1ds import (
    data_gadget,
    data_nyx,
    data_eBOSS_mock,
    data_Chabanier2019,
    data_Karacayli2022,
    data_Karacayli2024,
    data_Ravoux2023,
    data_QMLE_Ohio,
    mock_data,
    data_DESIY1,
    challenge_DESIY1,
)
from cup1d.likelihood import lya_theory, likelihood, fitter
from cup1d.likelihood.model_contaminants import Contaminants
from cup1d.likelihood.model_igm import IGM

from cup1d.likelihood.fitter import Fitter
from cup1d.likelihood.plotter import Plotter


def set_free_like_parameters(params, emulator_label):
    """Set free parameters for likelihood"""
    if params.fix_cosmo:
        free_parameters = []
    else:
        if params.vary_alphas and (
            ("nyx" in emulator_label) | ("Nyx" in emulator_label)
        ):
            free_parameters = ["As", "ns", "nrun"]
        else:
            free_parameters = ["As", "ns"]

    for ii in range(params.n_tau):
        free_parameters.append(f"ln_tau_{ii}")
    for ii in range(params.n_sigT):
        free_parameters.append(f"ln_sigT_kms_{ii}")
    for ii in range(params.n_gamma):
        free_parameters.append(f"ln_gamma_{ii}")
    for ii in range(params.n_kF):
        free_parameters.append(f"ln_kF_{ii}")
    for metal_line in params.metal_lines:
        for ii in range(params.n_metals["n_x_" + metal_line]):
            free_parameters.append(f"ln_x_{metal_line}_{ii}")
        for ii in range(params.n_metals["n_d_" + metal_line]):
            free_parameters.append(f"d_{metal_line}_{ii}")
        for ii in range(params.n_metals["n_l_" + metal_line]):
            free_parameters.append(f"l_{metal_line}_{ii}")
        for ii in range(params.n_metals["n_a_" + metal_line]):
            free_parameters.append(f"a_{metal_line}_{ii}")
    for ii in range(params.n_d_dla):
        free_parameters.append(f"ln_A_damp_{ii}")
    for ii in range(params.n_s_dla):
        free_parameters.append(f"ln_A_scale_{ii}")
    for ii in range(params.n_sn):
        free_parameters.append(f"ln_SN_{ii}")
    for ii in range(params.n_agn):
        free_parameters.append(f"ln_AGN_{ii}")
    for ii in range(params.n_res):
        free_parameters.append(f"R_coeff_{ii}")

    return free_parameters


def set_archive(training_set):
    """Set archive

    Parameters
    ----------
    training_set : str

    Returns
    -------
    archive : object

    """
    if training_set[:3] == "Nyx":
        archive = nyx_archive.NyxArchive(nyx_version=training_set[6:])
    else:
        archive = gadget_archive.GadgetArchive(postproc=training_set)
    return archive


def set_P1D(
    args, archive=None, true_cosmo=None, emulator=None, cull_data=False
):
    """Set P1D data

    Parameters
    ----------
    archive : object
        Archive object containing P1D data
    data_label : str
        Label of simulation/dataset used to generate mock data
    cov_label : str, optional
        Label of covariance matrix
    apply_smoothing : bool or None
        If True, apply smoothing to P1D. If None, do what is best for the input emulator
    z_min : float
        Minimum redshift of P1D measurements
    z_max : float
        Maximum redshift of P1D measurements
    cull_data : bool
        If True, cull data outside of k range from emulator

    Returns
    -------
    data : object
        P1D data
    """

    data_label = args.data_label

    if (
        (data_label[:3] == "mpg")
        | (data_label[:3] == "nyx")
        | (data_label[:5] == "mock_")
        | (data_label == "challenge_DESIY1")
        | (data_label == "eBOSS_mock")
    ):
        theory = lya_theory.set_theory(
            args, emulator, use_hull=False, fid_or_true="true"
        )

    if (data_label[:3] == "mpg") | (data_label[:3] == "nyx"):
        # check if we need to load another archive
        if data_label in archive.list_sim:
            archive_mock = archive
        else:
            if data_label[:3] == "mpg":
                archive_mock = set_archive(training_set="Cabayol23")
            elif data_label[:3] == "nyx":
                archive_mock = set_archive(training_set=args.nyx_training_set)

        if data_label not in archive_mock.list_sim:
            raise ValueError(
                data_label + " not available in archive ",
                archive_mock.list_sim,
            )
        ###################

        # set noise free P1Ds in Mpc
        p1d_ideal = archive_mock.get_testing_data(data_label)
        if len(p1d_ideal) == 0:
            raise ValueError("Could not set P1D data for", data_label)
        else:
            archive_mock = None
        ###################

        # set P1Ds in kms
        if data_label[:3] == "mpg":
            set_p1d_from_mock = data_gadget.Gadget_P1D
        elif data_label[:3] == "nyx":
            set_p1d_from_mock = data_nyx.Nyx_P1D

        data = set_p1d_from_mock(
            theory,
            true_cosmo,
            p1d_ideal,
            input_sim=data_label,
            data_cov_label=args.cov_label,
            cov_fname=args.p1d_fname,
            apply_smoothing=args.apply_smoothing,
            add_noise=args.add_noise,
            seed=args.seed_noise,
            z_min=args.z_min,
            z_max=args.z_max,
        )
    elif data_label[:5] == "mock_":
        # mock data from emulator
        data = mock_data.Mock_P1D(
            theory,
            true_cosmo,
            data_label=data_label[5:],
            add_noise=args.add_noise,
            seed=args.seed_noise,
            z_min=args.z_min,
            z_max=args.z_max,
            p1d_fname=args.p1d_fname,
            cov_only_diag=args.cov_syst_type,
        )

    elif data_label == "challenge_DESIY1":
        data = challenge_DESIY1.P1D_challenge_DESIY1(
            theory,
            true_cosmo,
            p1d_fname=args.p1d_fname,
            z_min=args.z_min,
            z_max=args.z_max,
        )

    # elif data_label == "eBOSS_mock":
    #     # need to be tested
    #     data = data_eBOSS_mock.P1D_eBOSS_mock(
    #         theory,
    #         true_cosmo,
    #         apply_smoothing=args.apply_smoothing,
    #         add_noise=args.add_noise,
    #         seed=args.seed_noise,
    #         z_min=args.z_min,
    #         z_max=args.z_max,
    #     )
    elif data_label == "Chabanier2019":
        data = data_Chabanier2019.P1D_Chabanier2019(
            z_min=args.z_min, z_max=args.z_max
        )
    elif data_label == "Ravoux2023":
        data = data_Ravoux2023.P1D_Ravoux2023(
            z_min=args.z_min, z_max=args.z_max
        )
    elif data_label == "Karacayli2024":
        data = data_Karacayli2024.P1D_Karacayli2024(
            z_min=args.z_min, z_max=args.z_max
        )
    elif data_label == "Karacayli2022":
        data = data_Karacayli2022.P1D_Karacayli2022(
            z_min=args.z_min, z_max=args.z_max
        )
    elif data_label == "challenge_v0":
        file = (
            os.environ["CHALLENGE_PATH"]
            + "fiducial_lym1d_p1d_qmleformat_IC.txt"
        )
        data = data_QMLE_Ohio.P1D_QMLE_Ohio(
            filename=file, z_min=args.z_min, z_max=args.z_max
        )
    elif data_label == "DESIY1":
        data = data_DESIY1.P1D_DESIY1(
            p1d_fname=args.p1d_fname,
            z_min=args.z_min,
            z_max=args.z_max,
            cov_syst_type=args.cov_syst_type,
        )
    else:
        raise ValueError(f"data_label {data_label} not implemented")

    # cull data within emulator range
    if cull_data:
        if args.true_cosmo_label is not None:
            cosmo = set_cosmo(cosmo_label=args.true_cosmo_label)
        else:
            cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

        dkms_dMpc_zmin = camb_cosmo.dkms_dMpc(cosmo, z=np.min(data.z))
        kmin_kms = emulator.kmin_Mpc / dkms_dMpc_zmin
        dkms_dMpc_zmax = camb_cosmo.dkms_dMpc(cosmo, z=np.max(data.z))
        kmax_kms = emulator.kmax_Mpc / dkms_dMpc_zmax
        data.cull_data(kmin_kms=kmin_kms, kmax_kms=kmax_kms)

    data.data_label = data_label

    return data


def set_like(data, emulator, args, data_hires=None):
    """Set likelihood"""

    zs = data.z
    if data_hires is not None:
        zs_hires = data_hires.z
    else:
        zs_hires = None

    # set free parameters
    free_parameters = set_free_like_parameters(args, emulator.emulator_label)
    print(free_parameters)

    ## set theory
    theory = lya_theory.set_theory(
        args, emulator, free_parameters=free_parameters
    )
    theory.model_igm.set_fid_igm(zs)
    fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)
    theory.set_fid_cosmo(zs, input_cosmo=fid_cosmo, zs_hires=zs_hires)

    ## set like
    like = likelihood.Likelihood(
        data,
        theory,
        extra_data=data_hires,
        free_param_names=free_parameters,
        cov_factor=args.cov_factor,
        emu_cov_factor=args.emu_cov_factor,
        emu_cov_type=args.emu_cov_type,
        args=args,
    )

    return like


def path_sampler(
    emulator_label,
    data_label,
    igm_label,
    n_igm,
    cosmo_label,
    cov_label,
    version="v3",
    drop_sim=None,
    apply_smoothing=True,
    data_label_hires=False,
    add_noise=False,
    seed_noise=0,
    fix_cosmo=False,
    vary_alphas=False,
):
    if drop_sim is not None:
        flag_drop = "_drop"
    else:
        flag_drop = ""

    if apply_smoothing:
        flag_smooth = "_smooth"
    else:
        flag_smooth = ""

    if data_label_hires:
        flag_hires = "_" + data_label_hires
    else:
        flag_hires = ""

    try:
        path = os.environ["LYA_DATA_PATH"]
    except:
        raise ValueError("LYA_DATA_PATH not set as environment variable")

    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += "cup1d/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += "sampler/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += version + "/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += "emu_" + emulator_label + "/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += "cov_" + cov_label + flag_hires + "/"
    if os.path.isdir(path) == False:
        os.mkdir(path)

    path += (
        "mock_"
        + data_label
        + "_igm_"
        + igm_label
        + "_cosmo_"
        + cosmo_label
        + "_nigm_"
        + str(n_igm)
        + flag_drop
        + flag_smooth
    )

    if add_noise:
        path += "_noise_" + str(seed_noise)
    if fix_cosmo:
        path += "_fix_cosmo"
    if vary_alphas:
        path += "_vary_alphas"
    path += "/"

    if os.path.isdir(path) == False:
        os.mkdir(path)

    return path


class Pipeline(object):
    """Full pipeline for extracting cosmology from P1D using sampler"""

    def __init__(self, args, make_plots=False, out_folder=None):
        """Set pipeline"""

        self.out_folder = out_folder

        ## MPI stuff
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # create print function (only for rank 0)
        fprint = create_print_function(verbose=args.verbose)
        self.fprint = fprint
        self.explore = args.explore

        # when reusing archive and emulator, these must be None for
        # rank != 0 to prevent a very large memory footprint
        if rank != 0:
            args.archive = None
            args.emulator = None

        ###################

        ## set training set (only for rank 0)
        if rank == 0:
            # start all clocks
            start_all = time.time()

            # only do it if running on mocks or using old emulators
            read_archive = False
            # running on mocks
            if args.data_label[:3] in ["mpg", "nyx"]:
                read_archive = True
            elif args.emulator_label not in ["CH24_mpg_gp", "CH24_nyx_gp"]:
                read_archive = True
            else:
                read_archive = False

            if read_archive:
                start = time.time()
                fprint("----------")
                fprint("Setting training set " + args.training_set)

                # only when reusing archive
                if args.archive is None:
                    archive = set_archive(args.training_set)
                else:
                    archive = args.archive
                end = time.time()
                multi_time = str(np.round(end - start, 2))
                fprint("Training set loaded in " + multi_time + " s")
            else:
                archive = None
        #######################

        ## set emulator
        if rank == 0:
            fprint("----------")
            fprint("Setting emulator")
            start = time.time()

            if args.emulator is None:
                _drop_sim = None
                if args.drop_sim:
                    _drop_sim = args.data_label

                emulator = set_emulator(
                    emulator_label=args.emulator_label,
                    archive=archive,
                    drop_sim=_drop_sim,
                )

                # if "Nyx" in emulator.emulator_label:
                #     emulator.list_sim_cube = archive.list_sim_cube
                #     if "nyx_14" in emulator.list_sim_cube:
                #         emulator.list_sim_cube.remove("nyx_14")
                # else:
                #     emulator.list_sim_cube = archive.list_sim_cube
            else:
                emulator = args.emulator

            multi_time = str(np.round(time.time() - start, 2))
            fprint("Emulator set in " + multi_time + " s")

            # distribute emulator to all ranks
            for irank in range(1, size):
                comm.send(emulator, dest=irank, tag=(irank + 1) * 7)
        else:
            # receive emulator from ranks 0
            emulator = comm.recv(source=0, tag=(rank + 1) * 7)

        #######################

        ## set P1D
        if rank == 0:
            fprint("----------")
            fprint("Setting P1D")
            start = time.time()

            # set fiducial cosmology
            if args.true_cosmo_label is not None:
                true_cosmo = set_cosmo(cosmo_label=args.true_cosmo_label)
            else:
                true_cosmo = None

            data = {"P1Ds": None, "extra_P1Ds": None}

            # set P1D
            data["P1Ds"] = set_P1D(
                args,
                archive=archive,
                true_cosmo=true_cosmo,
                emulator=emulator,
            )
            fprint(
                "Set " + str(len(data["P1Ds"].z)) + " P1Ds at z = ",
                data["P1Ds"].z,
            )

            # set hires P1D
            if args.data_label_hires is not None:
                data["extra_P1Ds"] = set_P1D(
                    args,
                    archive=archive,
                    true_cosmo=true_cosmo,
                    emulator=emulator,
                )
                fprint(
                    "Set " + str(len(data["extra_P1Ds"].z)) + " P1Ds at z = ",
                    data["extra_P1Ds"].z,
                )
            # distribute data to all tasks
            for irank in range(1, size):
                comm.send(data, dest=irank, tag=(irank + 1) * 11)
        else:
            # get testing_data from task 0
            data = comm.recv(source=0, tag=(rank + 1) * 11)

        if rank == 0:
            multi_time = str(np.round(time.time() - start, 2))
            fprint("P1D set in " + multi_time + " s")

        #######################

        ## Validating data

        # check if data is blinded
        fprint("----------")
        fprint("Is the data blinded: ", data["P1Ds"].apply_blinding)
        if data["P1Ds"].apply_blinding:
            fprint("Type of blinding: ", data["P1Ds"].blinding)

        if rank == 0:
            # TBD save to file!
            if make_plots:
                data["P1Ds"].plot_p1d()
                if args.data_label_hires is not None:
                    data["extra_P1Ds"].plot_p1d()

                try:
                    data["P1Ds"].plot_igm()
                except:
                    print("Real data, no true IGM history")

        #######################

        ## set likelihood
        fprint("----------")
        fprint("Setting likelihood")

        like = set_like(
            data["P1Ds"],
            emulator,
            args,
            data_hires=data["extra_P1Ds"],
        )

        # print(like.truth)
        # print("out")
        # sys.exit()

        ## Validating likelihood

        if rank == 0:
            # TBD save to file!
            if make_plots:
                like.plot_p1d(residuals=False)
                like.plot_p1d(residuals=True)
                like.plot_igm()

        # print parameters
        for p in like.free_params:
            fprint(p.name, p.value, p.min_value, p.max_value)

        #######################

        # self.set_emcee_options(
        #     args.data_label,
        #     args.cov_label,
        #     args.n_igm,
        #     n_steps=args.n_steps,
        #     n_burn_in=args.n_burn_in,
        #     test=args.test,
        # )

        ## set fitter

        self.fitter = Fitter(
            like=like,
            rootdir=self.out_folder,
            nburnin=args.n_burn_in,
            nsteps=args.n_steps,
            parallel=args.parallel,
            explore=args.explore,
            fix_cosmology=args.fix_cosmo,
        )

        #######################

        if rank == 0:
            multi_time = str(np.round(time.time() - start_all, 2))
            fprint("Setting the sampler took " + multi_time + " s \n\n")

    def set_emcee_options(
        self,
        data_label,
        cov_label,
        n_igm,
        n_steps=0,
        n_burn_in=0,
        test=False,
    ):
        # set steps
        if test == True:
            self.n_steps = 10
        else:
            if n_steps != 0:
                self.n_steps = n_steps
            else:
                if data_label == "Chabanier2019":
                    self.n_steps = 2000
                else:
                    self.n_steps = 1250

        # set burn-in
        if test == True:
            self.n_burn_in = 0
        else:
            if n_burn_in != 0:
                self.n_burn_in = n_burn_in
            else:
                if data_label == "Chabanier2019":
                    self.n_burn_in = 2000
                else:
                    if cov_label == "Chabanier2019":
                        self.n_burn_in = 1500
                    elif cov_label == "QMLE_Ohio":
                        self.n_burn_in = 1500
                    else:
                        self.n_burn_in = 1500

    def run_minimizer(self, p0, make_plots=True, save_chains=False):
        """
        Run the minimizer (only rank 0)
        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            start = time.time()
            self.fprint("----------")
            self.fprint("Running minimizer")
            # start fit from initial values
            self.fitter.run_minimizer(
                log_func_minimize=self.fitter.like.get_chi2, p0=p0
            )

            # save fit
            self.fitter.save_fitter(save_chains=save_chains)

            if make_plots:
                # plot fit
                self.plotter = Plotter(
                    self.fitter, save_directory=self.fitter.save_directory
                )
                self.plotter.plots_minimizer()

            # distribute best_fit to all tasks
            for irank in range(1, size):
                comm.send(
                    self.fitter.mle_cube, dest=irank, tag=(irank + 1) * 13
                )
        else:
            # get testing_data from task 0
            self.fitter.mle_cube = comm.recv(source=0, tag=(rank + 1) * 13)

    def run_sampler(self, make_plots=True):
        """
        Run the sampler (after minimizer)
        """

        def func_for_sampler(p0):
            res = self.fitter.like.get_log_like(values=p0, return_blob=True)
            return res[0], *res[2]

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            start = time.time()
            self.fprint("----------")
            self.fprint("Running sampler")

        # make sure all tasks start at the same time
        self.fitter.run_sampler(
            pini=self.fitter.mle_cube, log_func=func_for_sampler
        )

        if rank == 0:
            end = time.time()
            multi_time = str(np.round(end - start, 2))
            self.fprint("Sampler run in " + multi_time + " s")

            self.fprint("----------")
            self.fprint("Saving data")
            self.fitter.save_fitter(save_chains=True)

            # plot fit
            if make_plots:
                self.plotter = Plotter(
                    self.fitter, save_directory=self.fitter.save_directory
                )
                self.plotter.plots_sampler()
