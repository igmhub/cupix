import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.stats.distributions import chi2 as chi2_scipy
from scipy.optimize import minimize
from scipy.linalg import block_diag

import lace
from lace.cosmo import camb_cosmo
from cupix.utils.utils import is_number_string


def get_bin_coverage(xmin_o, xmax_o, xmin_n, xmax_n):
    """Trick to accelerate rebinning"""
    # check out https://stcorp.github.io/harp/doc/html/algorithms/regridding.html
    cover = np.zeros((len(xmin_n), len(xmin_o)))
    for jj in range(len(xmin_n)):
        cover[jj] = np.fmax(
            (np.fmin(xmax_o, xmax_n[jj]) - np.fmax(xmin_o, xmin_n[jj]))
            / (xmax_o - xmin_o),
            0,
        )
    return cover


class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

    def __init__(
        self,
        data,
        theory,
        free_param_names=None,
        free_param_limits=None,
        verbose=False,
        cov_factor=1.0,
        prior_Gauss_rms=None,
        emu_cov_factor=None,
        emu_cov_type="block",
        extra_data=None,
        min_log_like=-1e100,
        args=None,
    ):
        """Setup likelihood from theory and data. Options:
        - data (required) is the data to model
        - theory (required) instance of lya_theory
        - free_param_names is a list of param names, in any order
        - free_param_limits list of tuples, same order than free_param_names
        - if prior_Gauss_rms is None it will use uniform priors
        - ignore k-bins with k > kmin_kms
        - cov_factor adjusts the contribution from data covariance
        - emu_cov_factor adjusts the contribution from emulator covariance
        set between 0 and 1.
        - extra_p1d_data: extra P1D data, e.g., from HIRES
        - min_log_like: use this instead of - infinity"""

        self.verbose = verbose
        self.prior_Gauss_rms = prior_Gauss_rms
        self.cov_factor = cov_factor
        self.emu_cov_factor = emu_cov_factor
        self.emu_cov_type = emu_cov_type
        self.min_log_like = min_log_like
        self.data = data
        self.extra_data = extra_data
        # we only do this for latter save all relevant after fitting the model
        self.args = {}
        for attr, value in args.__dict__.items():
            if attr not in ["archive", "emulator"]:
                self.args[attr] = value

        if "rebin_k" not in self.args:
            self.args["rebin_k"] = 1

        # k_kms_new2 = np.linspace(kmin[0], kmax[-1], len(k_kms) * 20)
        if self.args["rebin_k"] != 1:
            self.rebin = {}
            self.rebin["k_kms"] = []  # new k_kms
            self.rebin["cover"] = []  # to accelerate rebinning
            self.rebin["sum_cover"] = []  # to accelerate rebinning
            for iz in range(len(self.data.z)):
                nelem = len(self.data.k_kms[iz]) * self.args["rebin_k"]
                _kms_reb = np.linspace(
                    self.data.k_kms_min[iz][0],
                    self.data.k_kms_max[iz][-1],
                    nelem,
                )
                self.rebin["k_kms"].append(_kms_reb)
                xmin_o = _kms_reb - 0.5 * (_kms_reb[1] - _kms_reb[0])
                xmax_o = _kms_reb + 0.5 * (_kms_reb[1] - _kms_reb[0])

                _cover = get_bin_coverage(
                    xmin_o,
                    xmax_o,
                    self.data.k_kms_min[iz],
                    self.data.k_kms_max[iz],
                )
                self.rebin["cover"].append(_cover)
                self.rebin["sum_cover"].append(np.sum(_cover, axis=1))

        self.theory = theory
        # Set inverse covariance. We do it here so we can account for emulator error
        self.set_icov()

        # setup parameters
        self.free_param_names = free_param_names
        self.set_free_parameters(free_param_names, free_param_limits)
        if verbose:
            print(len(self.free_params), "free parameters")

        self.set_Gauss_priors()

        # sometimes we want to know the true theory (when working with mocks)
        self.set_truth()

        # store also fiducial model
        self.set_fid()

    def rebinning(self, zs, Pk_kms_finek):
        """For rebinning Pk predictions"""
        Pk_kms_origk = []
        # _Pk_kms_finek = np.atleast_1d(Pk_kms_finek)
        for iz in range(len(zs)):
            indz = np.argmin(np.abs(self.data.z - zs[iz]))
            _Pk_kms = (
                np.sum(
                    self.rebin["cover"][indz] * Pk_kms_finek[iz][np.newaxis, :],
                    axis=1,
                )
                / self.rebin["sum_cover"][indz]
            )
            Pk_kms_origk.append(_Pk_kms)
        return Pk_kms_origk

    def set_Gauss_priors(self):
        """
        Sets Gaussian priors on the parameters
        """

        self.Gauss_priors = np.ones((len(self.free_params)))
        for ii, par_like in enumerate(self.free_params):
            if self.prior_Gauss_rms is not None:
                _prior = self.prior_Gauss_rms
            elif par_like.Gauss_priors_width is not None:
                _fid = par_like.value
                _width = par_like.Gauss_priors_width
                _low = par_like.get_value_in_cube(_fid - 0.5 * _width)
                _high = par_like.get_value_in_cube(_fid + 0.5 * _width)
                _prior = _high - _low
            else:
                _prior = 1e4  # so we get zero

            self.Gauss_priors[ii] = _prior

    def set_icov(self):
        """
        Computes and sets the inverse covariance matrix for the P1 power spectrum data and full power spectrum data.

        This method processes the main dataset (`data`) and any additional dataset (`extra_data`) associated
        with the object. For each dataset:
        - It computes the inverse covariance matrices for the power spectrum (`Pk_kms`) at different redshifts,
          incorporating an emulator error factor.
        - It computes the inverse covariance matrix for the full power spectrum data, if available.

        The resulting inverse covariance matrices are stored in instance attributes.

        Attributes Modified:
        --------------------
        icov_Pk_kms : list of numpy.ndarray
            List of inverse covariance matrices for the power spectrum of the main dataset at different redshifts.

        full_icov_Pk_kms : numpy.ndarray or None
            Inverse covariance matrix for the full power spectrum of the main dataset.
            Set to `None` if the full power spectrum is not available.

        extra_icov_Pk_kms : list of numpy.ndarray
            List of inverse covariance matrices for the power spectrum of the additional dataset at different redshifts.
            Set to `None` if `extra_data` is not provided.

        extra_full_icov_Pk_kms : numpy.ndarray or None
            Inverse covariance matrix for the full power spectrum of the additional dataset.
            Set to `None` if the full power spectrum is not available or if `extra_data` is not provided.

        Notes:
        -----
        - The emulator error is added to the diagonal of the covariance matrix before inverting. The error is
          computed as `(data.Pk_kms * emu_cov_factor) ** 2`, where `emu_cov_factor` is an attribute of the object.
        - The method iterates over redshift bins (`data.z`) and processes the covariance matrices accordingly.
        - If the dataset (`data` or `extra_data`) is `None`, no processing occurs for that dataset.

        Raises:
        -------
        ValueError:
            If the covariance matrix inversion fails (e.g., due to singularity).
        """

        # get emulator error
        filename = "cov_" + self.theory.emulator.emulator_label + ".npy"
        full_path = os.path.join(
            os.path.dirname(lace.__path__[0]), "data", "covariance", filename
        )
        dict_save = np.load(full_path, allow_pickle=True).item()
        emu_cov = dict_save["cov"]
        emu_cov_zz = dict_save["zz"]
        emu_cov_unique_zz = np.unique(emu_cov_zz)
        emu_cov_k_Mpc = dict_save["k_Mpc"]

        # Iterate over both datasets: main dataset (idata = 0) and additional dataset (idata = 1)
        for idata in range(2):
            if idata == 0:  # Main dataset
                data = self.data
                # Initialize list to store inverse covariance matrices for Pk_kms
                self.icov_Pk_kms = []
                self.cov_Pk_kms = []
                # Initialize the full inverse covariance matrix for Pk_kms
                self.full_icov_Pk_kms = None
                self.full_cov_Pk_kms = None
            else:  # Additional dataset
                data = self.extra_data
                # Initialize list for extra inverse covariance matrices
                self.extra_icov_Pk_kms = []
                self.extra_cov_Pk_kms = []
                # Initialize the full inverse covariance matrix for extra data
                self.extra_full_icov_Pk_kms = None
                self.extra_full_cov_Pk_kms = None

            if data is None:  # Skip if no data is provided for this dataset
                continue

            if data.Pksmooth_kms is not None:
                pksmooth = data.Pksmooth_kms
            else:
                pksmooth = data.Pk_kms

            # Total number of k values across all redshifts
            nks = 0
            for ii in range(len(data.z)):
                nks += len(data.Pk_kms[ii])

            # Process each redshift bin
            emu_cov_blocks = []
            for ii in range(len(data.z)):
                # Copy the covariance matrix for the current redshift bin
                cov = data.cov_Pk_kms[ii].copy()
                covemu = np.zeros_like(cov)
                # Indices of the diagonal elements
                # Add emulator error
                if self.emu_cov_factor is not None:
                    if self.emu_cov_factor != 0:
                        if ii == 0:
                            if idata == 0:
                                self.covemu_Pk_kms = []
                            else:
                                self.extra_covemu_Pk_kms = []

                        dkms_dMpc = self.theory.fid_cosmo["cosmo"].dkms_dMpc(
                            data.z[ii]
                        )
                        # data k_kms to Mpc
                        k_Mpc = data.k_kms[ii] * dkms_dMpc
                        add_emu_cov_kms = np.zeros(
                            (k_Mpc.shape[0], k_Mpc.shape[0])
                        )

                        # find closest z in cov
                        ind = np.argmin(np.abs(emu_cov_unique_zz - data.z[ii]))
                        # get cov from closest z
                        ind = np.argwhere(emu_cov_zz == emu_cov_unique_zz[ind])[
                            :, 0
                        ]
                        _emu_cov = emu_cov[ind, :][:, ind]
                        _k_Mpc = emu_cov_k_Mpc[ind]

                        # rescale covariance matrix by power spectrum,
                        # I stored the relative error
                        for i0 in range(k_Mpc.shape[0]):
                            # get closest k in emu cov matrix
                            j0 = np.argmin(np.abs(k_Mpc[i0] - _k_Mpc))
                            for i1 in range(k_Mpc.shape[0]):
                                # skip if diagonal and i0 != i1
                                if (self.emu_cov_type == "diagonal") & (
                                    i0 != i1
                                ):
                                    continue

                                # get closest k in emu cov matrix
                                j1 = np.argmin(np.abs(k_Mpc[i1] - _k_Mpc))
                                add_emu_cov_kms[i0, i1] = (
                                    _emu_cov[j0, j1]
                                    * pksmooth[ii][i0]
                                    * pksmooth[ii][i1]
                                )
                        emu_cov_blocks.append(add_emu_cov_kms)
                        # from Pk in Mpc to Pk in km/s
                        # add to cov
                        for i0 in range(k_Mpc.shape[0]):
                            for i1 in range(k_Mpc.shape[0]):
                                cov[i0, i1] += (
                                    add_emu_cov_kms[i0, i1]
                                    * self.emu_cov_factor
                                )

                # inflate errors
                cov *= self.cov_factor
                # Compute and store the inverse covariance matrix
                if idata == 0:
                    self.icov_Pk_kms.append(np.linalg.inv(cov))
                    self.cov_Pk_kms.append(cov)
                    if self.emu_cov_factor is not None:
                        if self.emu_cov_factor != 0:
                            self.covemu_Pk_kms.append(
                                add_emu_cov_kms * self.emu_cov_factor
                            )
                else:
                    self.extra_icov_Pk_kms.append(np.linalg.inv(cov))
                    self.extra_cov_Pk_kms.append(cov)
                    if self.emu_cov_factor is not None:
                        if self.emu_cov_factor != 0:
                            self.extra_covemu_Pk_kms.append(
                                add_emu_cov_kms * self.emu_cov_factor
                            )

            # Process the full power spectrum data if available
            if data.full_Pk_kms is not None:
                # Copy the full covariance matrix
                cov = data.full_cov_Pk_kms.copy()
                # Indices of the diagonal elements
                if self.emu_cov_factor is not None:
                    if self.emu_cov_factor != 0:
                        # diagonal
                        if self.emu_cov_type == "diagonal":
                            diag_emu_cov = []
                            for ii in range(len(emu_cov_blocks)):
                                diag_emu_cov.append(np.diag(emu_cov_blocks[ii]))
                            full_emu_cov = np.concatenate(diag_emu_cov)
                            ind = np.diag_indices_from(cov)
                            cov[ind] += full_emu_cov * self.emu_cov_factor
                        # block
                        elif self.emu_cov_type == "block":
                            full_emu_cov = block_diag(*emu_cov_blocks)
                            cov += full_emu_cov * self.emu_cov_factor
                        # full
                        else:
                            full_emu_cov = np.zeros_like(cov)
                            for i0 in range(cov.shape[0]):
                                dkms_dMpc = self.theory.fid_cosmo[
                                    "cosmo"
                                ].dkms_dMpc(data.full_zs[i0])
                                j0 = np.argmin(
                                    (data.full_zs[i0] - emu_cov_zz) ** 2
                                    + (
                                        data.full_k_kms[i0] * dkms_dMpc
                                        - emu_cov_k_Mpc
                                    )
                                    ** 2
                                )
                                for i1 in range(cov.shape[0]):
                                    dkms_dMpc = self.theory.fid_cosmo[
                                        "cosmo"
                                    ].dkms_dMpc(data.full_zs[i1])
                                    j1 = np.argmin(
                                        (data.full_zs[i1] - emu_cov_zz) ** 2
                                        + (
                                            data.full_k_kms[i1] * dkms_dMpc
                                            - emu_cov_k_Mpc
                                        )
                                        ** 2
                                    )
                                    full_emu_cov[i0, i1] = (
                                        emu_cov[j0, j1]
                                        * data.full_Pk_kms[i0]
                                        * data.full_Pk_kms[i1]
                                    )

                            cov += full_emu_cov * self.emu_cov_factor

                # inflate errors
                cov *= self.cov_factor
                # Compute and store the inverse covariance matrix
                if idata == 0:
                    self.full_icov_Pk_kms = np.linalg.inv(cov)
                    self.full_cov_Pk_kms = cov
                    if self.emu_cov_factor is not None:
                        if self.emu_cov_factor != 0:
                            self.emu_full_cov_Pk_kms = (
                                full_emu_cov * self.emu_cov_factor
                            )
                else:
                    self.extra_full_icov_Pk_kms = np.linalg.inv(cov)
                    self.extra_full_cov_Pk_kms = cov
                    if self.emu_cov_factor is not None:
                        if self.emu_cov_factor != 0:
                            self.extra_emu_full_cov_Pk_kms = (
                                full_emu_cov * self.emu_cov_factor
                            )

    def set_free_parameters(self, free_param_names, free_param_limits):
        """Setup likelihood parameters that we want to vary"""

        # setup list of likelihood free parameters
        self.free_params = []

        if free_param_limits is not None:
            assert len(free_param_limits) == len(
                free_param_names
            ), "wrong number of parameter limits"

        # get all parameters in theory, free or not
        params = self.theory.get_parameters()

        ## select free parameters, make sure ordering
        ## in self.free_params is same as in free_param_names
        for par_name in free_param_names:
            for par in params:
                if par.name == par_name:
                    if free_param_limits is not None:
                        ## Set min and max of each parameter if
                        ## a list is given. otherwise leave as default
                        par.min_value = free_param_limits[
                            free_param_names.index(par.name)
                        ][0]
                        par.max_value = free_param_limits[
                            free_param_names.index(par.name)
                        ][1]
                    self.free_params.append(par)

        Nfree = len(self.free_params)
        Nin = len(free_param_names)

        assert Nfree == Nin, "could not setup free parameters"

        if self.verbose:
            print("likelihood setup with {} free parameters".format(Nfree))

        return

    def parameters_from_sampling_point(self, values):
        """Translate input array of values (in cube) to likelihood parameters"""

        if values is None:
            return []

        assert len(values) == len(self.free_params), "size mismatch"
        Npar = len(values)
        like_params = []
        for ip in range(Npar):
            par = self.free_params[ip].get_new_parameter(values[ip])
            like_params.append(par)

        return like_params

    def cosmology_params_from_sampling_point(self, values):
        """For a given point in sampling space, return a list of
        cosmology params"""

        like_params = self.parameters_from_sampling_point(values)

        ## Dictionary of cosmology parameters
        cosmo_dict = {}

        for like_param in like_params:
            if like_param.name == "ombh2":
                cosmo_dict["ombh2"] = like_param.value
            elif like_param.name == "omch2":
                cosmo_dict["omch2"] = like_param.value
            elif like_param.name == "cosmomc_theta":
                cosmo_dict["cosmomc_theta"] = like_param.value
            elif like_param.name == "As":
                cosmo_dict["As"] = like_param.value
            elif like_param.name == "ns":
                cosmo_dict["ns"] = like_param.value
            elif like_param.name == "mnu":
                cosmo_dict["mnu"] = like_param.value
            elif like_param.name == "nrun":
                cosmo_dict["nrun"] = like_param.value

        assert (
            len(cosmo_dict) > 0
        ), "No cosmology parameters found in sampling space"

        return cosmo_dict

    def set_truth(self):
        """Store true cosmology from the simulation used to make mock data"""

        # access true cosmology used in mock data
        if hasattr(self.data, "truth") == False:
            print("will not store truth, working with real data")
            self.truth = None
            return

        self.truth = {}
        for par in self.data.truth:
            self.truth[par] = self.data.truth[par]

        # make sure that we compare the correct zs
        ztruth = self.data.truth["igm"]["z"]
        zfid = self.theory.model_igm.fid_igm["z"]
        mask_z = np.zeros(len(ztruth), dtype=int) - 1
        for ii in range(len(mask_z)):
            ind = np.argwhere(ztruth[ii] == zfid)[:, 0]
            if len(ind) != 0:
                mask_z[ii] = ind[0]

        ind = np.argwhere(mask_z != -1)[:, 0]
        mask_z = mask_z[ind]

        # equal_IGM for each IGM differently!!!
        equal_IGM = True
        for key in self.data.truth["igm"]:
            if key not in self.theory.model_igm.fid_igm:
                continue
            lenz = self.theory.model_igm.fid_igm[key].shape[0]
            if (
                np.allclose(
                    np.array(self.data.truth["igm"][key])[mask_z],
                    self.theory.model_igm.fid_igm[key],
                )
                == False
            ):
                equal_IGM = False
                break

        self.truth["like_params"] = {}
        self.truth["like_params_cube"] = {}
        pname2 = {"As": "Delta2_star", "ns": "n_star", "nrun": "alpha_star"}
        for par in self.free_params:
            if (
                ("tau" in par.name)
                | ("sigT" in par.name)
                | ("gamma" in par.name)
                | ("kF" in par.name)
            ):
                if equal_IGM:
                    if ("tau" in par.name) and (
                        self.args["mF_model_type"] == "chunks"
                    ):
                        self.truth["like_params"][par.name] = 1
                        self.truth["like_params_cube"][
                            par.name
                        ] = par.get_value_in_cube(
                            self.truth["like_params"][par.name]
                        )
                    else:
                        self.truth["like_params"][par.name] = 0
                        self.truth["like_params_cube"][
                            par.name
                        ] = par.get_value_in_cube(
                            self.truth["like_params"][par.name]
                        )
                else:
                    self.truth["like_params"][par.name] = np.infty
                    self.truth["like_params_cube"][par.name] = np.infty
            elif (par.name == "As") | (par.name == "ns") | (par.name == "nrun"):
                self.truth["like_params"][par.name] = self.truth["cosmo"][
                    par.name
                ]
                self.truth["like_params_cube"][
                    par.name
                ] = par.get_value_in_cube(self.truth["like_params"][par.name])
                self.truth["like_params"][pname2[par.name]] = self.truth[
                    "linP"
                ][pname2[par.name]]
            else:
                if par.name not in self.truth["cont"]:
                    print("could not find {} in truth".format(par.name))
                    continue
                self.truth["like_params"][par.name] = self.truth["cont"][
                    par.name
                ]
                self.truth["like_params_cube"][
                    par.name
                ] = par.get_value_in_cube(self.truth["cont"][par.name])

    def set_fid(self):
        """Store fiducial cosmology assumed for the fit"""

        self.fid = {}

        sim_cosmo = self.theory.fid_cosmo["cosmo"].cosmo

        self.fid["cosmo"] = {}
        self.fid["cosmo"]["ombh2"] = sim_cosmo.ombh2
        self.fid["cosmo"]["omch2"] = sim_cosmo.omch2
        self.fid["cosmo"]["As"] = sim_cosmo.InitPower.As
        self.fid["cosmo"]["ns"] = sim_cosmo.InitPower.ns
        self.fid["cosmo"]["nrun"] = sim_cosmo.InitPower.nrun
        self.fid["cosmo"]["H0"] = sim_cosmo.H0
        self.fid["cosmo"]["mnu"] = camb_cosmo.get_mnu(sim_cosmo)

        blob_params = ["Delta2_star", "n_star", "alpha_star"]
        blob = self.theory.fid_cosmo["cosmo"].get_linP_params()

        self.fid["igm"] = self.theory.model_igm.fid_igm
        self.fid["fit"] = {}
        self.fid["fit_cube"] = {}
        self.fid["linP"] = {}

        pname2 = {"As": "Delta2_star", "ns": "n_star", "nrun": "alpha_star"}
        for par in self.free_params:
            self.fid["fit"][par.name] = par.value
            self.fid["fit_cube"][par.name] = par.get_value_in_cube(par.value)
            if (par.name == "As") | (par.name == "ns") | (par.name == "nrun"):
                self.fid["fit"][pname2[par.name]] = blob[pname2[par.name]]
                self.fid["linP"][pname2[par.name]] = blob[pname2[par.name]]

    def get_px_kms(
        self,
        zs=None,
        _k_kms=None,
        theta_bin_deg=None,
        values=None,
        return_covar=False,
        return_blob=False,
        return_emu_params=False,
        apply_hull=True,
    ):
        """Compute theoretical prediction for 1D P(k)"""

        if _k_kms is None:
            k_kms = self.data.k_kms

        if zs is None:
            zs = self.data.z

        if theta_bin_deg is None:
            theta_bin_deg = self.data.theta_bin_deg

        if self.args["rebin_k"] != 1:
            k_kms = []
            zs = np.atleast_1d(zs)
            for iz in range(len(zs)):
                ind = np.argmin(np.abs(zs[iz] - self.data.z))
                k_kms.append(self.rebin["k_kms"][ind])
        else:
            k_kms = _k_kms

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params = self.parameters_from_sampling_point(values)
        else:
            like_params = []


        results = self.theory.get_px_kms(
            zs,
            k_kms,
            theta_bin_deg=theta_bin_deg,
            like_params=like_params,
            return_covar=return_covar,
            return_blob=return_blob,
            return_emu_params=return_emu_params,
            apply_hull=apply_hull,
        )

        if results is None:
            return None
        else:
            if self.args["rebin_k"] == 1:
                return results
            else:
                if return_blob | return_emu_params:
                    results2 = []
                    results2.append(self.rebinning(zs, results[0]))
                    for ii in range(1, len(results)):
                        results2.append(results[ii])
                    return results2
                else:
                    return self.rebinning(zs, results)

    def get_chi2(self, values=None, return_all=False, zmask=None):
        """Compute chi2 using data and theory, without adding
        emulator covariance"""

        log_like, log_like_all = self.get_log_like(
            values, ignore_log_det_cov=True, zmask=zmask
        )

        # print(-2 * log_like, -2 * log_like_all, -2 * np.sum(log_like_all))

        if return_all:
            return -2.0 * log_like, -2.0 * log_like_all
        else:
            return -2.0 * log_like

    def get_log_like(
        self,
        values=None,
        ignore_log_det_cov=True,
        return_blob=False,
        zmask=None,
    ):
        """Compute log(likelihood), including determinant of covariance
        unless you are setting ignore_log_det_cov=True."""

        # what to return if we are out of priors
        null_out = [-np.inf, -np.inf]
        if return_blob:
            blob = (0, 0, 0, 0, 0, 0)
            null_out.append(blob)

        # check that we are within unit cube
        if values is not None:
            if (values > 1.0).any() | (values < 0.0).any():
                return null_out

        # ask emulator prediction for P1D in each bin
        if zmask is not None:
            _res = []
            for iz in range(len(self.data.z)):
                ind = np.argwhere(np.abs(zmask - self.data.z[iz]) < 1e-3)
                if len(ind) == 0:
                    _res.append(0)
                else:
                    _ = self.get_p1d_kms(
                        np.atleast_1d(self.data.z[iz]),
                        np.atleast_2d(self.data.k_kms[iz]),
                        values,
                        return_blob=return_blob,
                    )
                    if _ is None:
                        _res = None
                        break
                    else:
                        _res.append(_[0])
        else:
            _res = self.get_p1d_kms(
                self.data.z, self.data.k_kms, values, return_blob=return_blob
            )

        # out of priors
        if _res is None:
            return null_out

        if return_blob:
            emu_p1d, blob = _res
        else:
            emu_p1d = _res

        # use high-res data
        if self.extra_data is not None:
            length = 2
            nz = np.max([len(self.data.z), len(self.extra_data.z)])

            if zmask is not None:
                _res = []
                for iz in range(len(self.extra_data.z)):
                    ind = np.argwhere(
                        np.abs(zmask - self.extra_data.z[iz]) < 1e-3
                    )
                    if len(ind) == 0:
                        _res.append(0)
                    else:
                        _res = self.get_p1d_kms(
                            np.atleast_1d(self.extra_data.z[iz]),
                            np.atleast_1d(self.extra_data.k_kms[iz]),
                            values,
                            return_blob=return_blob,
                        )
            else:
                _res = self.get_p1d_kms(
                    self.extra_data.z,
                    self.extra_data.k_kms,
                    values,
                    return_blob=return_blob,
                )

            emu_p1d_extra = _res
            # out of priors
            if emu_p1d_extra is None:
                return null_out

        else:
            length = 1
            nz = len(self.data.z)

        # compute log like contribution from each redshift bin
        log_like_all = np.zeros((length, nz))
        log_like = 0
        # loop over low and high res data
        for ii in range(length):
            if ii == 0:
                emu_p1d_use = emu_p1d
                data = self.data
                icov_Pk_kms = self.icov_Pk_kms
                full_icov_Pk_kms = self.full_icov_Pk_kms
            else:
                emu_p1d_use = emu_p1d_extra
                data = self.extra_data
                icov_Pk_kms = self.extra_icov_Pk_kms
                full_icov_Pk_kms = self.extra_full_icov_Pk_kms

            # loop over redshift bins
            for iz in range(len(data.z)):
                if zmask is not None:
                    ind = np.argwhere(np.abs(zmask - data.z[iz]) < 1e-3)
                    if len(ind) == 0:
                        continue
                # compute chi2 for this redshift bin
                diff = data.Pk_kms[iz] - emu_p1d_use[iz]
                chi2_z = np.dot(np.dot(icov_Pk_kms[iz], diff), diff)
                # check whether to add determinant of covariance as well
                if ignore_log_det_cov:
                    log_like_all[ii, iz] = -0.5 * chi2_z
                else:
                    log_det_cov = np.log(
                        np.abs(1 / np.linalg.det(icov_Pk_kms[iz]))
                    )
                    log_like_all[ii, iz] = -0.5 * (chi2_z + log_det_cov)

            if (full_icov_Pk_kms is None) | (zmask is not None):
                log_like += np.sum(log_like_all[ii])
            else:
                # compute chi2 using full cov
                diff = data.full_Pk_kms - np.concatenate(emu_p1d_use)
                chi2_all = np.dot(np.dot(full_icov_Pk_kms, diff), diff)
                if ignore_log_det_cov:
                    log_like += -0.5 * chi2_all
                else:
                    log_det_cov = np.log(
                        np.abs(1 / np.linalg.det(full_icov_Pk_kms))
                    )
                    log_like += -0.5 * (chi2_all + log_det_cov)

        # something went wrong
        if np.isnan(log_like):
            return null_out

        out = [log_like, log_like_all]
        if return_blob:
            out.append(blob)
        return out

    def regulate_log_like(self, log_like):
        """Make sure that log_like is not NaN, nor tiny"""

        if (log_like is None) or math.isnan(log_like):
            return self.min_log_like

        return max(self.min_log_like, log_like)

    def compute_log_prob(
        self, values, return_blob=False, ignore_log_det_cov=False, zmask=None
    ):
        """Compute log likelihood plus log priors for input values
        - if return_blob==True, it will return also extra information"""

        # Always force parameter to be within range (for now)
        if (max(values) > 1.0) or (min(values) < 0.0):
            if return_blob:
                dummy_blob = self.theory.get_blob()
                return self.min_log_like, dummy_blob
            else:
                return self.min_log_like

        # compute log_prior
        log_prior = self.get_log_prior(values)

        # compute log_like (option to ignore emulator covariance)
        if return_blob:
            log_like, chi2_all, blob = self.get_log_like(
                values,
                ignore_log_det_cov=ignore_log_det_cov,
                return_blob=True,
                zmask=zmask,
            )
        else:
            log_like, chi2_all = self.get_log_like(
                values,
                ignore_log_det_cov=ignore_log_det_cov,
                return_blob=False,
                zmask=zmask,
            )

        # regulate log-like (not NaN, not tiny)
        log_like = self.regulate_log_like(log_like)

        if return_blob:
            return log_like + log_prior, blob
        else:
            return log_like + log_prior

    def log_prob(self, values, ignore_log_det_cov=False, zmask=None):
        """Return log likelihood plus log priors"""

        return self.compute_log_prob(
            values,
            return_blob=False,
            ignore_log_det_cov=ignore_log_det_cov,
            zmask=zmask,
        )

    def log_prob_and_blobs(self, values, ignore_log_det_cov=False):
        """Function used by emcee to get both log_prob and extra information"""

        lnprob, blob = self.compute_log_prob(
            values, return_blob=True, ignore_log_det_cov=ignore_log_det_cov
        )
        # unpack tuple
        out = lnprob, *blob
        return out

    def get_log_prior(self, values):
        """Compute logarithm of prior"""

        assert len(values) == len(self.free_params), "size mismatch"

        # Always force parameter to be within range (for now)
        if max(values) > 1:
            return self.min_log_like
        if min(values) < 0:
            return self.min_log_like

        fid_values = [p.value_in_cube() for p in self.free_params]
        log_prior = -np.sum(
            (np.array(fid_values) - values) ** 2 / (2 * self.Gauss_priors**2)
        )
        return log_prior

    def minus_log_prob(self, values, zmask=None):
        """Return minus log_prob (needed to maximise posterior)"""

        return -1.0 * self.log_prob(values, zmask=zmask)

    def maximise_posterior(
        self, initial_values=None, method="nelder-mead", tol=1e-4
    ):
        """Run scipy minimizer to find maximum of posterior"""

        if not initial_values:
            initial_values = np.ones(len(self.free_params)) * 0.5

        return minimize(
            self.minus_log_prob, x0=initial_values, method=method, tol=tol
        )

    def plot_p1d(
        self,
        values=None,
        plot_every_iz=1,
        residuals=False,
        plot_fname=None,
        rand_posterior=None,
        show=True,
        return_covar=False,
        print_ratio=False,
        print_chi2=True,
        return_all=False,
        collapse=False,
        plot_realizations=True,
        zmask=None,
        n_perturb=100,
        plot_panels=False,
        z_at_time=False,
        fontsize=16,
    ):
        """Plot P1D in theory vs data. If plot_every_iz >1,
        plot only few redshift bins"""

        if (zmask is not None) | (plot_realizations == False):
            n_perturb = 0

        if zmask is None:
            _data_z = self.data.z
            _data_k_kms = self.data.k_kms
        else:
            _data_z = []
            _data_k_kms = []
            for iz in range(len(self.data.z)):
                _ = np.argwhere(np.abs(zmask - self.data.z[iz]) < 1e-3)
                if len(_) != 0:
                    _data_z.append(self.data.z[iz])
                    _data_k_kms.append(self.data.k_kms[iz])
            _data_z = np.array(_data_z)

        # z at time fits or full fit
        if z_at_time is False:
            _res = self.get_p1d_kms(
                _data_z, _data_k_kms, values, return_covar=return_covar
            )
            if _res is None:
                return print("Prior out of range")
            if return_covar:
                emu_p1d, emu_cov = _res
            else:
                emu_p1d = _res

            # the sum of chi2_all may be different from chi2 due to covariance
            chi2, chi2_all = self.get_chi2(
                values=values, return_all=True, zmask=zmask
            )
        else:
            emu_p1d = []
            chi2_all = []
            for iz in range(len(_data_z)):
                _res = self.get_p1d_kms(
                    _data_z[iz],
                    _data_k_kms[iz],
                    values[iz],
                    return_covar=return_covar,
                )
                # print(iz, _data_z[iz], _res)
                if _res is None:
                    return print("Prior out of range for z = ", _data_z[iz])
                if return_covar:
                    emu_p1d.append(_res[0])
                else:
                    if len(_res) == 1:
                        emu_p1d.append(_res[0])
                    else:
                        emu_p1d.append(_res)

                _chi2, _ = self.get_chi2(
                    values=values[iz],
                    return_all=True,
                    zmask=np.array([_data_z[iz]]),
                )
                chi2_all.append(_chi2)
            chi2 = np.sum(chi2_all)
            # account for extra_data
            chi2_all = np.array([chi2_all])

        if self.extra_data is not None:
            _res = self.get_p1d_kms(
                self.extra_data.z,
                self.extra_data.k_kms,
                values,
                return_covar=return_covar,
            )
            if _res is None:
                return print("Prior out of range")
            if return_covar:
                emu_p1d_extra, emu_cov_extra = _res
            else:
                emu_p1d_extra = _res

        # if rand_posterior is not None:
        #     Nz = len(self.data.z)
        #     rand_emu = np.zeros((rand_posterior.shape[0], Nz, len(k_emu_kms)))
        #     for ii in range(rand_posterior.shape[0]):
        #         rand_emu[ii] = self.get_p1d_kms(
        #             self.data.z, k_emu_kms, rand_posterior[ii]
        #         )
        #     err_posterior = np.std(rand_emu, axis=0)

        #     if self.extra_data is not None:
        #         Nz = len(self.extra_data.z)
        #         rand_emu_extra = np.zeros(
        #             (rand_posterior.shape[0], Nz, len(k_emu_kms_extra))
        #         )
        #         for ii in range(rand_posterior.shape[0]):
        #             rand_emu_extra[ii] = self.get_p1d_kms(
        #                 self.extra_data.z, k_emu_kms_extra, rand_posterior[ii]
        #             )
        #         err_posterior_extra = np.std(rand_emu_extra, axis=0)

        if self.extra_data is None:
            if plot_panels:
                fig, ax = plt.subplots(
                    len(_data_z) // 2 + len(_data_z) % 2,
                    2,
                    figsize=(12, len(_data_z)),
                    sharex=True,
                    sharey="row",
                )
                if len(_data_z) == 1:
                    ax = [ax]
                else:
                    ax = ax.reshape(-1)
                length = 1
                if len(_data_z) % 2 != 0:
                    ax[-1].axis("off")
            else:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                length = 1
                ax = [ax]
        else:
            fig, ax = plt.subplots(2, 1, figsize=(8, 8))
            length = 2

        # figure out y range for plot
        ymin = 1e10
        ymax = -1e10

        # print chi2
        n_free_p = len(self.free_params)
        ndeg = 0
        for iz in range(len(self.data.k_kms)):
            ndeg += np.sum(self.data.Pk_kms[iz] != 0)
        if self.extra_data is not None:
            for iz in range(len(self.extra_data.k_kms)):
                ndeg += np.sum(self.extra_data.Pk_kms[iz] != 0)
        prob = chi2_scipy.sf(chi2, ndeg - n_free_p)

        if plot_panels == False:
            label = (
                r"$\chi^2=$"
                + str(np.round(chi2, 6))
                + " (ndeg="
                + str(ndeg - n_free_p)
                + ", prob="
                + str(np.round(prob * 100, 6))
                + "%)"
            )
        else:
            label = r"$\chi_\mathrm{all}^2=$" + str(np.round(chi2, 2))

        if print_chi2:
            fig.suptitle(label, fontsize=fontsize)

        out = {}

        for ii in range(length):
            if ii == 0:
                data = self.data
                emu_p1d_use = emu_p1d
                if return_covar:
                    emu_cov_use = emu_cov
                if rand_posterior is not None:
                    err_posterior_use = err_posterior
                out["zs"] = []
                out["k_kms"] = []
                out["p1d_data"] = []
                out["p1d_model"] = []
                out["p1d_err"] = []
                out["chi2"] = []
                out["prob"] = []
            else:
                data = self.extra_data
                emu_p1d_use = emu_p1d_extra
                if return_covar:
                    emu_cov_use = emu_cov_extra
                if rand_posterior is not None:
                    err_posterior_use = err_posterior_extra
                out["extra_zs"] = []
                out["extra_k_kms"] = []
                out["extra_p1d_data"] = []
                out["extra_p1d_model"] = []
                out["extra_p1d_err"] = []
                out["extra_chi2"] = []
                out["extra_prob"] = []

            full_emu_p1d = np.concatenate(emu_p1d_use)
            if n_perturb > 0:
                perturb = np.random.multivariate_normal(
                    full_emu_p1d, self.full_cov_Pk_kms, n_perturb
                )

            zs = data.z
            Nz = len(zs)

            # plot only few redshifts for clarity
            for iz in range(0, Nz, plot_every_iz):
                if zmask is not None:
                    indemu = np.argwhere(np.abs(zmask - zs[iz]) < 1e-3)[:, 0]
                    if len(indemu) == 0:
                        continue
                    else:
                        indemu = indemu[0]
                else:
                    indemu = iz
                # access data for this redshift
                z = zs[iz]
                k_kms = data.k_kms[iz]
                p1d_data = data.Pk_kms[iz]
                p1d_cov = self.cov_Pk_kms[iz]
                p1d_err = np.sqrt(np.diag(p1d_cov))
                p1d_theory = emu_p1d_use[indemu]

                if rand_posterior is None:
                    if return_covar:
                        cov_theory = emu_cov_use[iz]
                        err_theory = np.sqrt(np.diag(cov_theory))
                else:
                    err_theory = err_posterior_use[iz]

                # plot everything
                if Nz > 1:
                    col = plt.cm.jet(iz / (Nz - 1))
                    if collapse:
                        yshift = 0
                    else:
                        yshift = 4 * iz / (Nz - 1)
                else:
                    col = "C0"
                    yshift = 0

                if plot_panels:
                    col = "C0"
                    yshift = 0

                if residuals:
                    if plot_panels:
                        axs = ax[iz]
                        yshift = 0
                    else:
                        axs = ax[ii]

                    axs.tick_params(
                        axis="both", which="major", labelsize=fontsize - 4
                    )
                    # shift data in y axis for clarity
                    axs.errorbar(
                        k_kms,
                        p1d_data / p1d_theory + yshift,
                        color=col,
                        yerr=p1d_err / p1d_theory,
                        fmt="o",
                        ms="4",
                        label="z=" + str(np.round(z, 2)),
                    )

                    ind = self.data.full_zs == z
                    for kk in range(n_perturb):
                        axs.plot(
                            k_kms,
                            perturb[kk, ind] / p1d_theory + yshift,
                            color=col,
                            alpha=0.025,
                        )

                    # print chi2
                    xpos = k_kms[0]
                    ndeg = np.sum(p1d_data != 0)
                    prob = chi2_scipy.sf(chi2_all[ii, iz], ndeg - n_free_p)
                    label = (
                        r"$\chi^2=$"
                        + str(np.round(chi2_all[ii, iz], 2))
                        + " (ndeg="
                        + str(ndeg - n_free_p)
                        + ", prob="
                        + str(np.round(prob * 100, 2))
                        + "%)"
                    )

                    if print_chi2:
                        if plot_panels == False:
                            ypos = 0.75 + yshift
                            axs.text(xpos, ypos, label, fontsize=10)

                    if print_ratio:
                        print(p1d_data / p1d_theory)
                    ymin = min(ymin, min(p1d_data / p1d_theory + yshift))
                    ymax = max(ymax, max(p1d_data / p1d_theory + yshift))

                    axs.axhline(1, color="k", linestyle=":", alpha=0.5)

                    if return_covar | (rand_posterior is not None):
                        axs.fill_between(
                            k_kms,
                            (p1d_theory + err_theory) / p1d_theory + yshift,
                            (p1d_theory - err_theory) / p1d_theory + yshift,
                            alpha=0.35,
                            color=col,
                        )
                else:
                    ax[ii].errorbar(
                        k_kms,
                        p1d_data * k_kms / np.pi,
                        color=col,
                        yerr=p1d_err * k_kms / np.pi,
                        fmt="o",
                        ms="4",
                        label="z=" + str(np.round(z, 2)),
                    )

                    ind = self.data.full_zs == z
                    for kk in range(n_perturb):
                        ax[ii].plot(
                            k_kms,
                            perturb[kk, ind] * k_kms / np.pi,
                            color=col,
                            alpha=0.05,
                        )

                    # print chi2
                    xpos = k_kms[-1] + 0.001
                    ypos = (p1d_theory * k_kms / np.pi)[-1]
                    ndeg = np.sum(p1d_data != 0)
                    prob = chi2_scipy.sf(chi2_all[ii, iz], ndeg - n_free_p)
                    label = (
                        r"$\chi^2=$"
                        + str(np.round(chi2_all[ii, iz], 2))
                        + " ("
                        + str(ndeg - n_free_p)
                        + ", "
                        + str(np.round(prob * 100, 2))
                        + "%)"
                    )
                    if print_chi2:
                        ax[ii].text(xpos, ypos, label, fontsize=8)

                    ax[ii].plot(
                        k_kms,
                        (p1d_theory * k_kms) / np.pi,
                        color=col,
                        linestyle="dashed",
                    )
                    if return_covar | (rand_posterior is not None):
                        ax[ii].fill_between(
                            k_kms,
                            (p1d_theory + err_theory) * k_kms / np.pi,
                            (p1d_theory - err_theory) * k_kms / np.pi,
                            alpha=0.35,
                            color=col,
                        )
                    ymin = min(ymin, min(p1d_data * k_kms / np.pi))
                    ymax = max(ymax, max(p1d_data * k_kms / np.pi))

                if residuals & plot_panels:
                    axs.legend(loc="upper right")
                    ymin = 1 - min((p1d_data - p1d_err) / p1d_theory + yshift)
                    ymax = 1 - max((p1d_data + p1d_err) / p1d_theory + yshift)
                    y2plot = 1.05 * np.max([np.abs(ymin), np.abs(ymax)])
                    if iz % 2 == 1:
                        # y2plot0 = ax[iz - 1].get_ylim()[1] - 1
                        # print(y2plot, y2plot0)
                        # y2plot = np.max([y2plot, y2plot0])
                        axs.set_ylim(1 - y2plot, 1 + y2plot)
                    elif iz == len(zs) - 1:
                        axs.set_ylim(1 - y2plot, 1 + y2plot)

                    if print_chi2:
                        axs.text(
                            0.05,
                            0.05,
                            label,
                            fontsize=10,
                            transform=axs.transAxes,
                        )

                if ii == 0:
                    out["zs"].append(z)
                    out["k_kms"].append(k_kms)
                    out["p1d_data"].append(p1d_data)
                    out["p1d_model"].append(p1d_theory)
                    out["p1d_err"].append(p1d_err)
                    out["chi2"].append(chi2_all[ii, iz])
                    out["prob"].append(prob)
                else:
                    out["extra_zs"].append(z)
                    out["extra_k_kms"].append(k_kms)
                    out["extra_p1d_data"].append(p1d_data)
                    out["extra_p1d_model"].append(p1d_theory)
                    out["extra_p1d_err"].append(p1d_err)
                    out["extra_chi2"].append(chi2_all[ii, iz])
                    out["extra_prob"].append(prob)

            # ax[ii].plot(k_kms[0], 1, linestyle="-", label="Data", color="k")
            ax[ii].plot(k_kms[0], 1, linestyle="--", label="Fit", color="k")
            if residuals:
                if plot_panels == False:
                    ax[ii].legend()
            else:
                ax[ii].legend(loc="lower right", ncol=4, fontsize=fontsize - 2)

            # ax[ii].set_xlim(min(k_kms[0]) - 0.001, max(k_kms[-1]) + 0.001)
            # if plot_panels == False:
            # ax[ii].set_xlabel(r"$k_\parallel$ [s/km]")
            # else:
            # ax[-1].set_xlabel(r"$k_\parallel$ [s/km]")

            if residuals:
                if plot_panels == False:
                    ax[ii].set_ylabel(
                        r"$P_{\rm 1D}^{\rm data}/P_{\rm 1D}^{\rm fit}$",
                        fontsize=fontsize,
                    )
                    ax[ii].set_ylim(ymin - 0.3, ymax + 0.3)
            else:
                ax[ii].set_ylim(0.8 * ymin, 1.3 * ymax)
                ax[ii].set_yscale("log")
                ax[ii].set_ylabel(
                    r"$k_\parallel \, P_{\rm 1D}(z, k_\parallel) / \pi$",
                    fontsize=fontsize,
                )

        fig.supxlabel(r"$k_\parallel$ [s/km]", fontsize=fontsize)
        fig.supylabel(
            r"$P_{\rm 1D}^{\rm data}/P_{\rm 1D}^{\rm fit}$",
            fontsize=fontsize,
        )

        plt.tight_layout()
        # plt.savefig("test.pdf")
        if plot_fname is not None:
            plt.savefig(plot_fname + ".pdf")
            plt.savefig(plot_fname + ".png")
        else:
            if show:
                plt.show()

        if return_all:
            return out
        else:
            return

    def plot_p1d_errors(
        self,
        values=None,
        plot_fname=None,
        show=True,
        zmask=None,
        z_at_time=False,
        fontsize=16,
    ):
        """Plot P1D in theory vs data. If plot_every_iz >1,
        plot only few redshift bins"""

        import scipy.stats as stats

        if zmask is None:
            _data_z = self.data.z
            _data_k_kms = self.data.k_kms
        else:
            _data_z = []
            _data_k_kms = []
            for iz in range(len(self.data.z)):
                _ = np.argwhere(np.abs(zmask - self.data.z[iz]) < 1e-3)
                if len(_) != 0:
                    _data_z.append(self.data.z[iz])
                    _data_k_kms.append(self.data.k_kms[iz])
            _data_z = np.array(_data_z)

        # z at time fits or full fit
        if z_at_time is False:
            _res = self.get_p1d_kms(
                _data_z, _data_k_kms, values, return_covar=False
            )
            if _res is None:
                return print("Prior out of range")
            emu_p1d = _res

            # the sum of chi2_all may be different from chi2 due to covariance
            chi2, chi2_all = self.get_chi2(
                values=values, return_all=True, zmask=zmask
            )
        else:
            emu_p1d = []
            for iz in range(len(_data_z)):
                _res = self.get_p1d_kms(
                    _data_z[iz],
                    _data_k_kms[iz],
                    values[iz],
                    return_covar=False,
                )
                # print(iz, _data_z[iz], _res)
                if _res is None:
                    return print("Prior out of range for z = ", _data_z[iz])
                if len(_res) == 1:
                    emu_p1d.append(_res[0])
                else:
                    emu_p1d.append(_res)

        if self.extra_data is not None:
            _res = self.get_p1d_kms(
                self.extra_data.z,
                self.extra_data.k_kms,
                values,
                return_covar=return_covar,
            )
            if _res is None:
                return print("Prior out of range")
            if return_covar:
                emu_p1d_extra, emu_cov_extra = _res
            else:
                emu_p1d_extra = _res

        fig, ax = plt.subplots(
            len(_data_z) // 2 + len(_data_z) % 2,
            2,
            figsize=(12, len(_data_z)),
            sharex=True,
            sharey=True,
        )
        if len(_data_z) == 1:
            ax = [ax]
        else:
            ax = ax.reshape(-1)
        length = 1
        # if (len(_data_z) % 2 + 1) != 0:
        #     ax[-1].axis("off")

        out = {}
        bins = np.linspace(-5, 5, 50)
        out["bins"] = bins

        data = self.data
        emu_p1d_use = emu_p1d
        out["zs"] = []
        out["(d-m)/err"] = []

        mu = 0
        sigma = 1
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)

        zs = data.z
        Nz = len(zs)

        for iz in range(0, Nz):
            if zmask is not None:
                indemu = np.argwhere(np.abs(zmask - zs[iz]) < 1e-3)[:, 0]
                if len(indemu) == 0:
                    continue
                else:
                    indemu = indemu[0]
            else:
                indemu = iz

            # access data for this redshift
            z = zs[iz]
            k_kms = data.k_kms[iz]
            p1d_data = data.Pk_kms[iz]
            p1d_cov = self.cov_Pk_kms[iz]
            p1d_err = np.sqrt(np.diag(p1d_cov))
            p1d_theory = emu_p1d_use[indemu]

            dme = (p1d_data - p1d_theory) / p1d_err

            ax[iz].tick_params(
                axis="both", which="major", labelsize=fontsize - 4
            )
            ax[iz].hist(
                dme,
                label="z=" + str(np.round(z, 2)),
                bins=bins,
                color="C0",
                density=True,
            )

            ax[iz].plot(x, stats.norm.pdf(x, mu, sigma), color="C1")

            out["zs"].append(z)
            out["(d-m)/err"].append(dme)

            ax[iz].legend()

        dme = np.concatenate(out["(d-m)/err"])
        print("dme, mean", np.mean(dme))
        print("dme, std", np.std(dme))
        ax[iz + 1].hist(dme, label="All", bins=bins, density=True)
        ax[iz + 1].plot(x, stats.norm.pdf(x, mu, sigma), color="C1")
        ax[iz + 1].legend()

        fig.supxlabel(r"(d-m)/err", fontsize=fontsize)
        fig.supylabel(r"$PDF$", fontsize=fontsize)

        plt.tight_layout()
        # plt.savefig("test.pdf")
        if plot_fname is not None:
            plt.savefig(plot_fname + ".pdf")
            plt.savefig(plot_fname + ".png")
        else:
            if show:
                plt.show()

        return out

    # def overplot_emulator_calls(
    #     self,
    #     param_1,
    #     param_2,
    #     values=None,
    #     tau_scalings=True,
    #     temp_scalings=True,
    # ):
    #     """For parameter pair (param1,param2), overplot emulator calls
    #     with values stored in archive, color coded by redshift"""

    #     # mask post-process scalings (optional)
    #     emu_data = self.theory.emulator.archive.data
    #     Nemu = len(emu_data)
    #     if not tau_scalings:
    #         mask_tau = [x["scale_tau"] == 1.0 for x in emu_data]
    #     else:
    #         mask_tau = [True] * Nemu
    #     if not temp_scalings:
    #         mask_temp = [
    #             (x["scale_T0"] == 1.0) & (x["scale_gamma"] == 1.0)
    #             for x in emu_data
    #         ]
    #     else:
    #         mask_temp = [True] * Nemu

    #     # figure out values of param_1,param_2 in archive
    #     emu_1 = np.array(
    #         [
    #             emu_data[i][param_1]
    #             for i in range(Nemu)
    #             if (mask_tau[i] & mask_temp[i])
    #         ]
    #     )
    #     emu_2 = np.array(
    #         [
    #             emu_data[i][param_2]
    #             for i in range(Nemu)
    #             if (mask_tau[i] & mask_temp[i])
    #         ]
    #     )

    #     # translate sampling point (in unit cube) to parameter values
    #     if values is not None:
    #         like_params = self.parameters_from_sampling_point(values)
    #     else:
    #         like_params = []
    #     emu_calls = self.theory.get_emulator_calls(like_params=like_params)
    #     # figure out values of param_1,param_2 called
    #     call_1 = [emu_call[param_1] for emu_call in emu_calls]
    #     call_2 = [emu_call[param_2] for emu_call in emu_calls]

    #     # overplot
    #     zs = self.data.z
    #     emu_z = np.array(
    #         [
    #             emu_data[i]["z"]
    #             for i in range(Nemu)
    #             if (mask_tau[i] & mask_temp[i])
    #         ]
    #     )
    #     zmin = min(min(emu_z), min(zs))
    #     zmax = max(max(emu_z), max(zs))
    #     plt.scatter(emu_1, emu_2, c=emu_z, s=1, vmin=zmin, vmax=zmax)
    #     plt.scatter(call_1, call_2, c=zs, s=50, vmin=zmin, vmax=zmax)
    #     cbar = plt.colorbar()
    #     cbar.set_label("Redshift", labelpad=+1)
    #     plt.xlabel(param_1)
    #     plt.ylabel(param_2)
    #     plt.show()

    #     return

    def plot_igm(
        self,
        cloud=False,
        free_params=None,
        save_directory=None,
        stat_best_fit="mle",
        zmask=None,
    ):
        """Plot IGM histories"""

        # true IGM parameters
        if self.truth is not None:
            pars_true = {}
            pars_true["z"] = self.truth["igm"]["z"]
            pars_true["tau_eff"] = self.truth["igm"]["tau_eff"]
            pars_true["gamma"] = self.truth["igm"]["gamma"]
            pars_true["sigT_kms"] = self.truth["igm"]["sigT_kms"]
            pars_true["kF_kms"] = self.truth["igm"]["kF_kms"]

        pars_fid = {}
        pars_fid["z"] = self.fid["igm"]["z"]
        pars_fid["tau_eff"] = self.fid["igm"]["tau_eff"]
        pars_fid["gamma"] = self.fid["igm"]["gamma"]
        pars_fid["sigT_kms"] = self.fid["igm"]["sigT_kms"]
        pars_fid["kF_kms"] = self.fid["igm"]["kF_kms"]

        if free_params is not None:
            if zmask is not None:
                zs = zmask
            else:
                zs = self.fid["igm"]["z"]
            pars_test = {}
            pars_test["z"] = zs
            pars_test["tau_eff"] = self.theory.model_igm.F_model.get_tau_eff(
                zs, like_params=free_params
            )
            pars_test["gamma"] = self.theory.model_igm.T_model.get_gamma(
                zs, like_params=free_params
            )
            pars_test["sigT_kms"] = self.theory.model_igm.T_model.get_sigT_kms(
                zs, like_params=free_params
            )
            pars_test["kF_kms"] = self.theory.model_igm.P_model.get_kF_kms(
                zs, like_params=free_params
            )

        fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
        ax = ax.reshape(-1)

        arr_labs = ["tau_eff", "gamma", "sigT_kms", "kF_kms"]
        latex_labs = [
            r"$\tau_\mathrm{eff}$",
            r"$\gamma$",
            r"$\sigma_T$",
            r"$k_F$",
        ]

        for ii in range(len(arr_labs)):
            if self.truth is not None:
                _ = pars_true[arr_labs[ii]] != 0
                ax[ii].plot(
                    pars_true["z"][_],
                    pars_true[arr_labs[ii]][_],
                    "C0:o",
                    alpha=0.75,
                    label="true",
                )

            _ = pars_fid[arr_labs[ii]] != 0
            ax[ii].plot(
                pars_fid["z"][_],
                pars_fid[arr_labs[ii]][_],
                "C1s--",
                label="fiducial",
                alpha=0.75,
                lw=3,
            )

            if free_params is not None:
                _ = pars_test[arr_labs[ii]] != 0
                ax[ii].plot(
                    pars_test["z"][_],
                    pars_test[arr_labs[ii]][_],
                    "C2o-.",
                    label="fit",
                    alpha=0.75,
                    lw=3,
                )

            if cloud:
                for sim_label in self.theory.emu_igm_all:
                    if is_number_string(sim_label[-1]) == False:
                        continue

                    _ = np.argwhere(
                        self.theory.emu_igm_all[sim_label][arr_labs[ii]] != 0
                    )[:, 0]
                    if len(_) > 0:
                        ax[ii].scatter(
                            self.theory.emu_igm_all[sim_label]["z"][_],
                            self.theory.emu_igm_all[sim_label][arr_labs[ii]][_],
                            marker=".",
                            color="black",
                            alpha=0.05,
                        )

            ax[ii].set_ylabel(latex_labs[ii])
            if ii == 0:
                ax[ii].set_yscale("log")
                ax[ii].legend()

            if (ii == 2) | (ii == 3):
                ax[ii].set_xlabel(r"$z$")

        plt.tight_layout()

        if save_directory is not None:
            name = os.path.join(
                save_directory, "IGM_histories_" + stat_best_fit
            )
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_cov_terms(self, save_directory=None):
        npanels = int(np.round(np.sqrt(len(self.cov_Pk_kms))))
        fig, ax = plt.subplots(
            npanels + 1, npanels, sharex=True, sharey=True, figsize=(10, 8)
        )
        ax = ax.reshape(-1)
        for ii in range(len(self.cov_Pk_kms)):
            cov_stat = np.diag(self.data.covstat_Pk_kms[ii])
            cov_syst = np.diag(self.data.cov_Pk_kms[ii]) - cov_stat
            cov_emu = np.diag(self.covemu_Pk_kms[ii])
            cov_tot = np.diag(self.cov_Pk_kms[ii])
            ax[ii].plot(
                self.data.k_kms[ii], cov_stat / cov_tot, label=r"$x$ = Stat"
            )
            ax[ii].plot(
                self.data.k_kms[ii], cov_syst / cov_tot, label=r"$x$ = Syst"
            )
            ax[ii].plot(
                self.data.k_kms[ii], cov_emu / cov_tot, label=r"$x$ = Emu"
            )
            ax[ii].text(0.0, 0.1, "z=" + str(self.data.z[ii]))
        if len(ax) > len(self.cov_Pk_kms):
            for ii in range(len(self.cov_Pk_kms), len(ax)):
                ax[ii].axis("off")
        ax[0].legend()
        fig.supxlabel(r"$k\,[\mathrm{km}^{-1}\mathrm{s}]$")
        fig.supylabel(r"$\sigma^2_x/\sigma^2_\mathrm{total}$")
        plt.tight_layout()

        if save_directory is not None:
            name = os.path.join(save_directory, "cov_terms")
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_cov_to_pk(self, use_pk_smooth=True, save_directory=None):
        npanels = int(np.round(np.sqrt(len(self.cov_Pk_kms))))

        fig, ax = plt.subplots(
            npanels + 1, npanels, sharex=True, sharey="row", figsize=(10, 8)
        )
        ax = ax.reshape(-1)
        for ii in range(len(self.cov_Pk_kms)):
            cov_stat = np.diag(self.data.covstat_Pk_kms[ii])
            cov_syst = np.diag(self.data.cov_Pk_kms[ii]) - cov_stat
            cov_emu = np.diag(self.covemu_Pk_kms[ii])
            cov_tot = np.diag(self.cov_Pk_kms[ii])
            if use_pk_smooth:
                pk = self.data.Pksmooth_kms[ii].copy()
            else:
                pk = self.data.Pk_kms[ii].copy()
            ax[ii].plot(
                self.data.k_kms[ii], np.sqrt(cov_stat) / pk, label=r"$x$ = Stat"
            )
            ax[ii].plot(
                self.data.k_kms[ii], np.sqrt(cov_syst) / pk, label=r"$x$ = Syst"
            )
            ax[ii].plot(
                self.data.k_kms[ii], np.sqrt(cov_emu) / pk, label=r"$x$ = Emu"
            )
            ax[ii].plot(
                self.data.k_kms[ii], np.sqrt(cov_tot) / pk, label=r"$x$ = Total"
            )
            ax[ii].text(
                0.2,
                0.97,
                "z=" + str(self.data.z[ii]),
                ha="right",
                va="top",
                transform=ax[ii].transAxes,
            )
        if len(ax) > len(self.cov_Pk_kms):
            for ii in range(len(self.cov_Pk_kms), len(ax)):
                ax[ii].axis("off")
        ax[0].legend(ncols=2)
        fig.supxlabel(r"$k\,[\mathrm{km}^{-1}\mathrm{s}]$")
        fig.supylabel(r"$\sigma_x/P_\mathrm{1D}$")
        plt.tight_layout()

        if save_directory is not None:
            name = os.path.join(save_directory, "cov_to_pk")
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_correlation_matrix(self, save_directory=None):
        def correlation_from_covariance(covariance):
            v = np.sqrt(np.diag(covariance))
            outer_v = np.outer(v, v)
            correlation = covariance / outer_v
            correlation[covariance == 0] = 0
            return correlation

        def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) > 0)

        plt.imshow(correlation_from_covariance(self.full_cov_Pk_kms))
        plt.colorbar()

        if save_directory is not None:
            name = os.path.join(save_directory, "correlation")
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_hull_fid(self):
        emu_call, M_of_z = self.theory.get_emulator_calls(self.data.z)
        p1 = np.zeros(
            (
                self.theory.hull.nz,
                len(self.theory.hull.params),
            )
        )
        for jj, key in enumerate(self.theory.hull.params):
            p1[:, jj] = emu_call[key]

        self.theory.hull.plot_hulls(p1)
