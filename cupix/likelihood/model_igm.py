import os
import lace
import numpy as np
from cup1d.nuisance import (
    mean_flux_model,
    thermal_model,
    pressure_model,
    mean_flux_model_chunks,
)
from cup1d.utils.utils import is_number_string


class IGM(object):
    """Contains all IGM models"""

    def __init__(
        self,
        free_param_names=None,
        z_pivot=3,
        F_model=None,
        T_model=None,
        P_model=None,
        fid_sim_igm_mF="mpg_central",
        fid_sim_igm_T="mpg_central",
        fid_sim_igm_kF="mpg_central",
        fid_val_par_mF=[0, 0],
        fid_val_par_sigT=[0, 0],
        fid_val_par_gamma=[0, 0],
        fid_val_par_kF=[0, 0],
        mF_model_type="pivot",
        emu_suite="mpg",
        type_priors="hc",
        emu_igm_params=["mF", "sigT_Mpc", "gamma", "kF_Mpc"],
        Gauss_priors=None,
        # set_metric=False,
    ):
        # load fiducial IGM history (used for fitting)
        self.fid_sim_igm_mF = fid_sim_igm_mF
        self.fid_sim_igm_T = fid_sim_igm_T
        self.fid_sim_igm_kF = fid_sim_igm_kF
        self.z_pivot = z_pivot

        fid_igm = self.get_igm(
            sim_igm_mF=fid_sim_igm_mF,
            sim_igm_T=fid_sim_igm_T,
            sim_igm_kF=fid_sim_igm_kF,
        )

        if emu_suite == "mpg":
            back_igm = self.get_igm(
                sim_igm_mF="mpg_central",
                sim_igm_T="mpg_central",
                sim_igm_kF="mpg_central",
            )
        elif emu_suite == "nyx":
            back_igm = self.get_igm(
                sim_igm_mF="nyx_central",
                sim_igm_T="nyx_central",
                sim_igm_kF="nyx_central",
            )
        else:
            raise ValueError("emu_suite must be 'mpg' or 'nyx'")

        self.set_priors(fid_igm, emu_suite=emu_suite, type_priors=type_priors)

        # setup fiducial IGM models
        if F_model is not None:
            self.F_model = F_model
        else:
            if mF_model_type == "pivot":
                self.F_model = mean_flux_model.MeanFluxModel(
                    free_param_names=free_param_names,
                    fid_igm=fid_igm,
                    fid_value=fid_val_par_mF,
                    z_tau=z_pivot,
                    priors=self.priors,
                    Gauss_priors=Gauss_priors,
                )
            elif mF_model_type == "chunks":
                self.F_model = mean_flux_model_chunks.MeanFluxModelChunks(
                    free_param_names=free_param_names,
                    fid_igm=fid_igm,
                    Gauss_priors=Gauss_priors,
                )
            else:
                raise ValueError("mF_model_type must be 'scaling' or 'chunks'")
        if T_model:
            self.T_model = T_model
        else:
            self.T_model = thermal_model.ThermalModel(
                free_param_names=free_param_names,
                fid_igm=fid_igm,
                fid_value_sigT=fid_val_par_sigT,
                fid_value_gamma=fid_val_par_gamma,
                z_T=z_pivot,
                priors=self.priors,
                back_igm=back_igm,
                Gauss_priors=Gauss_priors,
            )

        if P_model:
            self.P_model = P_model
        else:
            self.P_model = pressure_model.PressureModel(
                free_param_names=free_param_names,
                fid_igm=fid_igm,
                fid_value=fid_val_par_kF,
                z_kF=z_pivot,
                priors=self.priors,
                back_igm=back_igm,
                Gauss_priors=Gauss_priors,
            )

    def set_fid_igm(self, zs):
        self.fid_igm = {}
        self.fid_igm["z"] = zs
        self.fid_igm["tau_eff"] = self.F_model.get_tau_eff(zs)
        self.fid_igm["gamma"] = self.T_model.get_gamma(zs)
        self.fid_igm["sigT_kms"] = self.T_model.get_sigT_kms(zs)
        self.fid_igm["kF_kms"] = self.P_model.get_kF_kms(zs)

    def get_igm(self, sim_igm_mF=None, sim_igm_T=None, sim_igm_kF=None):
        """Load IGM history"""

        repo = os.path.dirname(lace.__path__[0])
        
        try:
            fname = os.path.join(
            repo, "data", "sim_suites", "Australia20", "IGM_histories.npy"
        )
            self.igm_hist_mpg = np.load(fname, allow_pickle=True).item()
        except:
            raise ValueError(
                fname
                + " not found. You can produce it using LaCE"
                + r" script save_mpg_IGM.py"
            )

        # try:
        #     fname = os.path.join(os.environ["NYX_PATH"], "IGM_histories.npy")
        #     self.igm_hist_nyx = np.load(fname, allow_pickle=True).item()
        # except:
        #     raise ValueError(
        #         fname
        #         + " not found. You can produce it using LaCE"
        #         + r" script save_nyx_IGM.py"
        #     )

        sim_igms = [sim_igm_mF, sim_igm_T, sim_igm_kF]

        igms_return = {}
        for ii, sim_igm in enumerate(sim_igms):
            if sim_igm[:3] == "mpg":
                igm_hist = self.igm_hist_mpg
            elif sim_igm[:3] == "nyx":
                igm_hist = self.igm_hist_nyx
            elif sim_igm in self.igm_hist_nyx:
                igm_hist = self.igm_hist_nyx
            elif sim_igm == "kF_both":
                res_fit = np.array([0.00078134, 0.00028125, 0.15766722])
                zz = np.linspace(1.8, 6, 100)
                igms_return["z_kF"] = zz
                igms_return["kF_kms"] = np.poly1d(res_fit)(zz)
                continue
            else:
                ValueError("sim_igm must be 'mpg' or 'nyx'")

            if sim_igm not in igm_hist:
                igm_return = igm_hist[sim_igm + "_0"]
            else:
                igm_return = igm_hist[sim_igm]

            if ii == 0:
                igms_return["z_tau"] = igm_return["z"]
                igms_return["tau_eff"] = igm_return["tau_eff"]
                igms_return["mF"] = igm_return["mF"]
            elif ii == 1:
                igms_return["gamma"] = igm_return["gamma"]
                igms_return["sigT_kms"] = igm_return["sigT_kms"]
                igms_return["sigT_Mpc"] = igm_return["sigT_Mpc"]
                igms_return["z_T"] = igm_return["z"]
            elif ii == 2:
                igms_return["kF_Mpc"] = igm_return["kF_Mpc"]
                igms_return["kF_kms"] = igm_return["kF_kms"]
                igms_return["z_kF"] = igm_return["z"]

            # important for nyx simulations, not all have kF
            # if so, we assign the values for nyx_central
            if np.sum(igm_return["kF_kms"] != 0) == 0:
                igms_return["kF_Mpc"] = igm_hist["nyx_central"]["kF_Mpc"]
                igms_return["kF_kms"] = igm_hist["nyx_central"]["kF_kms"]

        return igms_return

    def set_priors(self, fid_igm, emu_suite="mpg", type_priors="hc"):
        """Set priors for all IGM models

        This is only important for giving the minimizer and the sampler a uniform
        prior that it is not too broad. The metric below takes care of the real priors
        """

        if type_priors == "hc":
            percent = 95
        elif type_priors == "data":
            percent = 68
        else:
            raise ValueError("type_priors must be 'hc' or 'data'")

        if emu_suite == "mpg":
            all_igm = self.igm_hist_mpg
        elif emu_suite == "nyx":
            all_igm = self.igm_hist_nyx
        else:
            ValueError("sim_igm must be 'mpg' or 'nyx'")

        self.priors = {}
        for par in fid_igm:
            if (
                (par == "val_scaling")
                | (par == "z_tau")
                | (par == "z_T")
                | (par == "z_kF")
            ):
                continue

            if (par == "mF") | (par == "tau_eff"):
                z = fid_igm["z_tau"]
            elif (par == "kF_Mpc") | (par == "kF_kms"):
                z = fid_igm["z_kF"]
            elif (par == "gamma") | (par == "sigT_Mpc") | (par == "sigT_kms"):
                z = fid_igm["z_T"]

            res_div = np.zeros((len(all_igm), 2))
            for ii, sim in enumerate(all_igm):
                if (sim in ["accel2"]) | (np.char.isnumeric(sim[-1]) == False):
                    continue

                string_split = sim.split("_")
                sim_label = string_split[0] + "_" + string_split[1]
                if is_number_string(sim_label[-1]) == False:
                    continue

                try:
                    _ = np.argwhere(
                        np.isfinite(fid_igm[par])
                        & (fid_igm[par] != 0)
                        & (all_igm[sim][par] != 0)
                    )[:, 0]
                except:
                    continue
                if len(_) == 0:
                    continue
                res_div[ii, 0] = np.abs(
                    np.max(all_igm[sim][par][_] / fid_igm[par][_])
                )
                res_div[ii, 1] = np.abs(
                    np.min(all_igm[sim][par][_] / fid_igm[par][_])
                )

            _ = np.argwhere(
                np.isfinite(res_div[:, 0])
                & (res_div[:, 0] != 0)
                & (res_div[:, 0] != 1)
            )[:, 0]
            if len(_) == 0:
                print("no good points for ", par)
                self.priors[par] = [[-1, 1], [-1, 1]]
                continue
            y0_max = np.abs(np.log(np.percentile(res_div[_, 0], percent)))
            _ = np.argwhere(
                np.isfinite(res_div[:, 1])
                & (res_div[:, 1] != 0)
                & (res_div[:, 1] != 1)
            )[:, 0]
            y0_min = np.abs(np.log(np.percentile(1 / res_div[_, 1], percent)))
            y0_cen = 0.5 * (y0_max + y0_min)
            y1 = y0_cen / np.log((1 + z.max()) / (1 + self.z_pivot))
            self.priors[par] = [
                [-y1 * 2, y1 * 2],
                [-y0_min * 1.05, y0_max * 1.05],
            ]

    # def set_metric(self, emu_igm_params, tol_factor=95):
    #     # get all individual points separately

    #     all_points = {}
    #     for par in emu_igm_params:
    #         if par not in ["Delta2_p", "n_p", "alpha_p"]:
    #             all_points[par] = []

    #     for key in self.all_igm:
    #         if key[4].isdigit():
    #             # distance between tau scalings for mpg is too small
    #             if (key[:3] == "mpg") and (key[-1] != "0"):
    #                 continue
    #             for par in all_points:
    #                 ind_use = np.argwhere(self.all_igm[key][par] != 0)[:, 0]
    #                 all_points[par].append(self.all_igm[key][par][ind_use])

    #     for key in all_points:
    #         all_points[key] = np.concatenate(all_points[key])

    #     # compute the maximum distance between training points
    #     min_dist = {}

    #     # get closest point to each IGM point
    #     for key in all_points:
    #         npoints = all_points[key].shape[0]
    #         min_dist[key] = np.zeros(npoints)
    #         for ii in range(npoints):
    #             dist = np.abs(all_points[key][ii] - all_points[key])
    #             _ = dist != 0
    #             min_dist[key][ii] = dist[_].min()

    #     # get most distant of closest points
    #     max_dist = {}
    #     for key in min_dist:
    #         max_dist[key] = min_dist[key].max()

    #     # define function to get normalizer distance from new points
    #     def metric_par(p0):
    #         dist = (
    #             ((p0["mF"] - all_points["mF"]) / max_dist["mF"]) ** 2
    #             + (
    #                 (p0["sigT_Mpc"] - all_points["sigT_Mpc"])
    #                 / max_dist["sigT_Mpc"]
    #             )
    #             ** 2
    #             + ((p0["gamma"] - all_points["gamma"]) / max_dist["gamma"]) ** 2
    #             + ((p0["kF_Mpc"] - all_points["kF_Mpc"]) / max_dist["kF_Mpc"])
    #             ** 2
    #         )
    #         return np.sqrt(dist)

    #     # find maximum normalized distance between training points
    #     dist_norm = np.zeros(npoints)

    #     for ii in range(npoints):
    #         p0 = {}
    #         for key in all_points:
    #             p0[key] = all_points[key][ii]
    #         res = metric_par(p0)
    #         _ = res != 0
    #         dist_norm[ii] = res[_].min()

    #     # max_dist_norm = dist_norm.max() * tol_factor
    #     max_dist_norm = np.percentile(dist_norm, tol_factor)

    #     def metric_par(p0):
    #         dist = (
    #             ((p0["mF"] - all_points["mF"]) / max_dist["mF"]) ** 2
    #             + (
    #                 (p0["sigT_Mpc"] - all_points["sigT_Mpc"])
    #                 / max_dist["sigT_Mpc"]
    #             )
    #             ** 2
    #             + ((p0["gamma"] - all_points["gamma"]) / max_dist["gamma"]) ** 2
    #             + ((p0["kF_Mpc"] - all_points["kF_Mpc"]) / max_dist["kF_Mpc"])
    #             ** 2
    #         )
    #         return np.sqrt(dist.min()) / max_dist_norm

    # return metric_par
