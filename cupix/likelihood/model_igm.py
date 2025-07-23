import os
import lace
import numpy as np
from cupix.nuisance.mean_flux_class import MeanFlux
from cupix.nuisance.pressure_class import Pressure
from cupix.nuisance.thermal_class import Thermal
from cupix.utils.utils import is_number_string


class IGM(object):
    """Contains all IGM models"""

    def __init__(
        self,
        free_param_names=None,
        pars_igm=None,
        F_model=None,
        T_model=None,
        P_model=None,
    ):
        # set simulation from which we get fiducial IGM history
        for key in ["mF", "T", "kF"]:
            lab = "label_" + key
            if lab in pars_igm:
                setattr(self, "fid_sim_igm_" + key, pars_igm[lab])
            else:
                setattr(self, "fid_sim_igm_" + key, "mpg_central")

        if "Gauss_priors" in pars_igm:
            Gauss_priors = pars_igm["Gauss_priors"]
        else:
            Gauss_priors = None

        prop_coeffs = {}
        fid_vals = {}
        for key in ["tau_eff", "gamma", "sigT_kms", "kF_kms"]:
            if key in pars_igm:
                fid_vals[key] = pars_igm[key]
            for key2 in ["otype", "ztype", "znodes"]:
                key3 = key + "_" + key2
                if key3 in pars_igm:
                    prop_coeffs[key3] = pars_igm[key3]
                else:
                    if key3 == "tau_eff_otype":
                        prop_coeffs[key3] = "exp"
                    else:
                        if key3.endswith("otype"):
                            prop_coeffs[key3] = "const"
                        elif key3.endswith("ztype"):
                            prop_coeffs[key3] = "pivot"

        if "priors" in pars_igm:
            type_priors = pars_igm["priors"]
        else:
            type_priors = "hc"

        fid_igm = self.get_igm(
            sim_igm_mF=self.fid_sim_igm_mF,
            sim_igm_T=self.fid_sim_igm_T,
            sim_igm_kF=self.fid_sim_igm_kF,
        )

        self.set_priors(fid_igm, prop_coeffs, type_priors=type_priors)

        self.models = {
            "F_model": F_model,
            "T_model": T_model,
            "P_model": P_model,
        }

        for key in self.models:
            if self.models[key] is None:
                if key == "F_model":
                    model = MeanFlux
                elif key == "T_model":
                    model = Thermal
                elif key == "P_model":
                    model = Pressure

                self.models[key] = model(
                    free_param_names=free_param_names,
                    fid_igm=fid_igm,
                    fid_vals=fid_vals,
                    prop_coeffs=prop_coeffs,
                    flat_priors=self.priors,
                    Gauss_priors=Gauss_priors,
                )

    def set_fid_igm(self, zs):
        self.fid_igm = {}
        self.fid_igm["z"] = zs
        for key in self.models:
            for key2 in self.models[key].list_coeffs:
                if key2 == "tau_eff":
                    self.fid_igm[key] = self.models[key].get_tau_eff(zs)
                elif key2 == "gamma":
                    self.fid_igm[key] = self.models[key].get_gamma(zs)
                elif key2 == "sigT_kms":
                    self.fid_igm[key] = self.models[key].get_sigT_kms(zs)
                elif key2 == "kF_kms":
                    self.fid_igm[key] = self.models[key].get_kF_kms(zs)

    def get_igm(self, sim_igm_mF=None, sim_igm_T=None, sim_igm_kF=None):
        """Load IGM history"""

        repo = os.path.dirname(lace.__path__[0])
        fname = os.path.join(
            repo, "data", "sim_suites", "Australia20", "IGM_histories.npy"
        )
        try:
            self.igm_hist_mpg = np.load(fname, allow_pickle=True).item()
        except:
            raise ValueError(
                fname
                + " not found. You can produce it using LaCE"
                + r" script save_mpg_IGM.py"
            )

        # fname = os.path.join(os.environ["NYX_PATH"], "IGM_histories.npy")
        # try:
        #     self.igm_hist_nyx = np.load(fname, allow_pickle=True).item()
        # except:
        #     raise ValueError(
        #         fname
        #         + " not found. You can produce it using LaCE"
        #         + r" script save_nyx_IGM.py"
        #     )

        sim_igms = [sim_igm_mF, sim_igm_T, sim_igm_kF]

        igms_return = {}
        for ii, sim_igm in enumerate(sim_igms): # only keep MPG histories for now.
            if sim_igm[:3] == "mpg":
                igm_hist = self.igm_hist_mpg
            else:
                ValueError("sim_igm must be 'mpg'")

            if sim_igm not in igm_hist:
                igm_return = igm_hist[sim_igm + "_0"]
            else:
                igm_return = igm_hist[sim_igm]

            if ii == 0:
                igms_return["tau_eff_z"] = igm_return["z"]
                igms_return["tau_eff"] = igm_return["tau_eff"]
                igms_return["mF"] = igm_return["mF"]
                igms_return["F_suite"] = sim_igm
            elif ii == 1:
                igms_return["sigT_kms_z"] = igm_return["z"]
                igms_return["sigT_kms"] = igm_return["sigT_kms"]
                igms_return["sigT_Mpc"] = igm_return["sigT_Mpc"]
                igms_return["gamma_z"] = igm_return["z"]
                igms_return["gamma"] = igm_return["gamma"]
                igms_return["T_suite"] = sim_igm
            elif ii == 2:
                igms_return["kF_kms_z"] = igm_return["z"]
                igms_return["kF_kms"] = igm_return["kF_kms"]
                igms_return["kF_Mpc"] = igm_return["kF_Mpc"]
                igms_return["P_suite"] = sim_igm

            # important for nyx simulations, not all have kF
            # if so, we assign the values for nyx_central
            if np.sum(igm_return["kF_kms"] != 0) == 0:
                igms_return["kF_kms_z"] = igm_hist["nyx_central"]["z"]
                igms_return["kF_Mpc"] = igm_hist["nyx_central"]["kF_Mpc"]
                igms_return["kF_kms"] = igm_hist["nyx_central"]["kF_kms"]
                igms_return["P_suite"] = "nyx_central"

        return igms_return

    def set_priors(self, fid_igm, prop_coeffs, type_priors="hc", z_pivot=3):
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

        self.priors = {}
        for par in fid_igm:
            if (par == "val_scaling") | (
                par.endswith("_z") | par.endswith("_suite")
            ):
                continue

            if (par == "mF") | (par == "tau_eff"):
                z = fid_igm["tau_eff_z"]
                otype = prop_coeffs["tau_eff_otype"]
                emu_suite = fid_igm["F_suite"]
            elif (par == "kF_Mpc") | (par == "kF_kms"):
                z = fid_igm["kF_kms_z"]
                otype = prop_coeffs["kF_kms_otype"]
                emu_suite = fid_igm["P_suite"]
            elif par == "gamma":
                z = fid_igm["sigT_kms_z"]
            elif (par == "sigT_Mpc") | (par == "sigT_kms"):
                z = fid_igm["sigT_kms_z"]
                otype = prop_coeffs["sigT_kms_otype"]
                emu_suite = fid_igm["T_suite"]

            if emu_suite.startswith("mpg"):
                all_igm = self.igm_hist_mpg
            elif emu_suite.startswith("nyx"):
                all_igm = self.igm_hist_nyx
            else:
                ValueError("sim_igm must be 'mpg' or 'nyx'")

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

                res_div[ii, 0] = np.max(all_igm[sim][par][_] / fid_igm[par][_])
                res_div[ii, 1] = np.min(all_igm[sim][par][_] / fid_igm[par][_])

            _ = np.argwhere(
                np.isfinite(res_div[:, 0])
                & (res_div[:, 0] != 0)
                & (np.abs(res_div[:, 0]) != 1)
            )[:, 0]
            if len(_) == 0:
                print("no good points for ", par)
                self.priors[par] = [[-1, 1], [-1, 1]]
                continue

            if otype == "exp":
                y0_max = np.abs(
                    np.log(np.percentile(np.abs(res_div[_, 0]), percent))
                )
            elif otype == "const":
                y0_max = np.percentile(res_div[_, 0], percent)
            else:
                raise ValueError("otype must be 'exp' or 'const'", par)

            _ = np.argwhere(
                np.isfinite(res_div[:, 1])
                & (res_div[:, 1] != 0)
                & (np.abs(res_div[:, 1]) != 1)
            )[:, 0]
            if len(_) == 0:
                print("no good points for ", par)
                self.priors[par] = [[-1, 1], [-1, 1]]
                continue

            if otype == "exp":
                y0_min = np.abs(
                    np.log(np.percentile(1 / np.abs(res_div[_, 1]), percent))
                )
            elif otype == "const":
                y0_min = np.percentile(res_div[_, 1], 100 - percent)

            y0_cen = 0.5 * (y0_max + y0_min)
            if otype == "exp":
                y1 = y0_cen / np.log((1 + z.max()) / (1 + z_pivot))
            elif otype == "const":
                y1 = y0_cen / ((1 + z.max()) / (1 + z_pivot))

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
