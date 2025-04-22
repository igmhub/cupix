import numpy as np
import copy
import os
import lace
from scipy.interpolate import interp1d
from lace.cosmo import thermal_broadening
from cup1d.likelihood import likelihood_parameter


class ThermalModel(object):
    """Model the redshift evolution of the gas temperature parameters gamma
    and sigT_kms.
    We use a power law rescaling around a fiducial simulation at the centre
    of the initial Latin hypercube in simulation space."""

    def __init__(
        self,
        z_T=3.0,
        ln_sigT_kms_coeff=None,
        ln_gamma_coeff=None,
        free_param_names=None,
        fid_igm=None,
        order_extra=2,
        smoothing=False,
        priors=None,
        emu_suite="mpg",
        back_igm=None,
    ):
        """Model the redshift evolution of the thermal broadening scale and gamma.
        We use a power law rescaling around a fiducial simulation at the centre
        of the initial Latin hypercube in simulation space."""

        if fid_igm is None:
            repo = os.path.dirname(lace.__path__[0]) + "/"
            fname = repo + "data/sim_suites/Australia20/IGM_histories.npy"
            try:
                igm_hist = np.load(fname, allow_pickle=True).item()
            except:
                raise ValueError(
                    fname
                    + " not found. You can produce it using the LaCE"
                    + r" script save_mpg_IGM.py"
                )
            else:
                fid_igm = igm_hist["mpg_central"]
        self.priors = priors
        self.fid_igm = fid_igm

        mask = (fid_igm["gamma"] != 0) & np.isfinite(fid_igm["gamma"])
        if np.sum(mask) != fid_igm["gamma"].shape[0]:
            print(
                "The fiducial value of gamma is zero for z: ",
                fid_igm["z_T"][mask == False],
            )
        mask = (fid_igm["sigT_kms"] != 0) & np.isfinite(fid_igm["sigT_kms"])
        if np.sum(mask) != fid_igm["sigT_kms"].shape[0]:
            print(
                "The fiducial value of sigT_kms is zero for z: ",
                fid_igm["z_T"][mask == False],
            )

        mask = (
            (fid_igm["gamma"] != 0)
            & (fid_igm["sigT_kms"] != 0)
            & (fid_igm["z_T"] != 0)
            & np.isfinite(fid_igm["z_T"])
            & np.isfinite(fid_igm["gamma"])
            & np.isfinite(fid_igm["sigT_kms"])
        )
        if np.sum(mask) == 0:
            print(
                "No non-zero gamma and sigT_kms in fiducial IGM, switching to back_igm"
            )
            fid_igm = back_igm
            mask = (
                (fid_igm["gamma"] != 0)
                & (fid_igm["sigT_kms"] != 0)
                & (fid_igm["z_T"] != 0)
            )

        # fit power law to fiducial data to reduce noise, and extrapolate when needed
        for ii in range(2):
            if ii == 0:
                label = "gamma"

            else:
                label = "sigT_kms"
            pfit = np.polyfit(
                fid_igm["z_T"][mask],
                fid_igm[label][mask],
                order_extra,
            )
            p = np.poly1d(pfit)
            ind = np.argsort(fid_igm["z_T"][mask])
            if ii == 0:
                self.fid_z = fid_igm["z_T"][mask][ind]
                self.fid_gamma = fid_igm[label][mask][ind]
                fid_prop = self.fid_gamma
            else:
                self.fid_sigT_kms = fid_igm[label][mask][ind]
                fid_prop = self.fid_sigT_kms

            # use poly fit to interpolate when data is missing (needed for Nyx)
            mask2 = (fid_prop == 0) | np.isnan(fid_prop)
            fid_prop[mask2] = p(self.fid_z[mask2])

            # extrapolate to z=1.9 and 5.5 (if needed)
            if np.min(self.fid_z) > 1.9:
                low = True
            else:
                low = False
            if np.max(self.fid_z) < 5.5:
                high = True
            else:
                high = False

            if low and high:
                z_to_inter = np.concatenate([[1.9], self.fid_z, [5.5]])
            elif low:
                z_to_inter = np.concatenate([[1.9], self.fid_z])
            elif high:
                z_to_inter = np.concatenate([self.fid_z, [5.5]])
            else:
                z_to_inter = self.fid_z
            igm_to_inter = np.zeros_like(z_to_inter)

            # extrapolate to low z
            if low:
                igm_to_inter[0] = p(z_to_inter[0])
            # extrapolate to high z
            if high:
                igm_to_inter[-1] = p(z_to_inter[-1])

            ind_all = np.arange(z_to_inter.shape[0])
            if low and high:
                ind_use = ind_all[1:-1]
            elif low:
                ind_use = ind_all[1:]
            elif high:
                ind_use = ind_all[:-1]
            else:
                ind_use = ind_all

            # apply smoothing to IGM history
            if smoothing:
                igm_to_inter[ind_use] = p(z_to_inter[ind_use])
            else:
                igm_to_inter[ind_use] = fid_prop

            if ii == 0:
                self.fid_gamma_interp = interp1d(
                    z_to_inter, igm_to_inter, kind="cubic"
                )
            else:
                self.fid_sigT_kms_interp = interp1d(
                    z_to_inter, igm_to_inter, kind="cubic"
                )

        self.z_T = z_T

        # figure out parameters for sigT_kms (T0)
        if ln_sigT_kms_coeff:
            assert free_param_names is None
            self.ln_sigT_kms_coeff = ln_sigT_kms_coeff
        else:
            if free_param_names:
                # figure out number of sigT free params
                n_sigT = len(
                    [p for p in free_param_names if "ln_sigT_kms_" in p]
                )
            else:
                n_sigT = 2
            self.ln_sigT_kms_coeff = [0.0] * n_sigT
        # store list of likelihood parameters (might be fixed or free)
        self.set_sigT_kms_parameters()

        # figure out parameters for gamma (T0)
        if ln_gamma_coeff:
            assert free_param_names is None
            self.ln_gamma_coeff = ln_gamma_coeff
        else:
            if free_param_names:
                # figure out number of gamma free params
                n_gamma = len([p for p in free_param_names if "ln_gamma_" in p])
            else:
                n_gamma = 2
            self.ln_gamma_coeff = [0.0] * n_gamma
        # store list of likelihood parameters (might be fixed or free)
        self.set_gamma_parameters()

    def get_sigT_kms_Nparam(self):
        """Number of parameters in the model of T_0"""
        assert len(self.ln_sigT_kms_coeff) == len(
            self.sigT_kms_params
        ), "size mismatch"
        return len(self.ln_sigT_kms_coeff)

    def get_gamma_Nparam(self):
        """Number of parameters in the model of gamma"""
        assert len(self.ln_gamma_coeff) == len(
            self.gamma_params
        ), "size mismatch"
        return len(self.ln_gamma_coeff)

    def power_law_scaling_gamma(self, z, like_params=[]):
        """Power law rescaling around z_T"""

        ln_gamma_coeff = self.get_gamma_coeffs(like_params=like_params)

        xz = np.log((1 + z) / (1 + self.z_T))
        ln_poly = np.poly1d(ln_gamma_coeff)
        ln_out = ln_poly(xz)
        return np.exp(ln_out)

    def power_law_scaling_sigT_kms(self, z, like_params=[]):
        """Power law rescaling around z_T"""

        ln_sigT_kms_coeff = self.get_sigT_coeffs(like_params=like_params)

        xz = np.log((1 + z) / (1 + self.z_T))
        ln_poly = np.poly1d(ln_sigT_kms_coeff)
        ln_out = ln_poly(xz)
        return np.exp(ln_out)

    def get_sigT_kms(self, z, like_params=[]):
        """sigT_kms at the input redshift"""
        sigT_kms = self.power_law_scaling_sigT_kms(
            z, like_params=like_params
        ) * self.fid_sigT_kms_interp(z)
        return sigT_kms

    def get_T0(self, z, like_params=[]):
        """T_0 at the input redshift"""
        sigT_kms = self.power_law_scaling_sigT_kms(
            z, like_params=like_params
        ) * self.fid_sigT_kms_interp(z)
        T0 = thermal_broadening.T0_from_broadening_kms(sigT_kms)
        return T0

    def get_gamma(self, z, like_params=[]):
        """gamma at the input redshift"""
        gamma = self.power_law_scaling_gamma(
            z, like_params=like_params
        ) * self.fid_gamma_interp(z)
        return gamma

    def set_sigT_kms_parameters(self):
        """Setup sigT_kms likelihood parameters for thermal model"""

        self.sigT_kms_params = []
        Npar = len(self.ln_sigT_kms_coeff)
        for i in range(Npar):
            name = "ln_sigT_kms_" + str(i)
            pname = "sigT_kms"
            if i == 0:
                if self.priors is not None:
                    xmin = self.priors[pname][-1][0]
                    xmax = self.priors[pname][-1][1]
                else:
                    xmin = -0.25
                    xmax = 0.25
            else:
                if self.priors is not None:
                    xmin = self.priors[pname][-2][0]
                    xmax = self.priors[pname][-2][1]
                else:
                    xmin = -1.0
                    xmax = 1.0
            # note non-trivial order in coefficients
            value = self.ln_sigT_kms_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.sigT_kms_params.append(par)
        return

    def set_gamma_parameters(self):
        """Setup gamma likelihood parameters for thermal model"""

        self.gamma_params = []
        Npar = len(self.ln_gamma_coeff)
        for i in range(Npar):
            name = "ln_gamma_" + str(i)
            pname = "gamma"
            if i == 0:
                if self.priors is not None:
                    xmin = self.priors[pname][-1][0]
                    xmax = self.priors[pname][-1][1]
                else:
                    xmin = -0.25
                    xmax = 0.25
            else:
                if self.priors is not None:
                    xmin = self.priors[pname][-2][0]
                    xmax = self.priors[pname][-2][1]
                else:
                    xmin = -1.0
                    xmax = 1.0
            # note non-trivial order in coefficients
            value = self.ln_gamma_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.gamma_params.append(par)

        return

    def get_sigT_kms_parameters(self):
        """Return sigT_kms likelihood parameters from the thermal model"""
        return self.sigT_kms_params

    def get_gamma_parameters(self):
        """Return gamma likelihood parameters from the thermal model"""
        return self.gamma_params

    # def update_parameters(self, like_params):
    #     """Look for thermal parameters in list of parameters"""

    #     Npar_sigT_kms = self.get_sigT_kms_Nparam()
    #     Npar_gamma = self.get_gamma_Nparam()

    #     # loop over likelihood parameters
    #     for like_par in like_params:
    #         if "ln_sigT_kms" in like_par.name:
    #             # make sure you find the parameter
    #             found = False
    #             # loop over T0 parameters in thermal model
    #             for ip in range(len(self.sigT_kms_params)):
    #                 if self.sigT_kms_params[ip].name == like_par.name:
    #                     assert found == False, "can not update parameter twice"
    #                     self.ln_sigT_kms_coeff[
    #                         Npar_sigT_kms - ip - 1
    #                     ] = like_par.value
    #                     # self.ln_sigT_kms_coeff[ip]=like_par.value
    #                     found = True
    #             assert found == True, (
    #                 "could not update parameter " + like_par.name
    #             )
    #         elif "ln_gamma" in like_par.name:
    #             # make sure you find the parameter
    #             found = False
    #             # loop over gamma parameters in thermal model
    #             for ip in range(len(self.gamma_params)):
    #                 if self.gamma_params[ip].name == like_par.name:
    #                     assert found == False, "can not update parameter twice"
    #                     self.ln_gamma_coeff[
    #                         Npar_gamma - ip - 1
    #                     ] = like_par.value
    #                     found = True
    #             assert found == True, (
    #                 "could not update parameter " + like_par.name
    #             )

    #     return

    # def get_new_model(self, like_params=[]):
    #     """Return copy of model, updating values from list of parameters"""

    #     T = ThermalModel(
    #         fid_igm=self.fid_igm,
    #         z_T=self.z_T,
    #         ln_sigT_kms_coeff=copy.deepcopy(self.ln_sigT_kms_coeff),
    #         ln_gamma_coeff=copy.deepcopy(self.ln_gamma_coeff),
    #     )
    #     T.update_parameters(like_params)
    #     return T

    def get_sigT_coeffs(self, like_params=[]):
        """Return list of sigT coefficients"""
        if like_params:
            ln_sigT_kms_coeff = self.ln_sigT_kms_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_sigT_kms" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            if Npar != len(self.sigT_kms_params):
                raise ValueError("number of params mismatch in get_sigT_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.sigT_kms_params[ip].name == array_names)[
                    :, 0
                ]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter"
                        + self.sigT_kms_params[ip].name
                    )
                else:
                    ln_sigT_kms_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            ln_sigT_kms_coeff = self.ln_sigT_kms_coeff

        return ln_sigT_kms_coeff

    def get_gamma_coeffs(self, like_params=[]):
        """Return list of gamma coefficients"""
        if like_params:
            ln_gamma_coeff = self.ln_gamma_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_gamma" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            if Npar != len(self.gamma_params):
                raise ValueError(
                    "number of params mismatch in get_gamma_coeffs"
                )

            for ip in range(Npar):
                _ = np.argwhere(self.gamma_params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter"
                        + self.gamma_params[ip].name
                    )
                else:
                    ln_gamma_coeff[Npar - ip - 1] = array_values[_[0]]

        else:
            ln_gamma_coeff = self.ln_gamma_coeff

        return ln_gamma_coeff
