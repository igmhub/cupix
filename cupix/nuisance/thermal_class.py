import numpy as np
import os
import lace
from cupix.nuisance.base_igm import IGM_model

from lace.cosmo import thermal_broadening


class Thermal(IGM_model):
    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_igm=None,
        fid_vals=None,
        flat_priors=None,
        Gauss_priors=None,
    ):
        list_coeffs = ["sigT_kms", "gamma"]

        if prop_coeffs is None:
            prop_coeffs = {}
            for coeff in list_coeffs:
                prop_coeffs[coeff + "_ztype"] = "interp_spl"
                prop_coeffs[coeff + "_otype"] = "const"

        if flat_priors is None:
            flat_priors = {}
            for coeff in list_coeffs:
                flat_priors[coeff] = [[-1, 1], [-1.25, 1.25]]

        for coeff in list_coeffs:
            if coeff not in fid_vals:
                if prop_coeffs[coeff + "_ztype"] == "pivot":
                    fid_vals[coeff] = [0, 1]
                else:
                    fid_vals[coeff] = np.ones(
                        len(prop_coeffs[coeff + "_znodes"])
                    )

        super().__init__(
            coeffs=coeffs,
            list_coeffs=list_coeffs,
            prop_coeffs=prop_coeffs,
            free_param_names=free_param_names,
            z_0=z_0,
            fid_vals=fid_vals,
            flat_priors=flat_priors,
            Gauss_priors=Gauss_priors,
            fid_igm=fid_igm,
        )

    def get_sigT_kms(self, z, like_params=[], name_par="sigT_kms"):
        """sigT_kms at the input redshift"""

        sigT_kms = self.get_value(name_par, z, like_params=like_params)
        sigT_kms *= self.fid_interp[name_par](z)
        return sigT_kms

    def get_T0(self, z, like_params=[], name_par="sigT_kms"):
        """T_0 at the input redshift"""

        sigT_kms = self.get_sigT_kms(
            z, like_params=like_params, name_par=name_par
        )
        T0 = thermal_broadening.T0_from_broadening_kms(sigT_kms)
        return T0

    def get_gamma(self, z, like_params=[], name_par="gamma"):
        """gamma at the input redshift"""

        gamma = self.get_value(name_par, z, like_params=like_params)
        gamma *= self.fid_interp[name_par](z)
        return gamma
