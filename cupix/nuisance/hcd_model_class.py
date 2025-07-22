import numpy as np
from cupix.nuisance.base_contaminants import Contaminant


class HCD_Model(Contaminant):
    """New model for HCD contamination"""

    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_vals=None,
        flat_priors=None,
        Gauss_priors=None,
    ):
        # list of all coefficients
        list_coeffs = [
            "HCD_damp1",
            "HCD_damp2",
            "HCD_damp3",
            "HCD_scale1",
            "HCD_scale2",
            "HCD_scale3",
            "HCD_const",
        ]

        # priors for all coefficients
        if flat_priors is None:
            flat_priors = {
                "HCD_damp": [[-0.5, 0.5], [-10, 5]],
                "HCD_scale": [[-1, 1], [1, 10]],
                "HCD_const": [[-1, 1], [-0.2, 1e-6]],
            }

        # z dependence and output type of coefficients
        if prop_coeffs is None:
            prop_coeffs = {
                "HCD_damp1_ztype": "pivot",
                "HCD_damp2_ztype": "pivot",
                "HCD_damp3_ztype": "pivot",
                "HCD_scale1_ztype": "pivot",
                "HCD_scale2_ztype": "pivot",
                "HCD_scale3_ztype": "pivot",
                "HCD_const_ztype": "pivot",
                "HCD_damp1_otype": "exp",
                "HCD_damp2_otype": "exp",
                "HCD_damp3_otype": "exp",
                "HCD_scale1_otype": "exp",
                "HCD_scale2_otype": "exp",
                "HCD_scale3_otype": "exp",
                "HCD_const_otype": "const",
            }

        # fiducial values
        if fid_vals is None:
            fid_vals = {
                "HCD_damp1": [0, -9.5],
                "HCD_scale1": [0, 5],
                "HCD_damp2": [0, -9.5],
                "HCD_scale2": [0, 5],
                "HCD_damp3": [0, -9.5],
                "HCD_scale3": [0, 5],
                "HCD_const": [0, 0],
            }

        super().__init__(
            coeffs=coeffs,
            list_coeffs=list_coeffs,
            prop_coeffs=prop_coeffs,
            free_param_names=free_param_names,
            z_0=z_0,
            fid_vals=fid_vals,
            flat_priors=flat_priors,
            Gauss_priors=Gauss_priors,
        )

    def get_contamination(self, z, k_kms, like_params=[]):
        """Multiplicative contamination caused by HCDs"""
        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z, like_params=like_params)
            )

        dla_corr = []
        for iz in range(len(z)):
            const = vals["HCD_const"][iz]
            term1 = vals["HCD_damp1"][iz] * np.exp(
                (k_kms[iz][0] - k_kms[iz]) * vals["HCD_scale1"][iz]
            )
            term2 = vals["HCD_damp2"][iz] * np.exp(
                (k_kms[iz][0] - k_kms[iz]) * vals["HCD_scale2"][iz]
            )
            term3 = vals["HCD_damp3"][iz] * np.exp(
                (k_kms[iz][0] - k_kms[iz]) * vals["HCD_scale3"][iz]
            )
            dla_corr.append(1 + const + term1 + term2 + term3)

        if len(z) == 1:
            dla_corr = dla_corr[0]

        return dla_corr
