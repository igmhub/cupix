import numpy as np
from cupix.nuisance.base_contaminants import Contaminant


class MetalModel(Contaminant):
    """Model the contamination from Silicon Lya cross-correlations"""

    def __init__(
        self,
        metal_label,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_vals=None,
        flat_priors=None,
        Gauss_priors=None,
    ):
        """Model the evolution of a metal contamination (SiII or SiIII).
        We use a power law around z_0=3."""

        # label identifying the metal line
        self.metal_label = metal_label
        if metal_label == "Lya_SiIII":
            self.lambda_rest = [1206.51, 1215.67]
        elif metal_label == "Lya_SiIIa":
            self.lambda_rest = [1190.42, 1215.67]
        elif metal_label == "Lya_SiIIb":
            self.lambda_rest = [1193.28, 1215.67]
        elif metal_label == "Lya_SiIIc":
            self.lambda_rest = [1215.67, 1260.42]
        elif metal_label == "SiIIa_SiIII":
            # we should model this somewhere else
            self.lambda_rest = [1190.42, 1206.51]
        elif metal_label == "SiIIb_SiIII":
            # we should model this somewhere else
            self.lambda_rest = [1193.28, 1206.51]
        elif metal_label == "SiII_SiIII":
            # we should model this somewhere else
            self.lambda_rest = [1190.42, 1206.51]
            self.lambda_rest2 = [1193.28, 1206.51]
        elif metal_label == "SiIIc_SiIII":
            # we should model this somewhere else
            self.lambda_rest = [1206.51, 1260.42]
        else:
            raise ValueError("metal_label not supported", metal_label)

        c_kms = 299792.458
        self.dv = np.log(self.lambda_rest[1] / self.lambda_rest[0]) * c_kms
        if metal_label == "SiII_SiIII":
            self.dv2 = (
                np.log(self.lambda_rest2[1] / self.lambda_rest2[0]) * c_kms
            )

        list_coeffs = [
            "f_" + metal_label,
            "s_" + metal_label,
            "p_" + metal_label,
        ]

        if flat_priors is None:
            flat_priors = {
                "f_" + metal_label: [[-3, 3], [-11, -1]],
                "s_" + metal_label: [[-1, 1], [-10, 10]],
                "p_" + metal_label: [[-1, 1], [0.95, 1.05]],
            }

        if prop_coeffs is None:
            prop_coeffs = {
                "f_" + metal_label + "_ztype": "pivot",
                "s_" + metal_label + "_ztype": "pivot",
                "p_" + metal_label + "_ztype": "pivot",
                "f_" + metal_label + "_otype": "exp",
                "s_" + metal_label + "_otype": "exp",
                "p_" + metal_label + "_otype": "const",
            }

        if fid_vals is None:
            fid_vals = {
                "f_" + metal_label: [0, -10.5],
                "s_" + metal_label: [0, 0],
                "p_" + metal_label: [0, 1],
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

    def get_contamination(self, z, k_kms, mF, like_params=[]):
        """Multiplicative contamination at a given z and k (in s/km).
        The mean flux (mF) is used scale it (see McDonald et al. 2006)"""

        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z, like_params=like_params)
            )

        a = vals["f_" + self.metal_label] / (1 - mF)

        metal_corr = []
        for iz in range(len(z)):
            damping = 1 - 1 / (
                1 + np.exp(-vals["s_" + self.metal_label][iz] * k_kms[iz])
            )
            osc = self.dv * k_kms[iz] * vals["p_" + self.metal_label][iz]
            _metal = 1 + a[iz] ** 2 + 4 * a[iz] * np.cos(osc) * damping
            if self.metal_label == "SiII_SiIII":
                osc2 = self.dv2 * k_kms[iz] * vals["p_" + self.metal_label][iz]
                _metal = _metal * (
                    1 + a[iz] ** 2 + 4 * a[iz] * np.cos(osc2) * damping
                )
            metal_corr.append(_metal)

        if len(z) == 1:
            metal_corr = metal_corr[0]

        return metal_corr
