import numpy as np
from cupix.nuisance.base_contaminants import Contaminant


def vel_diff(lambda1, lambda2):
    c_kms = 299792.458
    return np.abs(np.log(lambda2 / lambda1)) * c_kms


def rstrength(lambda1, lambda2, f1, f2):
    return (lambda1 * f1) / (lambda2 * f2)


class SiAdd(Contaminant):
    """Model the contamination from Silicon Lya cross-correlations"""

    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_vals=None,
        null_vals=None,
        flat_priors=None,
        z_max=None,
        Gauss_priors=None,
    ):
        """Model the evolution of a metal contamination (SiII or SiIII).
        We use a power law around z_0=3."""

        self.wav = {
            # "SiIII": 1206.50,
            "SiIIc": 1260.42,
            "SiIIb": 1193.28,
            "SiIIa": 1190.42,
        }
        self.osc_strength = {
            "SiIIc": 1.22,
            "SiIIb": 0.575,
            "SiIIa": 0.277,
        }

        self.dv = {
            # "SiIII_SiIIc": vel_diff(self.wav["SiIII"], self.wav["SiIIc"]),
            # "SiIII_SiIIb": vel_diff(self.wav["SiIII"], self.wav["SiIIb"]),
            # "SiIII_SiIIa": vel_diff(self.wav["SiIII"], self.wav["SiIIa"]),
            "SiIIc_SiIIb": vel_diff(self.wav["SiIIc"], self.wav["SiIIb"]),
            "SiIIc_SiIIa": vel_diff(self.wav["SiIIc"], self.wav["SiIIa"]),
            "SiIIb_SiIIa": vel_diff(self.wav["SiIIb"], self.wav["SiIIa"]),
        }

        self.rat = {
            "SiIIa_SiIIc": rstrength(
                self.wav["SiIIa"],
                self.wav["SiIIc"],
                self.osc_strength["SiIIa"],
                self.osc_strength["SiIIc"],
            ),
            "SiIIb_SiIIc": rstrength(
                self.wav["SiIIb"],
                self.wav["SiIIc"],
                self.osc_strength["SiIIb"],
                self.osc_strength["SiIIc"],
            ),
            "SiIIa_SiIIb": rstrength(
                self.wav["SiIIa"],
                self.wav["SiIIb"],
                self.osc_strength["SiIIa"],
                self.osc_strength["SiIIb"],
            ),
        }

        self.off = {
            "SiIIc_SiIIb": 0,
            "SiIIc_SiIIa": 0,
            "SiIIb_SiIIa": 1,
            "SiIIacbc": 0,
            "SiIIacab": 0,
            "SiIIbcab": 0,
        }

        list_coeffs = [
            "f_SiIIa_SiIIb",
            "s_SiIIa_SiIIb",
        ]

        if flat_priors is None:
            flat_priors = {}
            for coeff in list_coeffs:
                if coeff.startswith("f"):
                    flat_priors[coeff] = [[-3, 3], [-11, 2]]
                else:
                    flat_priors[coeff] = [[-1, 1], [-10, 7]]

        if prop_coeffs is None:
            prop_coeffs = {}
            for coeff in list_coeffs:
                prop_coeffs[coeff + "_ztype"] = "pivot"
                prop_coeffs[coeff + "_otype"] = "exp"

        if fid_vals is None:
            fid_vals = {}
            for coeff in list_coeffs:
                fid_vals[coeff] = [0, -11.5]

        if null_vals is None:
            null_vals = {}
            for coeff in list_coeffs:
                null_vals[coeff] = -11.5

        if z_max is None:
            z_max = {}
            for coeff in list_coeffs:
                z_max[coeff] = 3.3

        super().__init__(
            coeffs=coeffs,
            list_coeffs=list_coeffs,
            prop_coeffs=prop_coeffs,
            free_param_names=free_param_names,
            z_0=z_0,
            fid_vals=fid_vals,
            null_vals=null_vals,
            z_max=z_max,
            flat_priors=flat_priors,
            Gauss_priors=Gauss_priors,
        )

    def get_contamination(self, z, k_kms, mF, like_params=[], remove=None):
        """Multiplicative contamination at a given z and k (in s/km).
        The mean flux (mF) is used scale it (see McDonald et al. 2006)"""

        # z = np.atleast_1d(z)
        # k_kms = np.atleast_2d(k_kms)
        # mF = np.atleast_1d(mF)

        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z, like_params=like_params)
            )
            if key in self.null_vals:
                if self.prop_coeffs[key + "_otype"] == "const":
                    null = self.null_vals[key]
                else:
                    null = np.exp(self.null_vals[key])
                _ = vals[key] <= null
                vals[key][_] = 0

        rac = self.rat["SiIIa_SiIIc"]
        rbc = self.rat["SiIIb_SiIIc"]
        rab = self.rat["SiIIa_SiIIb"]

        if remove is not None:
            for key in remove:
                if key in self.off:
                    self.off[key] = remove[key]
        # print(self.off)

        metal_corr = []

        for iz in range(len(z)):
            aSiII = vals["f_SiIIa_SiIIb"][iz].copy()

            # G_SiII_SiII = 2 - 2 / (
            #     1 + np.exp(-vals["s_SiIIa_SiIIb"][iz] * k_kms[iz])
            # )
            G_SiII_SiII = np.exp(
                -1 * vals["s_SiIIa_SiIIb"][iz] ** 2 * k_kms[iz] ** 2
            )

            Cac = (
                1
                + rac**2
                + 2 * rac * np.cos(self.dv["SiIIc_SiIIa"] * k_kms[iz])
            )

            Cbc = (
                1
                + rbc**2
                + 2 * rbc * np.cos(self.dv["SiIIc_SiIIb"] * k_kms[iz])
            )

            Cba = rbc**2 * (
                1
                + rab**2
                + 2 * rab * np.cos(self.dv["SiIIb_SiIIa"] * k_kms[iz])
            )

            dv_d = 0.5 * (self.dv["SiIIc_SiIIa"] - self.dv["SiIIc_SiIIb"])
            dv_s = 0.5 * (self.dv["SiIIc_SiIIa"] + self.dv["SiIIc_SiIIb"])
            Cacbc1 = 2 * (1 + rac * rbc) * np.cos(dv_d * k_kms[iz])
            Cacbc2 = 2 * (rac + rbc) * np.cos(dv_s * k_kms[iz])

            dv_d = 0.5 * (self.dv["SiIIc_SiIIa"] - self.dv["SiIIb_SiIIa"])
            dv_s = 0.5 * (self.dv["SiIIc_SiIIa"] + self.dv["SiIIb_SiIIa"])
            Cacba1 = 2 * rbc * (1 + rac * rab) * np.cos(dv_d * k_kms[iz])
            Cacba2 = 2 * rbc * (rac + rab) * np.cos(dv_s * k_kms[iz])

            dv_d = 0.5 * (self.dv["SiIIc_SiIIb"] - self.dv["SiIIb_SiIIa"])
            dv_s = 0.5 * (self.dv["SiIIc_SiIIb"] + self.dv["SiIIb_SiIIa"])
            Cbcba1 = 2 * rbc * (1 + rbc * rab) * np.cos(dv_d * k_kms[iz])
            Cbcba2 = 2 * rbc * (rbc + rab) * np.cos(dv_s * k_kms[iz])

            ktot = (
                self.off["SiIIc_SiIIa"] * Cac
                + self.off["SiIIc_SiIIb"] * Cbc
                + self.off["SiIIb_SiIIa"] * Cba
                + self.off["SiIIacbc"] * Cacbc1
                + self.off["SiIIacbc"] * Cacbc2
                + self.off["SiIIacab"] * Cacba1
                + self.off["SiIIacab"] * Cacba2
                + self.off["SiIIbcab"] * Cbcba1
                + self.off["SiIIbcab"] * Cbcba2
            )

            metal_corr.append(aSiII**2 * ktot * G_SiII_SiII)

        return metal_corr
