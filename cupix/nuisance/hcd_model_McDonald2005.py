import numpy as np
import copy
import os
from cupix.likelihood import likelihood_parameter
from cupix.utils.utils import get_discrete_cmap
from matplotlib import pyplot as plt


class HCD_Model_McDonald2005(object):
    """Model HCD contamination following McDonald et al. (2005)."""

    def __init__(
        self,
        z_0=3.0,
        fid_A_damp=[0, -6],
        null_value=-6,
        ln_A_damp_coeff=None,
        free_param_names=None,
    ):
        self.z_0 = z_0
        self.null_value = null_value
        if fid_A_damp is None:
            fid_A_damp = [0, -6]

        if ln_A_damp_coeff:
            if free_param_names is not None:
                raise ValueError("can not specify coeff and free_param_names")
            self.ln_A_damp_coeff = ln_A_damp_coeff
        else:
            if free_param_names:
                # figure out number of HCD free params
                n_hcd = len([p for p in free_param_names if "ln_A_damp_" in p])
                if n_hcd == 0:
                    n_hcd = 1
            else:
                n_hcd = 1

            self.ln_A_damp_coeff = [0.0] * n_hcd
            self.ln_A_damp_coeff[-1] = fid_A_damp[-1]
            if n_hcd == 2:
                self.ln_A_damp_coeff[-2] = fid_A_damp[-2]

        self.set_parameters()

    def set_parameters(self):
        """Setup likelihood parameters in the HCD model"""

        self.params = []
        Npar = len(self.ln_A_damp_coeff)
        for i in range(Npar):
            name = "ln_A_damp_" + str(i)
            if i == 0:
                # no contamination
                xmin = -7
                # 0 gives 30% contamination low k
                xmax = 2.5
            else:
                # not optimized
                xmin = -10
                xmax = 10
            # note non-trivial order in coefficients
            value = self.ln_A_damp_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.params.append(par)

        return

    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.ln_A_damp_coeff) == len(self.params), "size mismatch"
        return len(self.ln_A_damp_coeff)

    def get_A_damp(self, z, like_params=[]):
        """Amplitude of HCD contamination around z_0"""

        ln_A_damp_coeff = self.get_A_damp_coeffs(like_params=like_params)
        if ln_A_damp_coeff[-1] <= self.null_value:
            return 0

        xz = np.log((1 + z) / (1 + self.z_0))
        ln_poly = np.poly1d(ln_A_damp_coeff)
        ln_out = ln_poly(xz)
        return np.exp(ln_out)

    def get_contamination(self, z, k_kms, like_params=[]):
        """Multiplicative contamination caused by HCDs"""
        A_damp = self.get_A_damp(z, like_params=like_params)
        if A_damp == 0:
            return 1

        # fitting function from Palanque-Delabrouille et al. (2015)
        # that qualitatively describes Fig 2 of McDonald et al. (2005)
        f_HCD = 0.018 + 1 / (15000 * k_kms - 8.9)
        return 1 + A_damp * f_HCD

    def get_parameters(self):
        """Return likelihood parameters for the HCD model"""
        return self.params

    def get_A_damp_coeffs(self, like_params=[]):
        """Return list of mean flux coefficients"""

        if like_params:
            ln_A_damp_coeff = self.ln_A_damp_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_A_damp" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.ln_A_damp_coeff
            elif Npar != len(self.params):
                print(Npar, len(self.params))
                raise ValueError(
                    "number of params mismatch in get_A_damp_coeffs"
                )

            for ip in range(Npar):
                _ = np.argwhere(self.params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.params[ip].name
                    )
                else:
                    ln_A_damp_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            ln_A_damp_coeff = self.ln_A_damp_coeff

        return ln_A_damp_coeff

    def plot_contamination(
        self,
        z,
        k_kms,
        ln_A_damp_coeff=None,
        plot_every_iz=1,
        cmap=None,
        smooth_k=False,
    ):
        """Plot the contamination model"""

        # plot for fiducial value
        if ln_A_damp_coeff is None:
            ln_A_damp_coeff = self.ln_A_damp_coeff

        hcd_model = HCD_Model_McDonald2005(ln_A_damp_coeff=ln_A_damp_coeff)

        for ii in range(0, len(z), plot_every_iz):
            if smooth_k:
                k_use = np.logspace(
                    np.log10(k_kms[ii][0]), np.log10(k_kms[ii][-1]), 200
                )
            else:
                k_use = k_kms[ii]

            cont = hcd_model.get_contamination(z[ii], k_use)
            if isinstance(cont, int):
                cont = np.ones_like(k_use)
            if cmap is None:
                plt.plot(k_use, cont, label="z=" + str(z[ii]))
            else:
                plt.plot(k_use, cont, color=cmap(ii), label="z=" + str(z[ii]))

        plt.axhline(1, color="k", linestyle=":")

        plt.legend()
        plt.xscale("log")
        plt.xlabel(r"$k$ [1/Mpc]")
        plt.ylabel("HCD contamination")
        plt.tight_layout()

        return
