import numpy as np
import copy
import os
from cupix.likelihood import likelihood_parameter


class SN_Model(object):
    """Model SN contamination following Viel+13"""

    def __init__(
        self,
        z_0=3.0,
        fid_value=[0, -4],
        null_value=-4,
        ln_SN_coeff=None,
        free_param_names=None,
    ):
        self.z_0 = z_0
        if fid_value is None:
            fid_value = [0, -4]
        self.null_value = null_value

        if ln_SN_coeff:
            if free_param_names is not None:
                raise ValueError("can not specify coeff and free_param_names")
            self.ln_SN_coeff = ln_SN_coeff
        else:
            if free_param_names:
                # figure out number of SN free params
                n_SN = len([p for p in free_param_names if "ln_SN_" in p])
                if n_SN == 0:
                    n_SN = 1
            else:
                n_SN = 1

            self.ln_SN_coeff = [0.0] * n_SN
            self.ln_SN_coeff[-1] = fid_value[-1]
            if n_SN == 2:
                self.ln_SN_coeff[-2] = fid_value[-2]

        self.set_parameters()

    def set_parameters(self):
        """Setup likelihood parameters in the HCD model"""

        self.params = []
        Npar = len(self.ln_SN_coeff)
        for i in range(Npar):
            name = "ln_SN_" + str(i)
            if i == 0:
                xmin = -5
                xmax = 2
            else:
                # not optimized
                xmin = -1
                xmax = 1
            # note non-trivial order in coefficients
            value = self.ln_SN_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.params.append(par)

        return

    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.ln_SN_coeff) == len(self.params), "size mismatch"
        return len(self.ln_SN_coeff)

    def get_SN_damp(self, z, like_params=[]):
        """Amplitude of HCD contamination around z_0"""

        ln_SN_coeff = self.get_SN_coeffs(like_params=like_params)
        if ln_SN_coeff[-1] <= self.null_value:
            return 0
        else:
            xz = np.log((1 + z) / (1 + self.z_0))
            ln_poly = np.poly1d(ln_SN_coeff)
            ln_out = ln_poly(xz)
            return np.exp(ln_out)

    def get_contamination(self, z, k_Mpc, like_params=[]):
        """Multiplicative contamination caused by SNs"""
        SN_damp = self.get_SN_damp(z, like_params=like_params)
        if SN_damp == 0:
            return 1
        else:
            # Viel+13 Fig 8 panel b

            k0_Mpc = 0.07  # Mpc^-1
            k1_Mpc = 1.4  # Mpc^-1
            # Supernovae SN
            tmpLowk = [-0.06, -0.04, -0.02]
            tmpHighk = [-0.01, -0.01, -0.01]
            if z < 2.5:
                d0 = tmpLowk[0]
                d1 = tmpHighk[0]
            elif z < 3.5:
                d0 = tmpLowk[1]
                d1 = tmpHighk[1]
            else:
                d0 = tmpLowk[2]
                d1 = tmpHighk[2]
            delta = d0 + (d1 - d0) * (k_Mpc - k0_Mpc) / (k1_Mpc - k0_Mpc)
            corSN = 1 / (1.0 + delta * SN_damp)

            return corSN

    def get_parameters(self):
        """Return likelihood parameters for the HCD model"""
        return self.params

    def get_SN_coeffs(self, like_params=[]):
        """Return list of mean flux coefficients"""

        if like_params:
            ln_SN_coeff = self.ln_SN_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_SN" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.ln_SN_coeff
            elif Npar != len(self.params):
                print(Npar, len(self.params))
                raise ValueError("number of params mismatch in get_SN_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.params[ip].name
                    )
                else:
                    ln_SN_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            ln_SN_coeff = self.ln_SN_coeff

        return ln_SN_coeff
