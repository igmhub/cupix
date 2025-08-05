import numpy as np
import matplotlib.pyplot as plt
import copy
from cupix.utils.utils import get_discrete_cmap
from cupix.likelihood import likelihood_parameter


def signed_exp(x):
    return np.tanh(x) * np.exp(np.exp(np.abs(x)))


class MetalModel(object):
    """Model the contamination from Silicon Lya cross-correlations"""

    def __init__(
        self,
        metal_label,
        lambda_rest=None,
        z_X=3.0,
        ln_X_coeff=None,
        ln_D_coeff=None,
        A_coeff=None,
        X_fid_value=[0, -10],
        D_fid_value=[0, 5],
        A_fid_value=[0, 1.5],
        X_null_value=-10.5,
        A_null_value=0,
        free_param_names=None,
    ):
        """Model the evolution of a metal contamination (SiII or SiIII).
        We use a power law around z_X=3."""

        # label identifying the metal line
        self.metal_label = metal_label
        if metal_label == "SiIII":
            self.lambda_rest = 1206.50  # from McDonald et al. 2006
        elif metal_label == "SiII":
            self.lambda_rest = 1192.5  # like in Chabanier+19, Karacali+24
        else:
            if lambda_rest is None:
                raise ValueError("need to specify lambda_rest", metal_label)
            self.lambda_rest = lambda_rest

        # power law pivot point
        self.z_X = z_X
        # value below which no contamination (speed up model)
        self.X_null_value = X_null_value
        self.A_null_value = A_null_value
        self.dv = self.get_dv_kms()

        # figure out parameters
        if (
            (ln_X_coeff is not None)
            and (ln_D_coeff is not None)
            and (A_coeff is not None)
        ):
            if free_param_names is not None:
                raise ValueError("cannot specify coeff and free_param_names")
            self.ln_X_coeff = ln_X_coeff
            self.ln_D_coeff = ln_D_coeff
            self.A_coeff = A_coeff
        else:
            if free_param_names:
                # figure out number of free params for this metal line
                param_tag = "ln_x_" + metal_label + "_"
                n_X = len([p for p in free_param_names if param_tag in p])
                if n_X == 0:
                    n_X = 1

                param_tag = "ln_d_" + metal_label + "_"
                n_D = len([p for p in free_param_names if param_tag in p])
                if n_D == 0:
                    n_D = 1

                param_tag = "a_" + metal_label + "_"
                n_A = len([p for p in free_param_names if param_tag in p])
                if n_A == 0:
                    n_A = 1
            else:
                n_X = 1
                n_D = 1
                n_A = 1
            # start with value from McDonald et al. (2006), and no z evolution
            self.ln_X_coeff = np.zeros((n_X))
            if n_X == 1:
                self.ln_X_coeff[0] = X_fid_value[-1]
            else:
                for ii in range(n_X):
                    self.ln_X_coeff[ii] = X_fid_value[ii]

            self.ln_D_coeff = np.zeros((n_D))
            if n_D == 1:
                self.ln_D_coeff[0] = D_fid_value[-1]
            else:
                for ii in range(n_D):
                    self.ln_D_coeff[ii] = D_fid_value[ii]

            self.A_coeff = np.zeros((n_A))
            if n_A == 1:
                self.A_coeff[0] = A_fid_value[-1]
            else:
                for ii in range(n_A):
                    self.A_coeff[ii] = A_fid_value[ii]

        # store list of likelihood parameters (might be fixed or free)
        self.set_X_parameters()
        self.set_D_parameters()
        self.set_A_parameters()

    def get_Nparam(self):
        """Number of parameters in the model"""
        all_par = (
            len(self.ln_X_coeff) + len(self.ln_D_coeff) + len(self.A_coeff)
        )
        all_par2 = len(self.X_params) + len(self.D_params) + len(self.A_params)
        if all_par != all_par2:
            raise ValueError("parameter size mismatch")
        return all_par

    def set_X_parameters(self):
        """Setup likelihood parameters for metal model"""

        self.X_params = []
        Npar = len(self.ln_X_coeff)
        for i in range(Npar):
            name = "ln_x_" + self.metal_label + "_" + str(i)
            if i == 0:
                # log of overall amplitude at z_X
                # no contamination
                xmin = -11
                # max 10% contamination (oscillations) -4
                xmax = -2.5
            else:
                # not optimized
                xmin = -1
                xmax = 1
            # note non-trivial order in coefficients
            value = self.ln_X_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.X_params.append(par)
        return

    def set_D_parameters(self):
        """Setup likelihood parameters for metal model"""

        self.D_params = []
        Npar = len(self.ln_D_coeff)
        for i in range(Npar):
            name = "ln_d_" + self.metal_label + "_" + str(i)
            if i == 0:
                # log of overall amplitude at z_X
                # no contamination (0.3% at high k)
                xmin = 2
                # Almost no oscillations at high k
                xmax = 6
            else:
                # not optimized
                xmin = -1
                xmax = 1
            # note non-trivial order in coefficients
            value = self.ln_D_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.D_params.append(par)
        return

    def set_A_parameters(self):
        """Setup likelihood parameters for metal model"""

        self.A_params = []
        Npar = len(self.A_coeff)
        for i in range(Npar):
            name = "a_" + self.metal_label + "_" + str(i)
            if i == 0:
                # less than exponential damping
                xmin = 0.0
                # more damping than Gaussian
                xmax = 2.5
            else:
                xmin = -3
                xmax = 3
            # note non-trivial order in coefficients
            value = self.A_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.A_params.append(par)
        return

    def get_X_parameters(self):
        """Return likelihood parameters from the metal model"""
        return self.X_params

    def get_D_parameters(self):
        """Return likelihood parameters from the metal model"""
        return self.D_params

    def get_A_parameters(self):
        """Return likelihood parameters from the metal model"""
        return self.A_params

    def get_X_coeffs(self, like_params=[]):
        """Return list of coefficients for metal model"""

        if like_params:
            ln_X_coeff = self.ln_X_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_x_" + self.metal_label + "_" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.ln_X_coeff
            elif Npar != len(self.X_params):
                raise ValueError("number of params mismatch in get_X_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.X_params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter " + self.X_params[ip].name
                    )
                else:
                    ln_X_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            ln_X_coeff = self.ln_X_coeff

        return ln_X_coeff

    def get_D_coeffs(self, like_params=[]):
        """Return list of coefficients for metal model"""

        if like_params:
            ln_D_coeff = self.ln_D_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_d_" + self.metal_label + "_" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.ln_D_coeff
            elif Npar != len(self.D_params):
                raise ValueError("number of params mismatch in get_D_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.D_params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter " + self.D_params[ip].name
                    )
                else:
                    ln_D_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            ln_D_coeff = self.ln_D_coeff

        return ln_D_coeff

    def get_A_coeffs(self, like_params=[]):
        """Return list of coefficients for metal model"""

        if like_params:
            A_coeff = self.A_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "a_" + self.metal_label + "_" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.A_coeff
            elif Npar != len(self.A_params):
                raise ValueError("number of params mismatch in get_A_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.A_params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter " + self.A_params[ip].name
                    )
                else:
                    A_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            A_coeff = self.A_coeff

        return A_coeff

    def get_amplitude(self, z, like_params=[]):
        """Amplitude of contamination at a given z"""

        # Note that this represents "f" in McDonald et al. (2006)
        # It is later rescaled by <F> to compute "a" in eq. (15)

        _ln_X_coeff = self.get_X_coeffs(like_params).copy()

        if _ln_X_coeff[-1] <= self.X_null_value:
            return 0

        xz = np.log((1 + z) / (1 + self.z_X))

        # keep exponent for amplitude
        _ln_X_coeff[-1] = np.exp(_ln_X_coeff[-1])
        # for the redshift ev coeff, exp to reduce dynamic range
        if len(_ln_X_coeff) > 1:
            _ln_X_coeff[0] = signed_exp(_ln_X_coeff[0])

        ln_poly = np.poly1d(_ln_X_coeff)
        # redshift evolution linear
        return ln_poly(xz)

    def get_damping(self, z, like_params=[]):
        """Damping of contamination at a given z"""

        _ln_D_coeff = self.get_D_coeffs(like_params).copy()

        xz = np.log((1 + z) / (1 + self.z_X))

        # keep exponent for amplitude
        _ln_D_coeff[-1] = np.exp(_ln_D_coeff[-1])
        # for the redshift ev coeff, exp to reduce dynamic range
        if len(_ln_D_coeff) > 1:
            _ln_D_coeff[0] = signed_exp(_ln_D_coeff[0])

        ln_poly = np.poly1d(_ln_D_coeff)
        # redshift evolution linear
        return ln_poly(xz)

    def get_exp_damping(self, z, like_params=[]):
        """Exponent of damping at a given z"""

        A_coeff = self.get_A_coeffs(like_params)

        if A_coeff[-1] <= self.A_null_value:
            return 0

        xz = (1 + z) / (1 + self.z_X)
        poly = np.poly1d(A_coeff)
        return poly(xz)

    def get_dv_kms(self):
        """Velocity separation where the contamination is stronger"""

        # these constants should be written elsewhere
        lambda_lya = 1215.67
        c_kms = 2.997e5

        # we should properly compute this (check McDonald et al. 2006)
        # dv_kms = (lambda_lya-self.lambda_rest)/lambda_lya*c_kms
        # v3 in McDonald et al. (2006)
        dv_kms = np.log(lambda_lya / self.lambda_rest) * c_kms

        return dv_kms

    def get_contamination(self, z, k_kms, mF, like_params=[]):
        """Multiplicative contamination at a given z and k (in s/km).
        The mean flux (mF) is used scale it (see McDonald et al. 2006)"""
        print("made it to metal model cont")
        # Note that this represents "f" in McDonald et al. (2006)
        # It is later rescaled by <F> to compute "a" in eq. (15)
        f = self.get_amplitude(z, like_params=like_params)
        if f == 0:
            return 1
        alpha = self.get_exp_damping(z, like_params=like_params)
        print("midway model cont")
        if alpha == 0:
            damping = 1
        else:
            adim_damp = k_kms * self.get_damping(z, like_params=like_params)
            damping = (1 + adim_damp) ** alpha * np.exp(-1 * adim_damp**alpha)

        a = f / (1 - mF)
        cont = 1 + a**2 + 2 * a * np.cos(self.dv * k_kms) * damping
        print("end model cont")
        return cont

    def plot_contamination(
        self,
        z,
        k_kms,
        mF,
        ln_X_coeff=None,
        ln_D_coeff=None,
        A_coeff=None,
        plot_every_iz=1,
        cmap=None,
        smooth_k=False,
        dict_data=None,
        zrange=[0, 10],
        name=None,
    ):
        """Plot the contamination model"""

        # plot for fiducial value
        if ln_X_coeff is None:
            ln_X_coeff = self.ln_X_coeff
        if ln_D_coeff is None:
            ln_D_coeff = self.ln_D_coeff
        if A_coeff is None:
            A_coeff = self.A_coeff

        if cmap is None:
            cmap = get_discrete_cmap(len(z))

        metal_model = MetalModel(
            self.metal_label,
            ln_X_coeff=ln_X_coeff,
            ln_D_coeff=ln_D_coeff,
            A_coeff=A_coeff,
        )

        yrange = [1, 1]
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        fig2, ax2 = plt.subplots(
            len(z), sharex=True, sharey=True, figsize=(8, len(z) * 4)
        )
        if len(z) == 1:
            ax2 = [ax2]

        for ii in range(0, len(z), plot_every_iz):
            if smooth_k:
                k_use = np.logspace(
                    np.log10(k_kms[ii][0]), np.log10(k_kms[ii][-1]), 200
                )
            else:
                k_use = k_kms[ii]
            cont = metal_model.get_contamination(z[ii], k_use, mF[ii])
            if isinstance(cont, int):
                cont = np.ones_like(k_use)

            ax1.plot(k_use, cont, color=cmap(ii), label="z=" + str(z[ii]))
            ax2[ii].plot(k_use, cont, color=cmap(ii), label="z=" + str(z[ii]))

            yrange[0] = min(yrange[0], np.min(cont))
            yrange[1] = max(yrange[1], np.max(cont))

            if dict_data is not None:
                cont_data_res = metal_model.get_contamination(
                    z[ii], k_kms[ii], mF[ii]
                )
                if isinstance(cont_data_res, int):
                    cont_data_res = np.ones_like(k_kms[ii])

                yy = (
                    dict_data["p1d_data"][ii]
                    / dict_data["p1d_model"][ii]
                    * cont_data_res
                )
                err_yy = (
                    dict_data["p1d_err"][ii]
                    / dict_data["p1d_model"][ii]
                    * cont_data_res
                )
                if (z[ii] > zrange[1]) | (z[ii] < zrange[0]):
                    pass
                else:
                    ax1.errorbar(
                        dict_data["k_kms"][ii],
                        yy,
                        err_yy,
                        marker="o",
                        linestyle=":",
                        color=cmap(ii),
                        alpha=0.5,
                    )
                ax2[ii].errorbar(
                    dict_data["k_kms"][ii],
                    yy,
                    err_yy,
                    marker="o",
                    linestyle=":",
                    color=cmap(ii),
                    alpha=0.5,
                )

        ax1.axhline(1, color="k", linestyle=":")
        ax1.legend(ncol=4)
        ax1.set_ylim(yrange[0] * 0.95, yrange[1] * 1.05)
        ax1.set_xscale("log")
        ax1.set_xlabel(r"$k$ [1/Mpc]")
        ax1.set_ylabel(
            r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\,"
            + self.metal_label
            + "}$"
        )
        for ax in ax2:
            ax.axhline(1, color="k", linestyle=":")
            ax.legend()
            ax.set_ylim(yrange[0] * 0.95, yrange[1] * 1.05)
            ax.set_xlabel(r"$k$ [1/Mpc]")
            ax.set_ylabel(
                r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\,"
                + self.metal_label
                + "}$"
            )
            ax.set_xscale("log")

        fig1.tight_layout()
        fig2.tight_layout()

        if name is None:
            fig1.show()
            fig2.show()
        else:
            if len(z) != 1:
                fig1.savefig(name + "_all.pdf")
                fig1.savefig(name + "_all.png")
            fig2.savefig(name + "_z.pdf")
            fig2.savefig(name + "_z.png")

        return
