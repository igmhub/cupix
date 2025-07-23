import numpy as np
import matplotlib.pyplot as plt
import copy
from cupix.utils.utils import get_discrete_cmap
from cupix.likelihood import likelihood_parameter
from cupix.nuisance.mean_flux_model_chunks import split_into_n_chunks


class MetalModel(object):
    """Model the contamination from Silicon Lya cross-correlations"""

    def __init__(
        self,
        metal_label,
        lambda_rest=None,
        z_X=3.0,
        ln_X_coeff=None,
        ln_A_coeff=None,
        X_fid_value=[0, -10],
        A_fid_value=[0, -9],
        Gauss_priors=None,
        X_null_value=-10.5,
        A_null_value=-8.5,
        free_param_names=None,
        X_zev_type="pivot",
        A_zev_type="pivot",
    ):
        """Model the evolution of a metal contamination (SiII or SiIII).
        We use a power law around z_X=3."""

        # label identifying the metal line
        self.metal_label = metal_label
        c_kms = 299792.458
        if metal_label == "SiIIa_SiIIb":
            self.lambda_rest = [1190.42, 1193.28]
            self.osc_strength = [0.277, 0.575]
        elif metal_label == "CIVa_CIVb":
            self.lambda_rest = [1548.20, 1550.78]
            self.osc_strength = [0.190, 0.095]
        elif metal_label == "MgIIa_MgIIb":
            self.lambda_rest = [2795.53, 2802.70]
            self.osc_strength = [0.608, 0.303]
        else:
            if lambda_rest is None:
                raise ValueError("need to specify lambda_rest", metal_label)
        self.dv = np.log(self.lambda_rest[1] / self.lambda_rest[0]) * c_kms
        self.ratio_f = np.min(self.osc_strength) / np.max(self.osc_strength)

        # power law pivot point
        self.z_X = z_X
        self.Gauss_priors = Gauss_priors
        # value below which no contamination (speed up model)
        self.X_null_value = X_null_value
        self.A_null_value = A_null_value
        # type of redshift ev for models
        self.X_zev_type = X_zev_type
        self.A_zev_type = A_zev_type

        # figure out parameters
        if (ln_X_coeff is not None) and (ln_A_coeff is not None):
            if free_param_names is not None:
                raise ValueError("cannot specify coeff and free_param_names")
            self.ln_X_coeff = ln_X_coeff
            self.ln_A_coeff = ln_A_coeff
        else:
            if free_param_names:
                # figure out number of free params for this metal line
                param_tag = "ln_x_" + metal_label + "_"
                n_X = len([p for p in free_param_names if param_tag in p])
                if n_X == 0:
                    n_X = 1

                param_tag = "ln_a_" + metal_label + "_"
                n_A = len([p for p in free_param_names if param_tag in p])
                if n_A == 0:
                    n_A = 1
            else:
                n_X = 1
                n_A = 1

            # set fiducial values
            if self.X_zev_type == "pivot":
                self.ln_X_coeff = np.zeros((n_X))
                if n_X == 1:
                    self.ln_X_coeff[0] = X_fid_value[-1]
                else:
                    for ii in range(n_X):
                        self.ln_X_coeff[ii] = X_fid_value[ii]
            else:
                self.ln_X_coeff = np.zeros((n_X)) + X_fid_value[-1]

            if self.A_zev_type == "pivot":
                self.ln_A_coeff = np.zeros((n_A))
                if n_A == 1:
                    self.ln_A_coeff[0] = A_fid_value[-1]
                else:
                    for ii in range(n_A):
                        self.ln_A_coeff[ii] = A_fid_value[ii]
            else:
                self.ln_A_coeff = np.zeros((n_A)) + A_fid_value[-1]

        # store list of likelihood parameters (might be fixed or free)
        self.n_X = len(self.ln_X_coeff)
        self.n_A = len(self.ln_A_coeff)
        self.set_X_parameters()
        self.set_A_parameters()

    def get_Nparam(self):
        """Number of parameters in the model"""
        all_par = self.n_X + self.n_A
        all_par2 = len(self.X_params) + len(self.A_params)
        if all_par != all_par2:
            raise ValueError("parameter size mismatch")
        return all_par

    def set_X_parameters(self):
        """Setup likelihood parameters for metal model"""

        self.X_params = []
        for i in range(self.n_X):
            name = "ln_x_" + self.metal_label + "_" + str(i)
            if self.X_zev_type == "pivot":
                if i == 0:
                    # ln of overall amplitude at z_X
                    xmin = -11  # no contamination
                    xmax = -0.5
                else:
                    xmin = -2
                    xmax = 2
                # note non-trivial order in coefficients
                value = self.ln_X_coeff[self.n_X - i - 1]
                Gwidth = None
                if self.Gauss_priors is not None:
                    if name in self.Gauss_priors:
                        Gwidth = self.Gauss_priors[name][-(i + 1)]
            else:
                xmin = -11
                xmax = -1.5
                value = self.ln_X_coeff[i]
                Gwidth = None

            par = likelihood_parameter.LikelihoodParameter(
                name=name,
                value=value,
                min_value=xmin,
                max_value=xmax,
                Gauss_priors_width=Gwidth,
            )
            self.X_params.append(par)
        return

    def set_A_parameters(self):
        """Setup likelihood parameters for metal model"""

        self.A_params = []
        for i in range(self.n_A):
            name = "ln_a_" + self.metal_label + "_" + str(i)
            if self.A_zev_type == "pivot":
                if i == 0:
                    xmin = -10
                    xmax = 10
                else:
                    xmin = -3
                    xmax = 3
                # note non-trivial order in coefficients
                value = self.ln_A_coeff[self.n_A - i - 1]
                Gwidth = None
                if self.Gauss_priors is not None:
                    if name in self.Gauss_priors:
                        Gwidth = self.Gauss_priors[name][-(i + 1)]
            else:
                xmin = -10
                xmax = 10
                value = self.ln_A_coeff[i]
                Gwidth = None

            par = likelihood_parameter.LikelihoodParameter(
                name=name,
                value=value,
                min_value=xmin,
                max_value=xmax,
                Gauss_priors_width=Gwidth,
            )
            self.A_params.append(par)
        return

    def get_X_parameters(self):
        """Return likelihood parameters from the metal model"""
        return self.X_params

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
                    if self.X_zev_type == "pivot":
                        ln_X_coeff[Npar - ip - 1] = array_values[_[0]]
                    else:
                        ln_X_coeff[ip] = array_values[_[0]]
        else:
            ln_X_coeff = self.ln_X_coeff

        return ln_X_coeff

    def get_A_coeffs(self, like_params=[]):
        """Return list of coefficients for metal model"""

        if like_params:
            ln_A_coeff = self.ln_A_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_a_" + self.metal_label + "_" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.ln_A_coeff
            elif Npar != len(self.A_params):
                raise ValueError("number of params mismatch in get_A_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.A_params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter " + self.A_params[ip].name
                    )
                else:
                    if self.A_zev_type == "pivot":
                        ln_A_coeff[Npar - ip - 1] = array_values[_[0]]
                    else:
                        ln_A_coeff[ip] = array_values[_[0]]
        else:
            ln_A_coeff = self.ln_A_coeff

        return ln_A_coeff

    def get_amplitude(self, z, like_params=[]):
        """Exponent of damping at a given z"""

        ln_X_coeff = self.get_X_coeffs(like_params)

        if self.X_zev_type == "pivot":
            if ln_X_coeff[-1] <= self.X_null_value:
                return None

            xz = (1 + z) / (1 + self.z_X)
            poly = np.poly1d(ln_X_coeff)
            return np.exp(poly(xz))
        else:
            _ = np.argwhere(ln_X_coeff <= self.X_null_value)[:, 0]
            if len(_) == self.n_X:
                return None
            else:
                return np.exp(ln_X_coeff)

    def get_exp_damping(self, z, like_params=[]):
        """Exponent of damping at a given z"""

        ln_A_coeff = self.get_A_coeffs(like_params)

        if self.A_zev_type == "pivot":
            if ln_A_coeff[-1] <= self.A_null_value:
                return None

            xz = (1 + z) / (1 + self.z_X)
            poly = np.poly1d(ln_A_coeff)
            return np.exp(poly(xz))
        else:
            _ = np.argwhere(ln_A_coeff <= self.A_null_value)[:, 0]
            if len(_) == self.n_A:
                return None
            else:
                return np.exp(ln_A_coeff)

    def get_contamination(self, z, k_kms, mF, like_params=[]):
        """Multiplicative contamination at a given z and k (in s/km)."""

        # Note that this represents "f" in McDonald et al. (2006)
        # It is later rescaled by <F> to compute "a" in eq. (15)
        f = self.get_amplitude(z, like_params=like_params)
        if f is None:
            return 0

        # We damp the oscillations using a sigmoidal function
        # https://en.wikipedia.org/wiki/Generalised_logistic_function

        # The rapid the damping increases
        damping_growth = self.get_exp_damping(z, like_params=like_params)
        if damping_growth is None:
            damping_growth = np.zeros_like(z)
        else:
            damping_growth = np.zeros_like(z) + damping_growth

        if len(z) == 1:
            damping = np.exp(-1 * damping_growth**2 * k_kms[0] ** 2)
            cont = (
                f
                * (
                    1
                    + self.ratio_f**2
                    + 2 * self.ratio_f * np.cos(self.dv * k_kms[0])
                )
                * damping
            )
        else:
            cont = []
            for iz in range(len(z)):
                damping = np.exp(-1 * damping_growth[iz] ** 2 * k_kms[iz] ** 2)
                _cont = (
                    f[iz]
                    * (
                        1
                        + self.ratio_f**2
                        + 2 * self.ratio_f * np.cos(self.dv * k_kms[iz])
                    )
                    * damping
                )
                cont.append(_cont)

        return cont

    def plot_contamination(
        self,
        z,
        k_kms,
        mF,
        ln_X_coeff=None,
        ln_A_coeff=None,
        plot_every_iz=1,
        cmap=None,
        smooth_k=False,
        dict_data=None,
        zrange=[0, 10],
        name=None,
        plot_panels=True,
        func_rebin=None,
    ):
        """Plot the contamination model"""

        # plot for fiducial value
        if ln_X_coeff is None:
            ln_X_coeff = self.ln_X_coeff
        if ln_A_coeff is None:
            ln_A_coeff = self.ln_A_coeff

        if cmap is None:
            cmap = get_discrete_cmap(len(z))

        metal_model = MetalModel(
            self.metal_label, ln_X_coeff=ln_X_coeff, ln_A_coeff=ln_A_coeff
        )

        yrange = [0, 0]
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        if plot_panels:
            fig2, ax2 = plt.subplots(
                len(z), sharex=True, sharey=True, figsize=(8, len(z) * 4)
            )
            if len(z) == 1:
                ax2 = [ax2]

        for ii in range(0, len(z), plot_every_iz):
            if dict_data is not None:
                indz = np.argwhere(np.abs(dict_data["zs"] - z[ii]) < 1.0e-3)[
                    :, 0
                ]
                if len(indz) != 1:
                    continue
                else:
                    indz = indz[0]

            if (z[ii] > zrange[1]) | (z[ii] < zrange[0]):
                continue

            if smooth_k:
                k_use = np.logspace(
                    np.log10(k_kms[ii][0]), np.log10(k_kms[ii][-1]), 200
                )
            else:
                k_use = k_kms[ii]

            cont = metal_model.get_contamination(
                np.array([z[ii]]), [k_use], mF[ii]
            )

            if isinstance(cont, int):
                cont = np.zeros_like(k_use)
            else:
                if smooth_k == False:
                    cont_data_res = func_rebin([z[ii]], [cont])[0]

            ax1.plot(
                k_use, cont, color=cmap(ii), label="z=" + str(z[ii]), alpha=0.5
            )
            if plot_panels:
                ax2[ii].plot(
                    k_use,
                    cont,
                    color=cmap(ii),
                    label="z=" + str(z[ii]),
                    alpha=0.5,
                )

            yrange[0] = min(yrange[0], np.min(cont))
            yrange[1] = max(yrange[1], np.max(cont))

            if dict_data is not None:
                yy = dict_data["p1d_data"][indz] - (
                    dict_data["p1d_model"][indz] - cont_data_res
                )
                err_yy = dict_data["p1d_err"][indz]
                ax1.errorbar(
                    dict_data["k_kms"][indz],
                    yy,
                    err_yy,
                    marker="o",
                    linestyle=":",
                    color=cmap(ii),
                    alpha=0.5,
                )
                ax1.plot(
                    dict_data["k_kms"][indz],
                    cont_data_res,
                    marker="",
                    linestyle="--",
                    color=cmap(ii),
                    alpha=0.5,
                )
                if plot_panels:
                    ax2[ii].errorbar(
                        dict_data["k_kms"][indz],
                        yy,
                        err_yy,
                        marker="o",
                        linestyle=":",
                        color=cmap(ii),
                        alpha=0.5,
                    )

        ax1.axhline(0, color="k", linestyle=":")
        ax1.legend(ncol=4)
        # ax1.set_ylim(yrange[0] * 1.05, yrange[1] * 1.05)
        # ax1.set_xscale("log")
        ax1.set_xlabel(r"$k$ [1/Mpc]")
        ax1.set_ylabel(
            r"$P_\mathrm{1D}-P_\mathrm{1D}^\mathrm{no\,"
            + self.metal_label
            + "}$"
        )
        fig1.tight_layout()

        if plot_panels:
            for ax in ax2:
                ax.axhline(1, color="k", linestyle=":")
                ax.legend()
                # ax.set_ylim(yrange[0] * 1.05, yrange[1] * 1.05)
                ax.set_xlabel(r"$k$ [1/Mpc]")
                ax.set_ylabel(
                    r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\,"
                    + self.metal_label
                    + "}$"
                )
                # ax.set_xscale("log")
            fig2.tight_layout()

        if name is None:
            fig1.show()
            if plot_panels:
                fig2.show()
        else:
            if len(z) != 1:
                fig1.savefig(name + "_all.pdf")
                fig1.savefig(name + "_all.png")

            if plot_panels:
                fig2.savefig(name + "_z.pdf")
                fig2.savefig(name + "_z.png")

        return
