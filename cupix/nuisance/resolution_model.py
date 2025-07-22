import numpy as np
import copy, os
from matplotlib import pyplot as plt
from cupix.utils.utils import get_discrete_cmap
from cupix.likelihood import likelihood_parameter


def get_Rz(z, k_kms):
    # fig 32 https://arxiv.org/abs/2205.10939
    # lambda_AA = np.arange([3523.626, 3993.217, 4413.652, 4752.203, 5019.740, 5243.594, 5522.035, 5767.681, 5996.975, 6226.294, 6471.940, 6783.036])
    # resolution = np.array([2012.821, 2272.247, 2513.575, 2694.570, 2857.466, 2996.229, 3177.225, 3364.253, 3521.116, 3659.879, 3846.908, 4124.434])
    # rfit = np.polyfit(lambda_AA, resolution, 2)
    # plt.plot(lambda_AA, np.poly1d(rfit)(lambda_AA))

    c_kms = 2.99792458e5
    lya_AA = 1215.67
    rfit = np.array([4.53087663e-05, 1.70716005e-01, 8.60679006e02])
    R_coeff_lambda = np.poly1d(rfit)
    kms2AA = lya_AA * (1 + z) / c_kms
    # lambda_kms = lambda_AA * AA2kms
    k_AA = k_kms / kms2AA
    lambda_AA = 2 * np.pi / k_AA

    Rz = c_kms / (2.355 * R_coeff_lambda(lambda_AA))

    return Rz


class Resolution_Model(object):
    """New model for Resolution systematics"""

    def __init__(
        self,
        z_0=3.0,
        fid_R_coeff=[0, 0],
        R_coeff=None,
        free_param_names=None,
        Gauss_priors=None,
    ):
        self.z_0 = z_0
        self.Gauss_priors = Gauss_priors

        if R_coeff is not None:
            if free_param_names is not None:
                raise ValueError("can not specify coeff and free_param_names")
            self.R_coeff = R_coeff
        else:
            if free_param_names:
                # figure out number of Resolution_Model free params
                n_R = len([p for p in free_param_names if "R_coeff_" in p])
                if n_R == 0:
                    n_R = 1
            else:
                n_R = 1

            self.R_coeff = [0.0] * n_R
            self.R_coeff[-1] = fid_R_coeff[-1]
            if n_R == 2:
                self.R_coeff[-2] = fid_R_coeff[-2]

        self.set_R_parameters()

    def set_R_parameters(self):
        """Setup likelihood parameters in the Resolution_Model model"""

        self.R_params = []
        Npar = len(self.R_coeff)
        for i in range(Npar):
            name = "R_coeff_" + str(i)
            if i == 0:
                # 1.5% Gaussian prior, allow for 3 sigma
                xmin = -1
                xmax = +1
            else:
                # not optimized
                xmin = -10
                xmax = 10
            # note non-trivial order in coefficients
            Gwidth = None
            if self.Gauss_priors is not None:
                if name in self.Gauss_priors:
                    Gwidth = self.Gauss_priors[name][Npar - i - 1]
            value = self.R_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name,
                value=value,
                min_value=xmin,
                max_value=xmax,
                Gauss_priors_width=Gwidth,
            )
            self.R_params.append(par)

        return

    def get_Nparam(self):
        """Number of parameters in the model"""
        all_par = len(self.R_coeff)
        all_par2 = len(self.R_params)
        if all_par != all_par2:
            raise ValueError("parameter size mismatch")
        return all_par

    def get_R(self, z, like_params=[]):
        """Amplitude of Resolution_Model contamination around z_0"""

        R_coeff = self.get_R_coeffs(like_params=like_params)

        # xz = np.log((1 + z) / (1 + self.z_0))
        # ln_poly = np.poly1d(R_coeff)
        # ln_out = ln_poly(xz)
        # return np.exp(ln_out)

        xz = (1 + z) / (1 + self.z_0)
        poly = np.poly1d(R_coeff)
        return poly(xz)

    def get_parameters(self):
        """Return likelihood parameters for the Resolution_Model model"""
        return self.R_params

    def get_R_coeffs(self, like_params=[]):
        """Return list of mean flux coefficients"""

        if like_params:
            R_coeff = self.R_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "R_coeff_" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.R_coeff
            elif Npar != len(self.R_params):
                print(Npar, len(self.R_params))
                raise ValueError("number of params mismatch in get_R_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.R_params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.R_params[ip].name
                    )
                else:
                    R_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            R_coeff = self.R_coeff

        return R_coeff

    def get_contamination(self, z, k_kms, like_params=[]):
        """Multiplicative contamination caused by Resolution"""
        nelem = len(np.atleast_1d(z))
        res = []
        A = self.get_R(z, like_params=like_params)
        for ii in range(nelem):
            # if nelem == 1:
            #     print(A, get_Rz(z, k_kms), k_kms)
            #     return 1 + 1e-2 * A * get_Rz(z, k_kms) ** 2 * k_kms**2
            # else:
            res.append(
                1
                + 1e-2 * A[ii] * get_Rz(z[ii], k_kms[ii]) ** 2 * k_kms[ii] ** 2
            )

        return res

    def plot_contamination(
        self,
        z,
        k_kms,
        R_coeff=None,
        plot_every_iz=1,
        cmap=None,
        smooth_k=False,
        dict_data=None,
        zrange=[0, 10],
        name=None,
    ):
        """Plot the contamination model"""

        # plot for fiducial value
        if R_coeff is None:
            R_coeff = self.R_coeff

        if cmap is None:
            cmap = get_discrete_cmap(len(z))

        resolution_model = Resolution_Model(R_coeff=R_coeff)

        yrange = [1, 1]
        fig1, ax1 = plt.subplots(figsize=(8, 6))
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
            cont = resolution_model.get_contamination(z[ii], k_use)
            if isinstance(cont, int):
                cont = np.ones_like(k_use)

            ax1.plot(k_use, cont, color=cmap(ii), label="z=" + str(z[ii]))
            ax2[ii].plot(k_use, cont, color=cmap(ii), label="z=" + str(z[ii]))

            yrange[0] = min(yrange[0], np.min(cont))
            yrange[1] = max(yrange[1], np.max(cont))

            if dict_data is not None:
                yy = (
                    dict_data["p1d_data"][indz]
                    / dict_data["p1d_model"][indz]
                    * cont
                )
                err_yy = (
                    dict_data["p1d_err"][indz]
                    / dict_data["p1d_model"][indz]
                    * cont
                )

                ax1.errorbar(
                    dict_data["k_kms"][indz],
                    yy,
                    err_yy,
                    marker="o",
                    linestyle=":",
                    color=cmap(ii),
                    alpha=0.5,
                )
                ax2[ii].errorbar(
                    dict_data["k_kms"][indz],
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
        ax1.set_ylabel(r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\,Res}$")
        for ax in ax2:
            ax.axhline(1, color="k", linestyle=":")
            ax.legend()
            ax.set_ylim(yrange[0] * 0.95, yrange[1] * 1.05)
            ax.set_xlabel(r"$k$ [1/Mpc]")
            ax.set_ylabel(r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\,Res}$")
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
