import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.interpolate import make_interp_spline
from cupix.likelihood import likelihood_parameter


class Contaminant(object):
    """New model for HCD contamination"""

    def __init__(
        self,
        coeffs=None,
        list_coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_vals=None,
        null_vals=None,
        z_max=None,
        flat_priors=None,
        Gauss_priors=None,
    ):
        # store input data
        self.list_coeffs = list_coeffs
        self.z_0 = z_0
        self.fid_vals = fid_vals
        self.null_vals = null_vals
        self.Gauss_priors = Gauss_priors
        self.flat_priors = flat_priors
        self.z_max = z_max

        # set prop_coeffs (only for interp, not pivot)
        self.prop_coeffs = {}
        for key in self.list_coeffs:
            try:
                self.prop_coeffs[key + "_otype"] = prop_coeffs[key + "_otype"]
            except KeyError:
                raise ValueError("must specify otype in prop_coeffs for:", key)
            try:
                self.prop_coeffs[key + "_ztype"] = prop_coeffs[key + "_ztype"]
            except KeyError:
                raise ValueError("must specify ztype in prop_coeffs for:", key)

            if prop_coeffs[key + "_ztype"].startswith("interp"):
                try:
                    self.prop_coeffs[key + "_znodes"] = prop_coeffs[
                        key + "_znodes"
                    ]
                except KeyError:
                    raise ValueError("must specify zs in prop_coeffs for:", key)

        self.coeffs = {}
        if coeffs is not None:
            if free_param_names is not None:
                raise ValueError(
                    "can not specify both coeffs and free_param_names"
                )
            for key in self.list_coeffs:
                # set coeffs
                if key in coeffs:
                    self.coeffs[key] = coeffs[key]
                else:
                    raise ("Coeff not specified:", key)
        else:
            if free_param_names is None:
                raise ValueError(
                    "must specify either coeffs or free_param_names"
                )

            # figure out number of HCD free params
            self.n_pars = {}
            for key in self.list_coeffs:
                self.n_pars[key] = len(
                    [p for p in free_param_names if key + "_" in p]
                )
                if self.n_pars[key] == 0:
                    npar = 1
                else:
                    npar = self.n_pars[key]
                self.coeffs[key] = [0.0] * npar

                for ii in range(npar):
                    if self.prop_coeffs[key + "_ztype"] == "pivot":
                        # self.coeffs[key][-(ii + 1)] = self.fid_vals[key][
                        #     -(ii + 1)
                        # ]
                        if ii == 0:
                            self.coeffs[key][-1] = self.fid_vals[key][-1]
                        else:
                            self.coeffs[key][-(ii + 1)] = self.fid_vals[key][0]
                    else:
                        self.coeffs[key][ii] = self.fid_vals[key][-1]

        self.set_params()

    def set_params(self):
        """Setup likelihood parameters in the HCD model"""

        self.params = {}

        for key in self.list_coeffs:
            values = self.coeffs[key]
            for ii in range(len(values)):
                name = key + "_" + str(ii)
                set_prior = False
                for key2 in self.flat_priors:
                    if key2 in name:
                        if self.prop_coeffs[key + "_ztype"] == "pivot":
                            # xmin = self.flat_priors[key2][-(ii + 1)][0]
                            # xmax = self.flat_priors[key2][-(ii + 1)][1]
                            if ii == 0:
                                xmin = self.flat_priors[key2][-1][0]
                                xmax = self.flat_priors[key2][-1][1]
                            else:
                                xmin = self.flat_priors[key2][0][0]
                                xmax = self.flat_priors[key2][0][1]
                        else:
                            xmin = self.flat_priors[key2][-1][0]
                            xmax = self.flat_priors[key2][-1][1]
                        set_prior = True
                        break

                if set_prior is False:
                    raise ValueError("Cannot find priors of:", key)

                # note non-trivial order in coefficients
                Gwidth = None
                if self.Gauss_priors is not None:
                    if name in self.Gauss_priors:
                        if self.prop_coeffs[key + "_ztype"] == "pivot":
                            Gwidth = self.Gauss_priors[name][-(ii + 1)]
                        else:
                            Gwidth = self.Gauss_priors[name][ii]

                if self.prop_coeffs[key + "_ztype"] == "pivot":
                    _value = values[-(ii + 1)]
                else:
                    _value = values[ii]

                par = likelihood_parameter.LikelihoodParameter(
                    name=name,
                    value=_value,
                    min_value=xmin,
                    max_value=xmax,
                    Gauss_priors_width=Gwidth,
                )
                self.params[name] = par

    def get_Nparam(self):
        """Number of parameters in the model"""
        n_params = len(self.params)
        n_coeffs = 0
        for coeff in self.coeffs:
            n_coeffs += len(coeff)
        if n_params != n_coeffs:
            raise ValueError("mismatch between number of params and coeffs")
        return n_params

    def get_value(self, name, z, like_params=[]):
        coeff = self.get_coeff(name, like_params=like_params)

        if self.prop_coeffs[name + "_ztype"] == "pivot":
            xz = np.log((1 + z) / (1 + self.z_0))
            ln_poly = np.poly1d(coeff)
            ln_out = ln_poly(xz)
        elif self.prop_coeffs[name + "_ztype"].startswith("interp"):
            if self.prop_coeffs[name + "_ztype"].endswith("_lin"):
                ln_out = np.interp(z, self.prop_coeffs[name + "_znodes"], coeff)
            elif self.prop_coeffs[name + "_ztype"].endswith("_spl"):
                f_out = make_interp_spline(
                    self.prop_coeffs[name + "_znodes"],
                    coeff,
                    k=1,
                )
                ln_out = f_out(z)
            elif self.prop_coeffs[name + "_ztype"].endswith("_smspl"):
                f_out = make_smoothing_spline(
                    self.prop_coeffs[name + "_znodes"], coeff
                )
                ln_out = f_out(z)
            else:
                raise ValueError(
                    "prop_coeffs must be interp_lin, interp_spl, or interp_smspl for",
                    name,
                )
        else:
            raise ValueError("prop_coeffs must be interp or pivot for", name)

        if self.z_max is not None:
            if len(np.atleast_1d(z)) > 1:
                _ = z >= self.z_max[name]
                if np.any(_):
                    ln_out[_] = self.null_vals[name]
            else:
                if z >= self.z_max[name]:
                    ln_out = self.null_vals[name]

        if self.prop_coeffs[name + "_otype"] == "const":
            return ln_out
        elif self.prop_coeffs[name + "_otype"] == "exp":
            return np.exp(ln_out)
        else:
            raise ValueError("prop_coeffs must be const or exp for", name)

    def get_parameter(self, name):
        return self.params[name]

    def get_parameters(self):
        """Return likelihood parameters"""
        return self.params

    def get_coeff(self, name, like_params=[]):
        if like_params:
            coeff = self.coeffs[name].copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if (name + "_") in par.name:
                    array_names.append(par.name)
                    array_values.append(par.value)
                    Npar += 1
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # return fiducial value
            if Npar == 0:
                return coeff
            elif Npar != self.n_pars[name]:
                print(Npar, self.n_pars[name])
                raise ValueError("number of params mismatch for: " + name)

            for ii in range(Npar):
                ind_arr = np.argwhere(name + "_" + str(ii) == array_names)[0, 0]
                if self.prop_coeffs[name + "_ztype"] == "pivot":
                    coeff[-(ii + 1)] = array_values[ind_arr]
                else:
                    coeff[ii] = array_values[ind_arr]
        else:
            coeff = self.coeffs[name]

        return coeff

    def reset_coeffs(self, like_params):
        """Reset all coefficients to fiducial values"""
        for name in self.coeffs:
            Npar = 0
            print("orig", name, self.coeffs[name])
            array_names = []
            array_values = []
            for par in like_params:
                if (name + "_") in par.name:
                    array_names.append(par.name)
                    array_values.append(par.value)
                    Npar += 1
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # return fiducial value
            if Npar == 0:
                continue
            elif Npar != self.n_pars[name]:
                print(Npar, self.n_pars[name])
                raise ValueError("number of params mismatch for: " + name)

            for ii in range(Npar):
                ind_arr = np.argwhere(name + "_" + str(ii) == array_names)[0, 0]
                if self.prop_coeffs[name + "_ztype"] == "pivot":
                    self.coeffs[name][-(ii + 1)] = array_values[ind_arr]
                else:
                    self.coeffs[name][ii] = array_values[ind_arr]
            print("new", name, self.coeffs[name])

    def plot_parameters(self, z, like_params, folder=None):
        """Plot likelihood parameters"""

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(
            len(self.coeffs), 1, sharex=True, figsize=(8, 3 * len(self.coeffs))
        )
        if len(self.coeffs) == 1:
            ax = [ax]

        try:
            len_p = len(like_params[0])
        except:
            z_at_time = False
        else:
            z_at_time = True

        vals_out = {}
        coeffs_out = {}

        for ii, key in enumerate(self.coeffs.keys()):
            if z_at_time == False:
                vals = self.get_value(key, z, like_params=like_params)
                coeffs_out[key] = self.get_coeff(key, like_params=like_params)
            else:
                vals = []
                coeffs_out[key] = []
                for jj in range(len(z)):
                    vals.append(
                        self.get_value(key, z[jj], like_params=like_params[jj])
                    )
                    coeffs_out[key].append(
                        self.get_coeff(key, like_params=like_params[jj])[0]
                    )
                vals = np.array(vals)

            if key in self.null_vals:
                if np.all(vals == self.null_vals[key]):
                    continue
            elif key == "HCD_const":
                if np.all(vals == 0):
                    continue

            if self.prop_coeffs[key + "_otype"] == "exp":
                vals = np.log(vals)

            vals_out[key] = vals

            _ = vals != self.null_vals[key]
            ax[ii].plot(z[_], vals[_], "o-", label="data")
            xz = np.log((1 + z) / (1 + self.z_0))
            if np.any(_):
                res = np.polyfit(xz[_], vals[_], 1)
                ax[ii].plot(z[_], res[0] * xz[_] + res[1], "--", label="fit")
                ax[ii].set_ylabel(key)

        ax[0].legend()
        ax[-1].set_xlabel("z")

        plt.tight_layout()
        plt.show()
        if folder is not None:
            fig.savefig(folder + ".png")
            fig.savefig(folder + ".pdf")

        return vals_out, coeffs_out

    # def plot_contamination(
    #     self,
    #     z,
    #     k_kms,
    #     mF=1,
    #     coeffs=None,
    #     plot_every_iz=1,
    #     cmap=None,
    #     dict_data=None,
    #     zrange=[0, 10],
    #     name=None,
    # ):
    #     """Plot the contamination model"""

    #     from matplotlib import pyplot as plt
    #     from cup1d.utils.utils import get_discrete_cmap

    #     # plot for fiducial value
    #     if coeffs is None:
    #         coeffs = self.coeffs
    #     else:
    #         for key in self.list_coeffs:
    #             if key not in coeffs:
    #                 coeffs[key] = self.coeffs[key]

    #     if cmap is None:
    #         cmap = get_discrete_cmap(len(z))

    #     # new to be updated!!!!
    #     hcd_model = HCD_Model_new2(coeffs=coeffs)

    #     cont = hcd_model.get_contamination(z, k_kms)

    #     # cont = metal_model.get_contamination(
    #     #         np.array([z[ii]]), [k_use], mF[ii]
    #     #     )

    #     # if isinstance(cont, int):
    #     #     cont = np.ones_like(k_use)
    #     # else:
    #     #     if smooth_k == False:
    #     #         cont_data_res = func_rebin([z[ii]], [cont])[0]

    #     yrange = [1, 1]
    #     fig1, ax1 = plt.subplots(figsize=(8, 6))
    #     fig2, ax2 = plt.subplots(
    #         len(z), sharex=True, sharey=True, figsize=(8, len(z) * 4)
    #     )
    #     if len(z) == 1:
    #         ax2 = [ax2]

    #     for ii in range(0, len(z), plot_every_iz):
    #         if dict_data is not None:
    #             indz = np.argwhere(np.abs(dict_data["zs"] - z[ii]) < 1.0e-3)[
    #                 :, 0
    #             ]
    #             if len(indz) != 1:
    #                 continue
    #             else:
    #                 indz = indz[0]

    #         if (z[ii] > zrange[1]) | (z[ii] < zrange[0]):
    #             continue

    #         ax1.plot(
    #             k_kms[ii], cont[ii], color=cmap(ii), label="z=" + str(z[ii])
    #         )
    #         ax2[ii].plot(
    #             k_kms[ii], cont[ii], color=cmap(ii), label="z=" + str(z[ii])
    #         )

    #         yrange[0] = min(yrange[0], np.min(cont))
    #         yrange[1] = max(yrange[1], np.max(cont))

    #         if dict_data is not None:
    #             yy = (
    #                 dict_data["p1d_data"][indz]
    #                 / dict_data["p1d_model"][indz]
    #                 * cont[ii]
    #             )
    #             err_yy = (
    #                 dict_data["p1d_err"][indz]
    #                 / dict_data["p1d_model"][indz]
    #                 * cont[ii]
    #             )

    #             ax1.errorbar(
    #                 dict_data["k_kms"][indz],
    #                 yy,
    #                 err_yy,
    #                 marker="o",
    #                 linestyle=":",
    #                 color=cmap(ii),
    #                 alpha=0.5,
    #             )
    #             ax2[ii].errorbar(
    #                 dict_data["k_kms"][indz],
    #                 yy,
    #                 err_yy,
    #                 marker="o",
    #                 linestyle=":",
    #                 color=cmap(ii),
    #                 alpha=0.5,
    #             )

    #     ax1.axhline(1, color="k", linestyle=":")
    #     ax1.legend(ncol=4)
    #     ax1.set_ylim(yrange[0] * 0.95, yrange[1] * 1.05)
    #     ax1.set_xscale("log")
    #     ax1.set_xlabel(r"$k$ [1/Mpc]")
    #     ax1.set_ylabel(
    #         r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\," + name + "}$"
    #     )
    #     for ax in ax2:
    #         ax.axhline(1, color="k", linestyle=":")
    #         ax.legend()
    #         ax.set_ylim(yrange[0] * 0.95, yrange[1] * 1.05)
    #         ax.set_xlabel(r"$k$ [1/Mpc]")
    #         ax.set_ylabel(
    #             r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\," + name + "}$"
    #         )
    #         ax.set_xscale("log")

    #     fig1.tight_layout()
    #     fig2.tight_layout()

    #     if name is None:
    #         fig1.show()
    #         fig2.show()
    #     else:
    #         if len(z) != 1:
    #             fig1.savefig(name + "_all.pdf")
    #             fig1.savefig(name + "_all.png")
    #         fig2.savefig(name + "_z.pdf")
    #         fig2.savefig(name + "_z.png")
