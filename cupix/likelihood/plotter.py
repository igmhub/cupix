import inspect
import matplotlib.pyplot as plt
from corner import corner
import numpy as np
import os
from cupix.utils.utils import get_discrete_cmap, get_path_repo, purge_chains


class Plotter(object):
    def __init__(
        self,
        fitter=None,
        save_directory=None,
        fname_chain=None,
        zmask=None,
        args={},
    ):
        self.zmask = zmask
        if fitter is not None:
            self.fitter = fitter
        elif fname_chain is not None:
            from cupix.likelihood.input_pipeline import Args
            from cupix.likelihood.pipeline import Pipeline

            # load file with chain
            data = np.load(fname_chain, allow_pickle=True).item()

            # set input args to pipeline and evaluate
            args_possible = inspect.signature(Args)
            dict_input = {}
            for param in args_possible.parameters.values():
                if param.name in data["args"].keys():
                    if param.name in args:
                        dict_input[param.name] = args[param.name]
                    else:
                        dict_input[param.name] = data["args"][param.name]
                    # print(param.name, data["args"][param.name])
                else:
                    print(param.name)

            args = Args(**dict_input)
            # args.p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v6.fits"
            self.fitter = Pipeline(args, out_folder=save_directory).fitter

            # add sampler results to fitter
            self.fitter.mle_cube = data["fitter"]["mle_cube"]
            self.fitter.mle_cosmo = data["fitter"]["mle_cosmo"]
            self.fitter.mle = data["fitter"]["mle"]
            self.fitter.lnprop_mle = data["fitter"]["lnprob_mle"]
            if "lnprob" in data["fitter"].keys():
                self.fitter.lnprob = data["fitter"]["lnprob"]
                self.fitter.chain = data["fitter"]["chain"]
                self.fitter.blobs = data["fitter"]["blobs"]
        else:
            ValueError("Provide either fitter or fname_chain")

        self.cmap = get_discrete_cmap(len(self.fitter.like.data.z))
        self.save_directory = save_directory
        if save_directory is not None:
            os.makedirs(save_directory, exist_ok=True)

        self.mle_values = self.fitter.get_best_fit(stat_best_fit="mle")
        self.like_params = self.fitter.like.parameters_from_sampling_point(
            self.mle_values
        )
        self.mle_results = self.fitter.like.plot_px(
            values=self.mle_values,
            plot_every_iz=1,
            return_all=True,
            show=False,
            zmask=zmask,
        )

    def plots_minimizer(self, zrange=[0, 10], zmask=None):
        if self.zmask is not None:
            zmask = self.zmask
            zrange = [np.min(zmask) - 0.01, np.max(zmask) + 0.01]
        plt.close("all")
        # plot initial P1D (before fitting)
        self.plot_P1D_initial(residuals=False, zmask=zmask)
        plt.close()
        self.plot_P1D_initial(residuals=True, zmask=zmask)
        plt.close()

        # plot best fit
        self.plot_p1d(residuals=False, stat_best_fit="mle", zmask=zmask)
        plt.close()
        self.plot_p1d(residuals=True, stat_best_fit="mle", zmask=zmask)
        plt.close()
        self.plot_p1d(
            residuals=True, stat_best_fit="mle", zmask=zmask, plot_panels=True
        )
        plt.close()

        # plot errors
        self.plot_p1d_errors()
        plt.close()

        # plot cosmology
        if self.fitter.fix_cosmology == False:
            self.plot_mle_cosmo()
            plt.close()

        # plot IGM histories
        self.plot_igm(cloud=True, zmask=zmask)
        plt.close()

        # plot contamination
        self.plot_hcd_cont(plot_data=True, zrange=zrange)
        plt.close()
        self.plot_metal_cont(smooth_k=False, plot_data=True, zrange=zrange)
        plt.close()
        self.plot_agn_cont(plot_data=True, zrange=zrange)
        plt.close()

        # plot fit in hull
        self.plot_hull(zmask=zmask)
        plt.close()

    def plots_sampler(self):
        # plot lnprob
        plt.close("all")
        self.plot_lnprob()
        plt.close()

        # plot initial P1D (before fitting)
        self.plot_P1D_initial(residuals=False)
        plt.close()
        self.plot_P1D_initial(residuals=True)
        plt.close()

        # plot best fit
        self.plot_p1d(residuals=False, stat_best_fit="mle")
        plt.close()
        self.plot_p1d(residuals=True, stat_best_fit="mle")
        plt.close()
        self.plot_p1d(residuals=True, stat_best_fit="mle", plot_panels=True)
        plt.close()

        # plot errors
        self.plot_p1d_errors()
        plt.close()

        # plot cosmology
        if self.fitter.fix_cosmology == False:
            self.plot_corner(only_cosmo=True, only_cosmo_lims=False)
            plt.close()
            self.plot_corner(only_cosmo=True, only_cosmo_lims=True)
            plt.close()

        # plot corner
        self.plot_corner()
        plt.close()

        # plot IGM histories
        self.plot_igm(cloud=True)
        plt.close()

        # plot contamination
        self.plot_hcd_cont()
        plt.close()
        self.plot_metal_cont()
        # plt.close()
        self.plot_agn_cont()
        plt.close()

        # plot fit in hull
        self.plot_hull()
        plt.close()

    def get_hc_star(self, nyx_version="Jul2024"):
        suite_emu = self.fitter.like.theory.emulator.list_sim_cube[0][:3]
        if suite_emu == "mpg":
            fname = os.path.join(
                get_path_repo("lace"),
                "data",
                "sim_suites",
                "Australia20",
                "mpg_emu_cosmo.npy",
            )
        elif suite_emu == "nyx":
            fname = os.path.join(
                os.environ["NYX_PATH"], "nyx_emu_cosmo_" + nyx_version + ".npy"
            )
        else:
            ValueError("cosmo_label should be 'mpg' or 'nyx'")

        try:
            data_cosmo = np.load(fname, allow_pickle=True).item()
        except:
            ValueError(f"{fname} not found")

        labs = []
        delta2_star = np.zeros(len(data_cosmo))
        n_star = np.zeros(len(data_cosmo))
        alpha_star = np.zeros(len(data_cosmo))

        for ii, key in enumerate(data_cosmo):
            labs.append(key)
            delta2_star[ii] = data_cosmo[key]["star_params"]["Delta2_star"]
            n_star[ii] = data_cosmo[key]["star_params"]["n_star"]
            alpha_star[ii] = data_cosmo[key]["star_params"]["alpha_star"]
        return labs, delta2_star, n_star, alpha_star, suite_emu, data_cosmo

    def plot_mle_cosmo(self, fontsize=16):
        """Plot MLE cosmology"""

        (
            labs,
            delta2_star,
            n_star,
            alpha_star,
            suite_emu,
            data_cosmo,
        ) = self.get_hc_star()

        if suite_emu == "mpg":
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax = [ax]
        else:
            fig, ax = plt.subplots(1, 3, figsize=(14, 6))

        ax[0].scatter(delta2_star, n_star)
        if suite_emu != "mpg":
            ax[1].scatter(delta2_star, alpha_star)
            ax[2].scatter(n_star, alpha_star)

        dif0 = delta2_star.max() - delta2_star.min()
        dif1 = n_star.max() - n_star.min()
        dif2 = alpha_star.max() - alpha_star.min()
        sep_x = 0.01
        for ii, lab in enumerate(labs):
            if data_cosmo[lab]["sim_label"][-1].isdigit():
                sep_y = 0
            else:
                sep_y = 0.01
            ax[0].annotate(
                lab[4:],
                (delta2_star[ii] + sep_x * dif0, n_star[ii] + sep_y * dif1),
                fontsize=8,
            )
            if suite_emu != "mpg":
                ax[1].annotate(
                    lab[4:],
                    (
                        delta2_star[ii] + sep_x * dif0,
                        alpha_star[ii] + sep_y * dif2,
                    ),
                    fontsize=8,
                )
                ax[2].annotate(
                    lab[4:],
                    (n_star[ii] + sep_x * dif1, alpha_star[ii] + sep_y * dif2),
                    fontsize=8,
                )

        for ii in range(2):
            # best
            if ii == 0:
                marker = "X"
                color = "C1"
                Delta2_star = self.fitter.mle_cosmo["Delta2_star"]
                n_star = self.fitter.mle_cosmo["n_star"]
                if suite_emu != "mpg":
                    alpha_star = self.fitter.mle_cosmo["alpha_star"]
                label = "MLE"
            # truth
            else:
                if self.fitter.like.truth is not None:
                    marker = "s"
                    color = "C3"
                    Delta2_star = self.fitter.like.truth["linP"]["Delta2_star"]
                    n_star = self.fitter.like.truth["linP"]["n_star"]
                    if suite_emu != "mpg":
                        alpha_star = self.fitter.like.truth["linP"][
                            "alpha_star"
                        ]
                    label = "truth"
                else:
                    continue

            ax[0].scatter(
                Delta2_star, n_star, marker=marker, color=color, label=label
            )
            if suite_emu != "mpg":
                ax[1].scatter(
                    Delta2_star,
                    alpha_star,
                    marker=marker,
                    color=color,
                    label=label,
                )
                ax[2].scatter(
                    n_star, alpha_star, marker=marker, color=color, label=label
                )

        ax[0].set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
        ax[0].set_ylabel(r"$n_\star$", fontsize=fontsize)
        ax[0].legend()
        if suite_emu != "mpg":
            ax[1].set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
            ax[1].set_ylabel(r"$\alpha_\star$", fontsize=fontsize)
            ax[2].set_xlabel(r"$n_\star$", fontsize=fontsize)
            ax[2].set_ylabel(r"$\alpha_\star$", fontsize=fontsize)
        plt.tight_layout()

        if self.save_directory is not None:
            name = self.save_directory + "/cosmo_mle"
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_corner_chainconsumer(
        self,
        plot_params=None,
        delta_lnprob_cut=None,
        usetex=True,
        serif=True,
        only_cosmo=False,
        extra_nburn=0,
    ):
        """Make corner plot in ChainConsumer
        - plot_params: Pass a list of parameters to plot (in LaTeX form),
                    or leave as None to
                    plot all (including derived)
        - if delta_lnprob_cut is set, keep only high-prob points"""

        from chainconsumer import ChainConsumer, Chain, Truth
        import pandas as pd

        params_plot, strings_plot, _ = self.fitter.get_all_params(
            delta_lnprob_cut=delta_lnprob_cut, extra_nburn=extra_nburn
        )
        if only_cosmo:
            yesplot = ["$\\Delta^2_\\star$", "$n_\\star$"]
            for param in self.fitter.like.free_params:
                if param.name == "nrun":
                    yesplot.append("$\\alpha_\\star$")
        else:
            diff = np.max(params_plot, axis=0) - np.min(params_plot, axis=0)
            yesplot = np.array(strings_plot)[diff != 0]

        dict_pd = {}
        for ii, par in enumerate(strings_plot):
            if par in yesplot:
                dict_pd[par] = params_plot[:, ii]
        pd_data = pd.DataFrame(data=dict_pd)

        c = ChainConsumer()
        chain = Chain(
            samples=pd_data,
            name="a",
        )
        c.add_chain(chain)
        # summary = c.analysis.get_summary()["a"]
        if self.fitter.truth is not None:
            c.add_truth(
                Truth(
                    location=self.fitter.truth,
                    line_style="--",
                    color="C1",
                )
            )
        c.add_truth(
            Truth(
                location=self.fitter.mle,
                line_style=":",
                color="C2",
            )
        )

        fig = c.plotter.plot(figsize=(12, 12))

        if self.save_directory is not None:
            if only_cosmo:
                name = self.save_directory + "/corner_cosmo"
            else:
                name = self.save_directory + "/corner"
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")

        # return summary

    def plot_corner(
        self,
        delta_lnprob_cut=None,
        usetex=True,
        only_cosmo=False,
        extra_nburn=0,
        only_cosmo_lims=True,
        extra_data=None,
    ):
        """Make corner plot in corner"""

        params_plot, strings_plot, _ = self.fitter.get_all_params(
            delta_lnprob_cut=delta_lnprob_cut, extra_nburn=extra_nburn
        )
        if only_cosmo:
            yesplot = ["$\\Delta^2_\\star$", "$n_\\star$"]
            if "nrun" in self.fitter.like.free_param_names:
                yesplot.append("$\\alpha_\\star$")
        else:
            # only plot parameters that vary
            diff = np.max(params_plot, axis=0) - np.min(params_plot, axis=0)
            yesplot = np.array(strings_plot)[diff != 0]

        truth = np.zeros((len(yesplot)))
        MLE = np.zeros((len(yesplot)))
        chain = np.zeros((params_plot.shape[0], len(yesplot)))
        for ii, par in enumerate(yesplot):
            _ = np.argwhere(np.array(strings_plot) == par)[0, 0]
            chain[:, ii] = params_plot[:, _]
            if self.fitter.truth is not None:
                if par in self.fitter.truth:
                    truth[ii] = self.fitter.truth[par]
            if par in self.fitter.mle:
                MLE[ii] = self.fitter.mle[par]
            if par == "$A_s$":
                chain[:, ii] *= 1e9
                truth[ii] *= 1e9
                MLE[ii] *= 1e9
                yesplot[ii] = r"$A_s\times10^9$"

        fig = corner(
            chain,
            labels=yesplot,
            quantiles=(0.16, 0.5, 0.84),
            levels=(0.68, 0.95),
            show_titles=True,
            title_quantiles=(0.16, 0.5, 0.84),
            title_fmt=".4f",
            plot_datapoints=False,
            plot_density=False,
        )

        # add truth and MLE
        value1 = truth.copy()
        value2 = MLE.copy()
        ndim = len(value1)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            if self.fitter.truth is not None:
                ax.axvline(value1[i], color="C0", label="Truth")
            ax.axvline(value2[i], color="C2", label="MAP", linestyle=":")

            # Set up x limits
            xlim = np.array(ax.get_xlim())
            if self.fitter.truth is not None:
                val_min = np.nanmin([value1[i], value2[i]])
                val_max = np.nanmax([value1[i], value2[i]])
            else:
                val_min = np.nanmin([value2[i]])
                val_max = np.nanmax([value2[i]])

            if (xlim[0] > val_min) and np.isfinite(val_min):
                xlim[0] = val_min
            if (xlim[1] < val_max) and np.isfinite(val_max):
                xlim[1] = val_max
            xdiff = xlim[1] - xlim[0]
            ax.set_xlim(xlim[0] - 0.05 * xdiff, xlim[1] + 0.05 * xdiff)
        axes[0, 0].legend(loc="upper left")

        ## show flat priors for all parameters
        for xi in range(ndim):
            for par in self.fitter.like.free_params:
                if self.fitter.param_dict[par.name] == yesplot[xi]:
                    for yi in range(xi, ndim):
                        axes[yi, xi].axvline(
                            par.min_value, color="r", linestyle="-"
                        )
                        axes[yi, xi].axvline(
                            par.max_value, color="r", linestyle="-"
                        )
                    break
        for yi in range(ndim):
            for par in self.fitter.like.free_params:
                if self.fitter.param_dict[par.name] == yesplot[yi]:
                    for xi in range(yi):
                        axes[yi, xi].axhline(
                            par.min_value, color="r", linestyle="-"
                        )
                        axes[yi, xi].axhline(
                            par.max_value, color="r", linestyle="-"
                        )
                    break

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]

                if self.fitter.truth is not None:
                    ax.axvline(value1[xi], color="C0")
                    ax.axhline(value1[yi], color="C0")
                    ax.plot(value1[xi], value1[yi], ".C0")

                ax.axvline(value2[xi], color="C2", linestyle=":")
                ax.axhline(value2[yi], color="C2", linestyle=":")
                ax.plot(value2[xi], value2[yi], ".C2", linestyle=":")

                # Set up x and y limits
                xlim = np.array(ax.get_xlim())
                if self.fitter.truth is not None:
                    val_min = np.min([value1[xi], value2[xi]])
                    val_max = np.max([value1[xi], value2[xi]])
                else:
                    val_min = np.min([value2[xi]])
                    val_max = np.max([value2[xi]])
                if (xlim[0] > val_min) and np.isfinite(val_min):
                    xlim[0] = val_min
                if (xlim[1] < val_max) and np.isfinite(val_max):
                    xlim[1] = val_max
                xdiff = xlim[1] - xlim[0]
                ax.set_xlim(xlim[0] - 0.05 * xdiff, xlim[1] + 0.05 * xdiff)

                ylim = np.array(ax.get_ylim())
                if self.fitter.truth is not None:
                    val_min = np.min([value1[yi], value2[yi]])
                    val_max = np.max([value1[yi], value2[yi]])
                else:
                    val_min = np.min([value2[yi]])
                    val_max = np.max([value2[yi]])
                if (ylim[0] > val_min) and np.isfinite(val_min):
                    ylim[0] = val_min
                if (ylim[1] < val_max) and np.isfinite(val_max):
                    ylim[1] = val_max
                ydiff = ylim[1] - ylim[0]
                ax.set_ylim(ylim[0] - 0.05 * ydiff, ylim[1] + 0.05 * ydiff)

        if (extra_data is not None) and only_cosmo:
            axes[1, 0].scatter(
                extra_data[:, 0], extra_data[:, 1], color="C3", s=2
            )
            if "nrun" in self.fitter.like.free_param_names:
                axes[2, 0].scatter(
                    extra_data[:, 0], extra_data[:, 2], color="C3", s=2
                )
                axes[2, 1].scatter(
                    extra_data[:, 1], extra_data[:, 2], color="C3", s=2
                )

        if only_cosmo:
            (
                labs,
                delta2_star,
                n_star,
                alpha_star,
                suite_emu,
                data_cosmo,
            ) = self.get_hc_star()

            if "nrun" in self.fitter.like.free_param_names:
                axs = np.array(fig.axes).reshape((3, 3))
                nproj = 3
            else:
                axs = np.array(fig.axes).reshape((2, 2))
                nproj = 1

            for ii in range(nproj):
                if ii == 0:
                    x = delta2_star
                    y = n_star
                    _ = np.argwhere(np.array(yesplot) == "$n_\\star$")[0, 0]
                    ychain = chain[:, _]
                    ax = axs[1, 0]
                elif ii == 1:
                    x = delta2_star
                    y = alpha_star
                    _ = np.argwhere(np.array(yesplot) == "$\\alpha_\\star$")[
                        0, 0
                    ]
                    ychain = chain[:, _]
                    ax = axs[2, 0]
                else:
                    x = n_star
                    y = alpha_star
                    _ = np.argwhere(np.array(yesplot) == "$\\alpha_\\star$")[
                        0, 0
                    ]
                    ychain = chain[:, _]
                    ax = axs[2, 1]

                ax.scatter(x, y, marker="o", color="C1", alpha=0.5)

                if only_cosmo_lims:
                    diff = 0.05 * (y.max() - y.min())
                    ax.set_ylim(
                        np.min([y.min(), ychain.min()]) - diff,
                        np.max([y.max(), ychain.max()]) + diff,
                    )

            for ii in range(len(axs)):
                if ii == 0:
                    x = delta2_star
                    _ = np.argwhere(np.array(yesplot) == "$\\Delta^2_\\star$")[
                        0, 0
                    ]
                    xchain = chain[:, _]
                elif ii == 1:
                    x = n_star
                    _ = np.argwhere(np.array(yesplot) == "$n_\\star$")[0, 0]
                    xchain = chain[:, _]
                else:
                    x = alpha_star
                    _ = np.argwhere(np.array(yesplot) == "$\\alpha_\\star$")[
                        0, 0
                    ]
                    xchain = chain[:, _]

                if only_cosmo_lims:
                    for jj in range(len(axs)):
                        diff = 0.05 * (x.max() - x.min())
                        axs[jj, ii].set_xlim(
                            np.min([x.min(), xchain.min()]) - diff,
                            np.max([x.max(), xchain.max()]) + diff,
                        )

        if self.save_directory is not None:
            if only_cosmo:
                if only_cosmo_lims:
                    name = self.save_directory + "/corner_cosmo_full"
                else:
                    name = self.save_directory + "/corner_cosmo"
            else:
                name = self.save_directory + "/corner"
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")

    def plot_lnprob(self, extra_nburn=0):
        """Plot lnprob"""

        mask, _ = purge_chains(self.fitter.lnprob[extra_nburn:, :])

        for ii in range(self.fitter.lnprob.shape[1]):
            if ii in mask:
                plt.plot(self.fitter.lnprob[extra_nburn:, ii], alpha=0.5)
            else:
                plt.plot(self.fitter.lnprob[extra_nburn:, ii], "--", alpha=0.01)

        if self.save_directory is not None:
            plt.savefig(self.save_directory + "/lnprob.pdf")

    def plot_p1d(
        self,
        values=None,
        plot_every_iz=1,
        residuals=False,
        rand_posterior=None,
        stat_best_fit="mle",
        zmask=None,
        plot_panels=False,
        z_at_time=False,
    ):
        """Plot the P1D of the data and the emulator prediction
        for the MCMC best fit
        """

        ## Get best fit values for each parameter
        if values is None:
            values = self.mle_values

        if plot_panels:
            if residuals == False:
                plot_panels = False

        if self.save_directory is not None:
            if rand_posterior is None:
                fname = "P1D_mle"
            else:
                fname = "best_fit_" + stat_best_fit + "_err_posterior"
            if residuals:
                plot_fname = self.save_directory + "/" + fname + "_residuals"
                if plot_panels:
                    plot_fname += "_panels"
            else:
                plot_fname = self.save_directory + "/" + fname
        else:
            plot_fname = None

        self.fitter.like.plot_p1d(
            values=values,
            plot_every_iz=plot_every_iz,
            residuals=residuals,
            rand_posterior=rand_posterior,
            plot_fname=plot_fname,
            zmask=zmask,
            plot_panels=plot_panels,
            z_at_time=z_at_time,
        )

    def plot_p1d_errors(self, values=None, zmask=None):
        """Plot the P1D of the data and the emulator prediction
        for the MCMC best fit
        """

        ## Get best fit values for each parameter
        if values is None:
            values = self.mle_values

        if self.save_directory is not None:
            fname = "P1D_mle_errors"
            plot_fname = self.save_directory + "/" + fname
        else:
            plot_fname = None

        z_at_time = False
        if zmask is not None:
            if len(zmask) == 1:
                z_at_time = True

        self.fitter.like.plot_p1d_errors(
            values=values,
            plot_fname=plot_fname,
            zmask=zmask,
            z_at_time=z_at_time,
        )

    def plot_P1D_initial(self, plot_every_iz=1, residuals=False, zmask=None):
        """Plot the P1D of the data and the emulator prediction
        for the fiducial model"""

        if self.save_directory is not None:
            if residuals:
                plot_fname = self.save_directory + "/P1D_initial_residuals"
            else:
                plot_fname = self.save_directory + "/P1D_initial"
        else:
            plot_fname = None

        self.fitter.like.plot_p1d(
            values=None,
            plot_every_iz=plot_every_iz,
            residuals=residuals,
            plot_fname=plot_fname,
            zmask=zmask,
        )

    def plot_histograms(self, cube=False, delta_lnprob_cut=None):
        """Make histograms for all dimensions, using re-normalized values if
        cube=True
        - if delta_lnprob_cut is set, use only high-prob points"""

        # get chain (from sampler or from file)
        chain, lnprob, blobs = self.fitter.get_chain(
            delta_lnprob_cut=delta_lnprob_cut
        )
        plt.figure()

        for ip in range(self.fitter.ndim):
            param = self.fitter.like.free_params[ip]
            if cube:
                values = chain[:, ip]
                title = param.name + " in cube"
            else:
                cube_values = chain[:, ip]
                values = param.value_from_cube(cube_values)
                title = param.name

            plt.hist(values, 100, color="k", histtype="step")
            plt.title(title)
            plt.show()

        return

    def plot_igm(
        self,
        value=None,
        rand_sample=None,
        stat_best_fit="mle",
        cloud=True,
        zmask=None,
    ):
        """Plot IGM histories"""

        if value is None:
            value = self.mle_values

        self.fitter.like.plot_igm(
            cloud=cloud,
            free_params=self.like_params,
            stat_best_fit=stat_best_fit,
            save_directory=self.save_directory,
            zmask=zmask,
        )

        # if rand_sample is not None:
        #     # chain, lnprob, blobs = self.fitter.get_chain()
        #     # nn = min(chain.shape[0], nn)
        #     # mask = np.random.permutation(chain.shape[0])[:nn]
        #     # rand_sample = chain[mask]

        #     # sample the chain to get errors on IGM parameters
        #     pars_samp = {}
        #     pars_samp["tau_eff"] = np.zeros((nn, len(z)))
        #     pars_samp["gamma"] = np.zeros((nn, len(z)))
        #     pars_samp["sigT_kms"] = np.zeros((nn, len(z)))
        #     pars_samp["kF_kms"] = np.zeros((nn, len(z)))
        #     for ii in range(nn):
        #         like_params = self.fitter.like.parameters_from_sampling_point(
        #             rand_sample[ii]
        #         )
        #         models = self.fitter.like.theory.update_igm_models(like_params)
        #         pars_samp["tau_eff"][ii] = models["F_model"].get_tau_eff(z)
        #         pars_samp["gamma"][ii] = models["T_model"].get_gamma(z)
        #         pars_samp["sigT_kms"][ii] = models["T_model"].get_sigT_kms(z)
        #         pars_samp["kF_kms"][ii] = models["P_model"].get_kF_kms(z)

        #     if rand_sample is not None:
        #         err = np.abs(
        #             np.percentile(pars_samp[arr_labs[ii]], [16, 84], axis=0)
        #             - pars_best[arr_labs[ii]]
        #         )
        #         ax[ii].errorbar(
        #             pars_best["z"],
        #             pars_best[arr_labs[ii]],
        #             err,
        #             label="best-fitting",
        #             alpha=0.75,
        #         )

    def compare_corners(
        self,
        chain_files,
        labels,
        plot_params=None,
        save_string=None,
        rootdir=None,
        subfolder=None,
        delta_lnprob_cut=None,
        usetex=True,
        serif=True,
    ):
        """Function to take a list of chain files and overplot the chains
        Pass a list of chain files (ints) and a list of labels (strings)
         - plot_params: list of parameters (in code variables, not latex form)
                        to plot if only a subset is desired
         - save_string: to save the plot. Must include
                        file extension (i.e. .pdf, .png etc)
         - if delta_lnprob_cut is set, keep only high-prob points"""

        from chainconsumer import ChainConsumer, Chain, Truth

        assert len(chain_files) == len(labels)

        truth_dict = {}
        c = ChainConsumer()

        ## Add each chain we want to plot
        for aa, chain_file in enumerate(chain_files):
            sampler = EmceeSampler(
                read_chain_file=chain_file, subfolder=subfolder, rootdir=rootdir
            )
            params, strings, _ = sampler.get_all_params(
                delta_lnprob_cut=delta_lnprob_cut
            )
            c.add_chain(params, parameters=strings, name=labels[aa])

            ## Do not check whether truth results are the same for now
            ## Take the longest truth dictionary for disjoint chains
            if len(sampler.truth) > len(truth_dict):
                truth_dict = sampler.truth

        c.configure(
            diagonal_tick_labels=False,
            tick_font_size=15,
            label_font_size=25,
            max_ticks=4,
            usetex=usetex,
            serif=serif,
        )

        if plot_params == None:
            fig = c.plotter.plot(figsize=(15, 15), truth=truth_dict)
        else:
            ## From plot_param list, build list of parameter
            ## strings to plot
            plot_param_strings = []
            for par in plot_params:
                plot_param_strings.append(self.fitter.param_dict[par])
            fig = c.plotter.plot(
                figsize=(10, 10),
                parameters=plot_param_strings,
                truth=truth_dict,
            )
        if save_string:
            fig.savefig("%s" % save_string)
        fig.show()

        return

    def plot_hcd_cont(
        self,
        plot_every_iz=1,
        smooth_k=False,
        plot_data=False,
        zrange=[0, 10],
    ):
        """Function to plot the HCD contamination"""

        if plot_data:
            dict_data = self.mle_results
        else:
            dict_data = None

        list_params = {}
        Npar_damp = 0
        Npar_scale = 0
        for p in self.fitter.like.free_params:
            if ("A_damp" in p.name) | ("A_scale" in p.name):
                key = self.fitter.param_dict[p.name]
                list_params[p.name] = self.fitter.mle[key]
                print(p.name, self.fitter.mle[key])
            if "A_damp" in p.name:
                Npar_damp += 1
            elif "A_scale" in p.name:
                Npar_scale += 1

        Npar = len(list_params)
        if Npar == 0:
            return

        # print(Npar, Npar_damp, Npar_scale)

        if Npar_scale != 0:
            A_scale_coeff = np.zeros(Npar_scale)
            for ii in range(Npar_scale):
                name = "ln_A_scale_" + str(ii)
                # note non-trivial order in coefficients
                A_scale_coeff[Npar_scale - ii - 1] = list_params[name]
        else:
            A_scale_coeff = None

        if Npar_damp != 0:
            A_damp_coeff = np.zeros(Npar_damp)
            for ii in range(Npar_damp):
                name = "ln_A_damp_" + str(ii)
                # note non-trivial order in coefficients
                A_damp_coeff[Npar_damp - ii - 1] = list_params[name]
        else:
            A_damp_coeff = None

        if self.save_directory is not None:
            name = self.save_directory + "/HCD_cont"
        else:
            name = None

        self.fitter.like.theory.model_cont.hcd_model.plot_contamination(
            self.fitter.like.data.z,
            self.fitter.like.data.k_AA,
            ln_A_damp_coeff=A_damp_coeff,
            ln_A_scale_coeff=A_scale_coeff,
            plot_every_iz=plot_every_iz,
            cmap=self.cmap,
            smooth_k=smooth_k,
            dict_data=dict_data,
            zrange=zrange,
            name=name,
        )

    def plot_metal_cont(
        self,
        plot_every_iz=1,
        stat_best_fit="mle",
        smooth_k=False,
        plot_data=False,
        zrange=[0, 10],
        mle_results=None,
        plot_panels=True,
    ):
        """Function to plot metal contamination"""

        if plot_data:
            if mle_results is not None:
                dict_data = mle_results
            else:
                dict_data = self.mle_results
        else:
            dict_data = None

        # get mean flux from best-fitting model
        # mF = self.fitter.like.theory.model_igm.F_model.get_mean_flux(
        #     np.array(self.fitter.like.data.z), self.like_params
        # )
        # get fiducial mean flux
        mF = self.fitter.like.theory.model_igm.F_model.get_mean_flux(
            np.array(self.fitter.like.data.z),
        )

        # plot contamination of all metals
        metal_models = self.fitter.like.theory.model_cont.metal_models
        for model_name in metal_models:
            metal = metal_models[model_name].metal_label

            x_list_params = {}
            d_list_params = {}
            l_list_params = {}
            a_list_params = {}
            for p in self.fitter.like.free_params:
                if "ln_x_" + metal + "_" in p.name:
                    key = self.fitter.param_dict[p.name]
                    x_list_params[p.name] = self.fitter.mle[key]
                    print(p.name, self.fitter.mle[key])
                if "ln_d_" + metal + "_" in p.name:
                    key = self.fitter.param_dict[p.name]
                    d_list_params[p.name] = self.fitter.mle[key]
                    print(p.name, self.fitter.mle[key])
                if "ln_l_" + metal + "_" in p.name:
                    key = self.fitter.param_dict[p.name]
                    l_list_params[p.name] = self.fitter.mle[key]
                    print(p.name, self.fitter.mle[key])
                if "a_" + metal + "_" in p.name:
                    key = self.fitter.param_dict[p.name]
                    a_list_params[p.name] = self.fitter.mle[key]
                    print(p.name, self.fitter.mle[key])

            x_Npar = len(x_list_params)
            ln_X_coeff = np.zeros(x_Npar)
            for ii in range(x_Npar):
                name = "ln_x_" + metal + "_" + str(ii)
                # note non-trivial order in coefficients
                ln_X_coeff[x_Npar - ii - 1] = x_list_params[name]

            d_Npar = len(d_list_params)
            ln_D_coeff = np.zeros(d_Npar)
            for ii in range(d_Npar):
                name = "ln_d_" + metal + "_" + str(ii)
                # note non-trivial order in coefficients
                ln_D_coeff[d_Npar - ii - 1] = d_list_params[name]

            l_Npar = len(l_list_params)
            ln_L_coeff = np.zeros(l_Npar)
            for ii in range(l_Npar):
                name = "ln_l_" + metal + "_" + str(ii)
                # note non-trivial order in coefficients
                ln_L_coeff[l_Npar - ii - 1] = l_list_params[name]

            a_Npar = len(a_list_params)
            A_coeff = np.zeros(a_Npar)
            for ii in range(a_Npar):
                name = "a_" + metal + "_" + str(ii)
                # note non-trivial order in coefficients
                A_coeff[a_Npar - ii - 1] = a_list_params[name]

            if (
                (x_Npar == 0)
                and (d_Npar == 0)
                and (l_Npar == 0)
                and (a_Npar == 0)
            ):
                continue

            if x_Npar == 0:
                ln_X_coeff = None
            if d_Npar == 0:
                ln_D_coeff = None
            if l_Npar == 0:
                ln_N_coeff = None
            if a_Npar == 0:
                A_coeff = None

            if self.save_directory is not None:
                name = self.save_directory + "/" + metal + "_cont"
            else:
                name = None

            metal_models[model_name].plot_contamination(
                self.fitter.like.data.z,
                self.fitter.like.data.k_AA,
                mF,
                ln_X_coeff=ln_X_coeff,
                D_coeff=ln_D_coeff,
                L_coeff=ln_L_coeff,
                A_coeff=A_coeff,
                plot_every_iz=plot_every_iz,
                cmap=self.cmap,
                smooth_k=smooth_k,
                dict_data=dict_data,
                zrange=zrange,
                name=name,
                plot_panels=plot_panels,
            )

    def plot_agn_cont(
        self,
        plot_every_iz=1,
        smooth_k=False,
        plot_data=False,
        zrange=[0, 10],
    ):
        """Function to plot AGN contamination"""

        if plot_data:
            dict_data = self.mle_results
        else:
            dict_data = None

        list_params = {}
        for p in self.fitter.like.free_params:
            if "ln_AGN" in p.name:
                key = self.fitter.param_dict[p.name]
                list_params[p.name] = self.fitter.mle[key]
                print(p.name, self.fitter.mle[key])

        Npar = len(list_params)
        if Npar == 0:
            return
        coeff = np.zeros(Npar)
        for ii in range(Npar):
            name = "ln_AGN_" + str(ii)
            # note non-trivial order in coefficients
            coeff[Npar - ii - 1] = list_params[name]

        if self.save_directory is not None:
            name = self.save_directory + "/AGN_cont"
        else:
            name = None

        self.fitter.like.theory.model_cont.agn_model.plot_contamination(
            self.fitter.like.data.z,
            self.fitter.like.data.k_AA,
            coeff,
            plot_every_iz=plot_every_iz,
            cmap=self.cmap,
            smooth_k=smooth_k,
            dict_data=dict_data,
            zrange=zrange,
            name=name,
        )

    def plot_res_cont(
        self,
        plot_every_iz=1,
        smooth_k=False,
        plot_data=False,
        zrange=[0, 10],
    ):
        """Function to plot AGN contamination"""

        if plot_data:
            dict_data = self.mle_results
        else:
            dict_data = None

        list_params = {}
        for p in self.fitter.like.free_params:
            if "R_coeff" in p.name:
                key = self.fitter.param_dict[p.name]
                list_params[p.name] = self.fitter.mle[key]
                print(p.name, self.fitter.mle[key])

        Npar = len(list_params)
        if Npar == 0:
            return
        coeff = np.zeros(Npar)
        for ii in range(Npar):
            name = "R_coeff_" + str(ii)
            # note non-trivial order in coefficients
            coeff[Npar - ii - 1] = list_params[name]

        if self.save_directory is not None:
            name = self.save_directory + "/resolution_cont"
        else:
            name = None

        self.fitter.like.theory.model_syst.resolution_model.plot_contamination(
            self.fitter.like.data.z,
            self.fitter.like.data.k_AA,
            R_coeff=coeff,
            plot_every_iz=plot_every_iz,
            cmap=self.cmap,
            smooth_k=smooth_k,
            dict_data=dict_data,
            zrange=zrange,
            name=name,
        )

    def plot_hull(self, p0=None, save_plot=True, zmask=None):
        """Function to plot data within hull"""

        if p0 is None:
            p0 = self.fitter.mle_cube

        like_params = self.fitter.like.parameters_from_sampling_point(p0)

        if zmask is not None:
            _z = zmask
        else:
            _z = self.fitter.like.data.z

        emu_call, M_of_z, blob = self.fitter.like.theory.get_emulator_calls(
            _z,
            like_params=like_params,
            return_M_of_z=True,
            return_blob=True,
        )

        p1 = np.zeros(
            (
                self.fitter.like.theory.hull.nz,
                len(self.fitter.like.theory.hull.params),
            )
        )
        for jj, key in enumerate(self.fitter.like.theory.hull.params):
            p1[:, jj] = emu_call[key]

        self.fitter.like.theory.hull.plot_hulls(p1)

        if self.save_directory is not None:
            name = self.save_directory + "/fit_in_hull"
            if save_plot:
                plt.savefig(name + ".pdf")
                plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_illustrate_contaminants(
        self, values, zmask, fontsize=18, lines_use=None
    ):
        all_contaminants = np.array(lines_use + ["DLA", "res", ""])

        # cont2label = {
        #     "Lya_SiIII": "+ Lya-SiIII(1206)",
        #     "DLA": "+ HCDs",
        #     "Lya_SiII": "+ Lya-SiII(1193)",
        #     "res": "+ resolution",
        #     "SiII_SiII": "+ SiII(1190)-SiII(1193)",
        #     "SiIIa_SiIII": "+ SiII(1190)-SiIII(1206)",
        #     "SiIIb_SiIII": "+ SiII(1193)-SiIII(1206)",
        # }
        cont2label = {
            "Lya_SiIII": r"Ly$\alpha$-SiIII",
            "DLA": "HCDs",
            "Lya_SiIIa": r"Ly$\alpha$-SiIIa",
            "Lya_SiIIb": r"Ly$\alpha$-SiIIb",
            "res": "resolution",
            "SiIIa_SiIIb": "SiII(1190)-SiII(1193)",
            "SiIIa_SiIII": "SiII(1190)-SiIII",
            "SiIIb_SiIII": "SiII(1193)-SiIII",
        }

        _data_z = []
        _data_k_AA = []
        _data_theta_bin_deg = []
        _data_Pk_AA = []
        _data_ePk_AA = []
        _data_icov_AA = []
        for iz in range(len(self.fitter.like.data.z)):
            _ = np.argwhere(np.abs(zmask - self.fitter.like.data.z[iz]) < 1e-3)
            if len(_) != 0:
                _data_z.append(self.fitter.like.data.z[iz])
                _data_k_AA.append(self.fitter.like.data.k_AA[iz])
                _data_Pk_AA.append(self.fitter.like.data.Pk_AA[iz])
                _data_ePk_AA.append(
                    np.sqrt(np.diag(self.fitter.like.data.cov_Pk_AA[iz]))
                )
                _data_theta_bin_deg.append(self.fitter.like.data.theta_bin_deg[iz])
                _data_icov_AA.append(
                    np.linalg.inv(self.fitter.like.data.cov_Pk_AA[iz])
                )
        _data_z = np.array(_data_z)

        # compute which contaminants produce greatest change in chi2

        chi2_each = np.zeros(len(all_contaminants))
        for icont, conts in enumerate(all_contaminants):
            _values = values.copy()
            if "DLA" in conts:
                ind = np.argwhere(
                    np.array(self.fitter.like.free_param_names) == "ln_A_damp_0"
                )[0, 0]
                _values[ind] = 0
            if "res" in conts:
                ind = np.argwhere(
                    np.array(self.fitter.like.free_param_names) == "R_coeff_0"
                )[0, 0]
                _values[ind] = 0.5

            for line in self.fitter.like.theory.model_cont.metal_lines:
                if (line in conts) & (
                    "ln_x_" + line + "_0" in self.fitter.like.free_param_names
                ):
                    ind = np.argwhere(
                        np.array(self.fitter.like.free_param_names)
                        == "ln_x_" + line + "_0"
                    )[0, 0]
                    _values[ind] = 0.0

            _res = self.fitter.like.get_px_AA(_data_z, _data_k_AA, _theta_bin_deg, _values)
            if len(_res[0]) == 1:
                _res = _res[0][0]
            else:
                _res = _res[0]
            diff = _data_Pk_kms[0] - _res
            chi2_each[icont] = np.dot(np.dot(_data_icov_kms[0], diff), diff)

        ind = np.argsort(chi2_each[:-1] - chi2_each[-1])[::-1]

        remove_contaminants = []
        labels = []
        remove_contaminants.append(list(all_contaminants[:-1]))
        labels.append("base")
        for ii in range(len(ind)):
            _remove = []
            labels.append(cont2label[all_contaminants[ind[ii]]])
            for cont in all_contaminants[:-1]:
                if cont not in (all_contaminants[:-1])[ind[: ii + 1]]:
                    _remove.append(cont)
            remove_contaminants.append(_remove)
        labels.append("full")
        # for ii in range(len(remove_contaminants)):
        #     print(ii, remove_contaminants[ii])
        # for ii in range(len(labels)):
        #     print(ii, labels[ii])

        emu_p1d = []
        chi2_all = []
        for conts in remove_contaminants:
            _values = values.copy()
            if "DLA" in conts:
                ind = np.argwhere(
                    np.array(self.fitter.like.free_param_names) == "ln_A_damp_0"
                )[0, 0]
                _values[ind] = 0
            if "res" in conts:
                ind = np.argwhere(
                    np.array(self.fitter.like.free_param_names) == "R_coeff_0"
                )[0, 0]
                _values[ind] = 0.5

            for line in self.fitter.like.theory.model_cont.metal_lines:
                if (line in conts) & (
                    "ln_x_" + line + "_0" in self.fitter.like.free_param_names
                ):
                    ind = np.argwhere(
                        np.array(self.fitter.like.free_param_names)
                        == "ln_x_" + line + "_0"
                    )[0, 0]
                    _values[ind] = 0.0

            _res = self.fitter.like.get_p1d_kms(_data_z, _data_k_kms, _values)
            if len(_res[0]) == 1:
                _res = _res[0][0]
            else:
                _res = _res[0]
            emu_p1d.append(_res)

            diff = _data_Pk_kms[0] - _res
            chi2_all.append(np.dot(np.dot(_data_icov_kms[0], diff), diff))

        nax = len(all_contaminants) // 2
        naxres = len(all_contaminants) % 2
        fig, ax = plt.subplots(
            nax + naxres,
            2,
            sharex=True,
            sharey="row",
            figsize=(12, (nax + naxres) * 2.5),
        )
        ax = ax.reshape(-1)

        for ii in range(len(emu_p1d)):
            if ii == 0:
                lab = labels[ii]
            else:
                lab = "... + " + labels[ii]
            ax[ii].errorbar(
                _data_k_kms[0],
                _data_Pk_kms[0] / emu_p1d[ii] - 1,
                _data_ePk_kms[0] / emu_p1d[ii],
                color="C0",
                ls=":",
                marker=".",
                label=lab,
            )
            if ii != len(emu_p1d) - 1:
                ax[ii].plot(
                    _data_k_kms[0],
                    emu_p1d[ii + 1] / emu_p1d[ii] - 1,
                    "C1-",
                    label=labels[ii + 1],
                )
                ax[ii].text(
                    0.05,
                    0.1,
                    r"$\Delta\chi^2=$"
                    + str(np.round(chi2_all[ii + 1] - chi2_all[ii], 1)),
                    fontsize=fontsize - 2,
                    transform=ax[ii].transAxes,
                )
            else:
                ax[ii].text(
                    0.05,
                    0.1,
                    r"$\chi^2=$" + str(np.round(chi2_all[ii], 1)),
                    fontsize=fontsize - 2,
                    transform=ax[ii].transAxes,
                )
            ax[ii].axhline(color="k", ls=":", alpha=0.5)

            ax[ii].tick_params(
                axis="both", which="major", labelsize=fontsize - 2
            )
            _handles, _labels = ax[ii].get_legend_handles_labels()
            if ii != len(emu_p1d) - 1:
                order = [1, 0]
            else:
                order = [0]
            ax[ii].legend(
                [_handles[idx] for idx in order],
                [_labels[idx] for idx in order],
                loc="upper right",
                fontsize=fontsize - 4,
            )
            # ax[ii].legend(loc="upper right", fontsize=fontsize - 2)
        ax[-2].set_xlabel(r"$k_\parallel$ [s/km]", fontsize=fontsize)
        ax[-1].set_xlabel(r"$k_\parallel$ [s/km]", fontsize=fontsize)
        # fig.suptitle(r"$z=$" + str(zmask[0]), fontsize=fontsize + 2)
        if naxres == 1:
            ax[-1].axis("off")

        fig.supylabel(
            # r"$P_{\rm 1D}/P_{\rm 1D}^{\rm model}-1$",
            r"Residuals",
            x=0.01,
            fontsize=fontsize + 2,
        )
        plt.tight_layout()

        # ax[0].set_ylim(-1.05*max_min, 1.05*max_min)
        # ax[0].set_ylim(-0.2, 0.2)

        if self.save_directory is not None:
            name = self.save_directory + "/cont_illustrate" + str(zmask[0])
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()


def plot_cov(
    p1d_fname, kmin=1e-3, nknyq=0.5, fontsize=14, save_directory=None, lab=""
):
    from astropy.io import fits

    try:
        hdu = fits.open(p1d_fname)
    except:
        raise ValueError("Cannot read: ", p1d_fname)

    if "fft" in p1d_fname:
        type_measurement = "FFT"
    elif "qmle" in p1d_fname:
        type_measurement = "QMLE"
    else:
        raise ValueError("Cannot find type_measurement in: ", p1d_fname)

    dict_with_keys = {}
    for ii in range(len(hdu)):
        if "EXTNAME" in hdu[ii].header:
            dict_with_keys[hdu[ii].header["EXTNAME"]] = ii

    pk = hdu[dict_with_keys["P1D_BLIND"]].data["PLYA"]
    stat = hdu[dict_with_keys["COVARIANCE_STAT"]].data
    syst = hdu[dict_with_keys["SYSTEMATICS"]].data

    # print(hdu[dict_with_keys["SYSTEMATICS"]].header)

    if type_measurement == "QMLE":
        sys_labels = [
            "E_DLA_COMPLETENESS",
            "E_BAL_COMPLETENESS",
            "E_RESOLUTION",
            "E_CONTINUUM",
            "E_CONTINUUM_ADD",
            "E_NOISE_SCALE",
            "E_NOISE_ADD",
        ]

    elif type_measurement == "FFT":
        sys_labels = [
            "E_PSF",
            "E_RESOLUTION",
            "E_SIDE_BAND",
            "E_LINES",
            "E_DLA",
            "E_BAL",
            "E_CONTINUUM",
            "E_DLA_COMPLETENESS",
            "E_BAL_COMPLETENESS",
        ]

    zz_unique = np.unique(syst["Z"])
    nz = len(zz_unique)
    nelem = len(sys_labels)

    diag_stat = np.sqrt(np.diag(stat))

    nax = int(np.ceil(nz / 3))

    fig, ax = plt.subplots(nax, 3, sharex=True, sharey=True, figsize=(20, 16))
    ax = ax.reshape(-1)

    for iz in range(len(zz_unique)):
        dv = 2.99792458e5 * 0.8 / 1215.67 / (1 + zz_unique[iz])
        k_nyq = np.pi / dv
        ind = np.argwhere(
            (syst["Z"] == zz_unique[iz])
            # & (diag_cov_raw > 0)
            # & (diag_cov_raw < max_cov)
            # & np.isfinite(Pk_kms_raw)
            # & np.isfinite(diag_cov_raw)
            & (syst["K"] > kmin)
            & (syst["K"] < k_nyq * nknyq)
        )[:, 0]

        for ii in range(nelem):
            y = syst[sys_labels[ii]][ind] / pk[ind]
            _ = np.isfinite(y) & (y > 0)
            ax[iz].plot(
                (syst["K"][ind])[_],
                y[_],
                color="C" + str(ii),
                label=sys_labels[ii],
            )

        ax[iz].plot(
            syst["K"][ind],
            diag_stat[ind] / pk[ind],
            color="k",
            label="stat",
        )
        ax[iz].set_title(r"$z = $" + str(zz_unique[iz]), fontsize=fontsize)
    ax[iz].set_yscale("log")
    ax[iz].set_xscale("log")
    ax[iz].set_ylim(1e-3, 0.5)

    ax[0].legend(fontsize=fontsize - 4, loc="upper right", ncol=3)

    if type_measurement == "FFT":
        ax[-1].axis("off")

    fig.supxlabel(r"$k[\mathrm{km}^{-1}\mathrm{s}]$", fontsize=fontsize)
    fig.supylabel(r"$\sigma/P$", fontsize=fontsize)

    plt.tight_layout()

    if save_directory is not None:
        name = save_directory + "/cov_err_" + type_measurement + "_"
        plt.savefig(name + lab + ".pdf")
        plt.savefig(name + lab + ".png")
    else:
        plt.show()
