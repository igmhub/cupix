import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import os
import cupix
from cupix.inference.minimizer_funcs import (
    plot_corner as plot_corner_func,
    plot_ellipse as plot_ellipse_func,
    save_analysis_npz,
)


class Minimizer(object):
    """Wrapper around an iminuit minimizer for the Posterior class"""

    def __init__(self, post, config={}):
        """Setup minimizer from likelihood."""

        self.verbose = config.get("verbose", False)
        self.post = post

        # extract information from free_params
        free_params = self.post.free_params
        free_param_names = []
        ini_values = []
        param_errors = []
        Np = len(free_params)
        for ip in range(Np):
            free_param_names.append(free_params[ip].name)
            ini_values.append(free_params[ip].ini_value)
            param_errors.append(free_params[ip].delta)

        if self.verbose:
            print("Free parameters in minimizer")
            print(free_param_names)
            print("Inivial values set to", ini_values)

        # iminuit class itself
        self.minimizer = Minuit(
            self.minus_log_prob_interface, ini_values, name=free_param_names
        )

        # parameter limits
        for ip in range(Np):
            par = free_params[ip]
            self.minimizer.limits[par.name] = (par.min_value, par.max_value)

        if self.verbose:
            print("Set the iMinuit params to:\n", self.minimizer.params)

        # set errordef=0.5 if using log-likelihood
        self.minimizer.errordef = 0.5

        # error only used to set initial parameter step
        self.minimizer.errors = param_errors

        self.results_dict = None

    def silence(self):
        """set verbose=False in all classes"""
        self.verbose = False
        self.post.verbose = False
        self.post.like.verbose = False
        self.post.like.theory.verbose = False
        return

    def minus_log_prob_interface(self, values):

        # ask posterior to evalute log-prob
        log_prob = self.post.get_log_posterior_from_values(values=values)

        minus_log_prob = -1.0 * log_prob

        return minus_log_prob

    def minimize(self, compute_hesse=True):
        """Run migrad optimizer, and optionally compute Hessian matrix"""

        if self.verbose:
            print("will run migrad")
            self.minimizer.print_level = 0
        self.minimizer.migrad()

        if compute_hesse:
            if self.verbose:
                print("will compute Hessian matrix")
            self.minimizer.hesse()

    def _minimize_if_needed(self, compute_hesse=True):
        """Check that we have computed the best fit already"""

        if self.minimizer.valid == False:
            self.minimize(compute_hesse=compute_hesse)
        else:
            if self.verbose:
                print("already minimized")
            if self.minimizer.covariance is None and compute_hesse == True:
                if self.verbose:
                    print("compute Hessian")
                self.minimizer.hesse()

    def get_best_fit_params(self):
        """Run minimizer if needed, return dictionary"""

        # make sure you have run the minimizer
        self._minimize_if_needed(compute_hesse=False)

        # get best-fit values from minimizer (should check this is really the best-fit)
        best_fit_values = self.minimizer.values

        # transform to dictionary of parameters
        best_fit_params = self.post.get_params_from_values(best_fit_values)

        return best_fit_params

    def get_best_fit_chi2(self, return_info=False):
        """Compute chi2 for best-fit parameters (will minimize if needed)"""

        # get best-fit parameters (will run minimizer if needed)
        params = self.get_best_fit_params()

        # add fixed parameters in the posterior
        params.update(self.post.fixed_params)

        # ask likelihood for chi2
        return self.post.like.get_chi2(params=params, return_info=return_info)

    def get_best_fit_probability(self):

        # get best-fit parameters (will run minimizer if needed)
        params = self.get_best_fit_params()
        n_free_p = len(params)

        # add fixed parameters in the posterior
        params.update(self.post.fixed_params)

        return self.post.like.get_probability(params, n_free_p=n_free_p)

    def get_best_fit_value(self, pname, return_hesse=False):
        """Return best-fit value for pname parameter (assuming it was run).
        - return_hess: set to true to return also Gaussian error"""

        # make sure you have run the minimizer
        self._minimize_if_needed(compute_hesse=return_hesse)

        # get index for this parameter
        ipar = self.get_param_index(pname)
        if self.verbose:
            print("asked best-fit for param", ipar, pname)

        # get best-fit values from minimizer
        best_fit_values = np.array(self.minimizer.values)
        if self.verbose:
            print("best-fit values =", best_fit_values)

        # check if you were asked for errors as well
        if return_hesse:
            errors = self.minimizer.errors
            return best_fit_values[ipar], errors[ipar]
        else:
            return best_fit_values[ipar]

    def get_param_index(self, pname):
        """Find index of pname in self.post.free_params"""
        for ip, par in enumerate(self.post.free_params):
            if par.name == pname:
                ipar = ip
        return ipar

    def plot_ellipse(
        self,
        pname_x,
        pname_y,
        nsig=2,
        plot_truth=False,
        true_val_label="true value",
        xrange=None,
        yrange=None,
        ax=None,
        color="blue",
        label=None,
        title=None,
        outdir=None,
        outfile=None
    ):
        """Plot Gaussian contours for parameters (pname_x,pname_y)
        - nsig: number of sigma contours to plot."""
        if plot_truth:
            true_vals = {}
            for par in self.post.free_params:
                if par.name in [pname_x, pname_y]:
                    true_vals[par.name] = par.true_value
        else:
            true_vals = None
        ax = plot_ellipse_func(
            self.get_results_dict(),
            pname_x,
            pname_y,
            nsig=nsig,
            ax=None,
            color=color,
            label=label,
            true_vals=true_vals,
            true_val_label=true_val_label,
            xrange=xrange,
            yrange=yrange,
            title=title,
            outdir=outdir,
            outfile=outfile
        )

        return ax

    def plot_corner(
        self,
        nsig=2,
        fig=None,  # if you want to overplot on an existing figure
        axes=None,  # must not be None if fig is not None
        show_truth=True,
        true_val_label="true value",
        figsize=None,
        color="C0",
        label="",
        title=None,
        outdir=None,
        outfile=None
    ):
        """
        Gaussian corner plot from best-fit values and covariance.

        Parameters
        ----------
        nsig : int
            Number of sigma contours.
        true_vals : dict or None
            Dictionary of true parameter values keyed by parameter name.
        """
        fig, axes = plot_corner_func(
            self.get_results_dict(),
            self.post.free_params,
            nsig=nsig,
            fig=None,
            axes=None,
            show_truth=show_truth,
            true_val_label=true_val_label,
            figsize=figsize,
            color=color,
            label=label,
            title=title,
            outdir=outdir,
            outfile=outfile
        )

        return fig, axes

    def plot_best_fit(
        self,
        multiply_by_k=True,
        every_other_theta=False,
        show=True,
        theorylabel=None,
        datalabel=None,
        plot_fname=None,
        ylim=None,
        xlim=None,
        ylim2=None,
        title=None,
        residual_to_theory=False,
        extra_params=None,
        extra_label=None,
        include_chi2=False,
        include_probability=False,
        outdir=None,
        outfile=None,
    ):
        """Plot best-fit PX vs data."""

        # obtain dictionary of best-fit parameters (will minimize if needed)
        params = self.get_best_fit_params()

        # add fixed parameters in the posterior
        params.update(self.post.fixed_params)

        # use plotting tool in likelihood object to plot data and theory
        self.post.like.plot_px(
            params=params,
            every_other_theta=every_other_theta,
            multiply_by_k=multiply_by_k,
            xlim=xlim,
            ylim=ylim,
            show=show,
            theorylabel=theorylabel,
            datalabel=datalabel,
            plot_fname=plot_fname,
            ylim2=ylim2,
            title=title,
            residual_to_theory=residual_to_theory,
            extra_params=extra_params,
            extra_label=extra_label,
            include_probability=include_probability,
            include_chi2=include_chi2,
        )
        if outdir is not None:
            if outfile is None:
                plt.savefig(os.path.join(outdir, "best_fit_plot.png"))
            else:
                plt.savefig(os.path.join(outdir, outfile))
        return

    def print_results(self):
        """Print to screen summary of minimizer results"""
        results = self.get_results_dict()
        print(
            "chi2 = {:.3f} (ndf = {}) , prob = {:.3f}".format(
                results["chi2"], results["ndf"], results["prob"]
            )
        )
        for par in self.post.free_params:
            pname = par.name
            info = "{} = {:.4f} +/- {:.4f}".format(
                pname, results[pname], results[pname + "_err"]
            )
            if par.true_value is not None:
                info += " (true value = {:.4f})".format(par.true_value)
            if par.gauss_prior_mean is not None:
                mean = par.gauss_prior_mean
                rms = par.gauss_prior_width
                info += " (prior = {:.4f} +/- {:.4f})".format(mean, rms)
            if par.min_value is not None and par.max_value is not None:
                info += " (limits = [{:.4f}, {:.4f}])".format(
                    par.min_value, par.max_value
                )
            print(info)
        for key, val in self.post.fixed_params.items():
            print("{} = {:.4f} (fixed)".format(key, val))

    def get_results_dict(self):
        """Return dictionary with best-fit results, errors, and covariance matrix."""
        # check if it already exists
        if self.results_dict is None:
            results_dict = {}
            for par in self.post.free_params:
                parname = par.name
                bestfit, err = self.get_best_fit_value(parname, return_hesse=True)
                results_dict[parname] = bestfit
                results_dict[parname + "_err"] = err
            results_dict["param_names"] = [
                par.name for par in self.post.free_params
            ]  # to save the ordering
            covariance = self.minimizer.covariance
            results_dict["cov"] = covariance
            prob = self.get_best_fit_probability()
            results_dict["prob"] = prob
            chi2 = self.get_best_fit_chi2()
            results_dict["chi2"] = chi2
            ndata = self.post.like.get_ndata()
            results_dict["ndata"] = ndata
            ndf = self.post.get_ndf()
            results_dict["ndf"] = ndf
            self.results_dict = results_dict
            return results_dict
        else:
            return self.results_dict

    def save_results(self, outfile=None, outdir=None):
        if outdir is None:
            repo = os.path.dirname(cupix.__path__[0])
            outdir = os.path.join(repo, "data", "fitter_results")
        if outfile is None:
            outfile = f"iminuit_results.npz"
        savepath = os.path.join(outdir, outfile)
        print("Saving results to", savepath)
        save_analysis_npz(self.get_results_dict(), filename=savepath)
