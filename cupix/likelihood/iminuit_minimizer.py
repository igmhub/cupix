import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import copy


# our own modules
from cupix.likelihood import likelihood


class IminuitMinimizer(object):
    """Wrapper around an iminuit minimizer for Lyman alpha likelihood"""

    def __init__(self, like, error=0.02, verbose=False):
        """Setup minimizer from likelihood."""

        self.verbose = verbose
        self.like = like

        ini_values = np.full(len(self.like.free_param_names), 0.5)
        for i, parname in enumerate(self.like.free_param_names):
            par = self.like.like_parameter_by_name(parname)
            if par.ini_value is not None:
                ini_values[i] =  par.get_value_in_cube(par.ini_value)
        if self.verbose:
            print("Setting ini_values (in cube) to:", ini_values)

        # setup iminuit object (errordef=0.5 if using log-likelihood)

        self.minimizer = Minuit(like.minus_log_prob, ini_values, name=self.like.free_param_names)
        for i,parname in enumerate(self.like.free_param_names):
            self.minimizer.limits[parname] = (0.0, 1.0)
        if self.verbose:
            print("Set the iMinuit params to:\n", self.minimizer.params)
        # self.minimizer = Minuit(like.get_chi2, ini_values)
        self.minimizer.errordef = 0.5
        # error only used to set initial parameter step
        self.minimizer.errors = error

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

        return

    def plot_best_fit(self, multiply_by_k=True, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None, ylim=None, xlim=None):
        """Plot best-fit P1D vs data."""

        # get best-fit values from minimizer (should check that it was run)
        best_fit_values = np.array(self.minimizer.values)
        if self.verbose:
            print("best-fit values =", best_fit_values)

        like_params_to_plot = copy.deepcopy(self.like.like_params)
        for i, lp in enumerate(like_params_to_plot):
            if lp.name in self.like.free_param_names:
                index = self.like.free_param_names.index(lp.name)
                lp.value = lp.value_from_cube(best_fit_values[index])

        # plt.title("iminuit best fit")
        # plot_px(self, z, like_params, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None):    

        self.like.plot_px(
            z = self.like.iz_choice,
            like_params=like_params_to_plot,
            every_other_theta=every_other_theta,
            multiply_by_k=multiply_by_k,
            xlim=xlim,
            ylim=ylim,
            show=show,
            theorylabel=theorylabel,
            datalabel=datalabel,
            plot_fname=plot_fname
        )
        return


    def best_fit_value(self, pname, return_hesse=False):
        """Return best-fit value for pname parameter (assuming it was run).
        - return_hess: set to true to return also Gaussian error"""

        # get best-fit values from minimizer (in unit cube)
        cube_values = np.array(self.minimizer.values)
        if self.verbose:
            print("cube values =", cube_values)

        # get index for this parameter, and normalize value
        ipar = self.like.index_by_name(pname)
        par = self.like.like_parameter_by_name(pname)
        par_value = par.value_from_cube(cube_values[ipar])

        # check if you were asked for errors as well
        if return_hesse:
            cube_errors = self.minimizer.errors
            par_error = cube_errors[ipar] * (par.max_value - par.min_value)
            return par_value, par_error
        else:
            return par_value
        

    def plot_ellipses(self, pname_x, pname_y, nsig=2, cube_values=False, true_vals=None, true_val_label="true value"):
        """Plot Gaussian contours for parameters (pname_x,pname_y)
        - nsig: number of sigma contours to plot
        - cube_values: if True, will use unit cube values."""

        from matplotlib.patches import Ellipse
        from numpy import linalg as LA

        # figure out order of parameters in free parameters list
        ix = self.like.index_by_name(pname_x)
        iy = self.like.index_by_name(pname_y)

        # find out best-fit values, errors and covariance for parameters
        val_x = self.minimizer.values[ix]
        val_y = self.minimizer.values[iy]
        sig_x = self.minimizer.errors[ix]
        sig_y = self.minimizer.errors[iy]
        r = self.minimizer.covariance[ix, iy] / sig_x / sig_y

        # rescale from cube values (unless asked not to)
        if not cube_values:
            par_x = self.like.like_parameter_by_name(pname_x)
            val_x = par_x.value_from_cube(val_x)
            sig_x = sig_x * (par_x.max_value - par_x.min_value)
            par_y = self.like.like_parameter_by_name(pname_y)
            val_y = par_y.value_from_cube(val_y)
            sig_y = sig_y * (par_y.max_value - par_y.min_value)
            # multiply As by 10^9 for now, otherwise ellipse crashes
            if pname_x == "As":
                val_x *= 1e9
                sig_x *= 1e9
                pname_x += " x 1e9"
            if pname_y == "As":
                val_y *= 1e9
                sig_y *= 1e9
                pname_y += " x 1e9"

        # shape of ellipse from eigenvalue decomposition of covariance
        w, v = LA.eig(
            np.array(
                [
                    [sig_x**2, sig_x * sig_y * r],
                    [sig_x * sig_y * r, sig_y**2],
                ]
            )
        )

        # semi-major and semi-minor axis of ellipse
        a = np.sqrt(w[0])
        b = np.sqrt(w[1])

        # figure out inclination angle of ellipse
        alpha = np.arccos(v[0, 0])
        if v[1, 0] < 0:
            alpha = -alpha
        # compute angle in degrees (expected by matplotlib)
        alpha_deg = alpha * 180 / np.pi

        # make plot
        fig = plt.subplot(111)
        for isig in range(1, nsig + 1):
            ell = Ellipse(
                (val_x, val_y), 2 * isig * a, 2 * isig * b, angle=alpha_deg
            )
            ell.set_alpha(0.6 / isig)
            fig.add_artist(ell)
        if true_vals is not None:
            plt.plot(true_vals[pname_x], true_vals[pname_y], "rx", label=true_val_label, markersize=10)
            plt.legend()
        plt.xlabel(pname_x)
        plt.ylabel(pname_y)
        plt.xlim(val_x - (nsig + 1) * sig_x, val_x + (nsig + 1) * sig_x)
        plt.ylim(val_y - (nsig + 1) * sig_y, val_y + (nsig + 1) * sig_y)
        
