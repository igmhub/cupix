import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import copy
from cupix.likelihood.likelihood_parameter import par_index, LikelihoodParameter, like_parameter_by_name
from cupix.likelihood import likelihood


class IminuitMinimizer(object):
    """Wrapper around an iminuit minimizer for Lyman alpha likelihood"""

    def __init__(self, like, like_params, free_param_names, error=0.02, verbose=False):
        """Setup minimizer from likelihood."""

        self.verbose = verbose
        self.like = like
        self.free_param_names = free_param_names
        self.like_params = like_params
        self.out_like_params = None
        free_params = []
        for par in self.free_param_names:
            for lpar in self.like_params:
                if lpar.name == par:
                    free_params.append(lpar)
        assert len(self.free_param_names)==len(free_params), "Couldn't find all desired free parameters in like_params"
        print("Free params are", free_params)
        ini_values = np.full(len(self.free_param_names), 0.5)
        for i,par in enumerate(free_params):
            if par.ini_value is not None:
                ini_values[i] =  par.get_value_in_cube(par.ini_value)
        
        if self.verbose:
            print("Setting ini_values (in cube) to:", ini_values)
        # set priors if needed
        self.like.set_Gauss_priors(free_params)
        
        
        # setup iminuit object (errordef=0.5 if using log-likelihood)

        self.minimizer = Minuit(self.minus_log_prob_interface, ini_values, name=self.free_param_names)
        for i,parname in enumerate(self.free_param_names):
            self.minimizer.limits[parname] = (0.0, 1.0)
        if self.verbose:
            print("Set the iMinuit params to:\n", self.minimizer.params)
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

    def plot_best_fit(self, multiply_by_k=True, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None, ylim=None, xlim=None, ylim2=None, title=None, residual_to_theory=False):
        """Plot best-fit P1D vs data."""

        # get best-fit values from minimizer (should check that it was run)
        best_fit_values = np.array(self.minimizer.values)

        like_params_to_plot = copy.deepcopy(self.like_params)
        for i, lp in enumerate(like_params_to_plot):
            if lp.name in self.free_param_names:
                
                
                index = self.free_param_names.index(lp.name)
                lp.value = lp.value_from_cube(best_fit_values[index])
                if self.verbose:
                    print("best-fit value for", lp.name, "is", lp.value)

        # plt.title("iminuit best fit")
        # plot_px(self, z, like_params, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None):    

        self.like.plot_px(
            like_params=like_params_to_plot,
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
            residual_to_theory=residual_to_theory
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
        ipar = self.free_param_names.index(pname)
        par = like_parameter_by_name(self.like_params,pname)
        par_value = par.value_from_cube(cube_values[ipar])

        # check if you were asked for errors as well
        if return_hesse:
            cube_errors = self.minimizer.errors
            par_error = cube_errors[ipar] * (par.max_value - par.min_value)
            return par_value, par_error
        else:
            return par_value

    def set_bestfit_like_params(self):
        print("Assuming minimizer already run.")
        cube_values = np.array(self.minimizer.values)
        like_params_temp = [] # list of LikelihoodParameters objects that will get passed through the likelihood functions
        for i,val in enumerate(cube_values):
            parname = self.free_param_names[i]
            par_i = par_index(self.like_params, parname)
            like_param = self.like_params[par_i]
            # copy over all but update the current values
            like_params_temp.append(
                LikelihoodParameter(
                name=like_param.name,
                min_value=like_param.min_value,
                max_value=like_param.max_value,
                ini_value=like_param.ini_value,
                value=like_param.value_from_cube(val)
                ))
        self.out_like_params = like_params_temp

    def fit_probability(self):
        if self.out_like_params is None:
            self.set_bestfit_like_params()
        return self.like.fit_probability(self.out_like_params, n_free_p=len(self.free_param_names))

    def chi2(self):
        if self.out_like_params is None:
            self.set_bestfit_like_params()
        return self.like.get_chi2(self.out_like_params)

    def plot_ellipses(self, pname_x, pname_y, nsig=2, cube_values=False, true_vals=None, true_val_label="true value", xrange=None, yrange=None):
        """Plot Gaussian contours for parameters (pname_x,pname_y)
        - nsig: number of sigma contours to plot
        - cube_values: if True, will use unit cube values."""

        from matplotlib.patches import Ellipse
        from numpy import linalg as LA

        # figure out order of parameters in free parameters list
        ix = self.free_param_names.index(pname_x)
        iy = self.free_param_names.index(pname_y)

        # find out best-fit values, errors and covariance for parameters
        val_x = self.minimizer.values[ix]
        val_y = self.minimizer.values[iy]
        sig_x = self.minimizer.errors[ix]
        sig_y = self.minimizer.errors[iy]
        r = self.minimizer.covariance[ix, iy] / sig_x / sig_y

        # rescale from cube values (unless asked not to)
        if not cube_values:
            par_x = like_parameter_by_name(self.like_params, pname_x)
            val_x = par_x.value_from_cube(val_x)
            sig_x = sig_x * (par_x.max_value - par_x.min_value)
            par_y = like_parameter_by_name(self.like_params, pname_y)
            val_y = par_y.value_from_cube(val_y)
            sig_y = sig_y * (par_y.max_value - par_y.min_value)

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
        print("a, b", a, b)
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
        # plot a marker at the central value
        plt.plot(val_x, val_y, "ro", label="best fit")
        if true_vals is not None:
            plt.axvline(true_vals[pname_x], color='grey', linestyle='--', label=true_val_label)
            plt.axhline(true_vals[pname_y], color='grey', linestyle='--')
            
        plt.xlabel(pname_x)
        plt.ylabel(pname_y)
        if xrange==None or yrange==None:
            if true_vals is None:
                plt.xlim(val_x - (nsig + 1) * sig_x, val_x + (nsig + 1) * sig_x)
                plt.ylim(val_y - (nsig + 1) * sig_y, val_y + (nsig + 1) * sig_y)
            else:
                minx = min(val_x - (nsig + 1) * sig_x, true_vals[pname_x]-.1*abs(true_vals[pname_x]))
                maxx = max(val_x + (nsig + 1) * sig_x, true_vals[pname_x]+.1*abs(true_vals[pname_x]))
                miny = min(val_y - (nsig + 1) * sig_y, true_vals[pname_y]-.1*abs(true_vals[pname_y]))
                maxy = max(val_y + (nsig + 1) * sig_y, true_vals[pname_y]+.1*abs(true_vals[pname_y]))
                plt.ylim([miny,maxy])
                plt.xlim([minx,maxx])
        else:
            plt.ylim(yrange)
            plt.xlim(xrange)
        plt.axvline(like_parameter_by_name(self.like_params, pname_x).ini_value, color='orange', linestyle='dotted', label='ini value')
        plt.axhline(like_parameter_by_name(self.like_params, pname_y).ini_value, color='orange', linestyle='dotted')
        plt.legend()

    def results_dict_2par(self):
        """Return dictionary with best-fit results and covariance matrix."""
        results_dict = {}
        for parname in self.like.free_param_names:
            bestfit, err = self.best_fit_value(parname, return_hesse=True)
            results_dict[parname] = bestfit
            results_dict[parname+'_err'] = err
        covariance = self.minimizer.covariance
        results_dict['cov'] = covariance

        ## save the following for the sake of plotting ellipses:
        ix = self.like.index_by_name(self.like.free_param_names[0])
        iy = self.like.index_by_name(self.like.free_param_names[1])

        # find out best-fit values, errors and covariance for parameters
        sig_x = self.minimizer.errors[ix]
        sig_y = self.minimizer.errors[iy]
        r = self.minimizer.covariance[ix, iy] / sig_x / sig_y
        results_dict['r']=r
        results_dict['par_x']=parname[0]
        results_dict['par_y']=parname[1]
        prob = self.fit_probability()
        results_dict['prob'] = prob
        chi2 = self.get_chi2()
        results_dict['chi2'] = chi2
        return results_dict

    def minus_log_prob_interface(self, values):
        # a log like function that ingests the likelihood log_like function
        # but instead of taking parameter dictionaries or LikelihoodParamter objects,
        # takes a list of parameter values in the unit cube, and uses like.free_params information to transform them to physical values before passing to the likelihood log_like function
        # this will be the function that is passed to the minimizer, and will be called by the minimizer with different parameter values in the unit cube
        
        assert len(values) == len(self.free_param_names), "Length of iterated values must equal free parameter name list lenghth."
        
        like_params_temp = [] # list of LikelihoodParameters objects that will get passed through the likelihood functions
        for par in self.like_params:
            if par.name in self.free_param_names:
                freepar_index = self.free_param_names.index(par.name)
                # copy over all but update the current values
                like_params_temp.append(
                    LikelihoodParameter(
                    name=par.name,
                    min_value=par.min_value,
                    max_value=par.max_value,
                    ini_value=par.ini_value,
                    value=par.value_from_cube(values[freepar_index])
                    ))
            else:
                # copy over parameter
                like_params_temp.append(par)

        return(self.like.minus_log_prob(like_params_temp, self.free_param_names))

def save_analysis_npz(results, filename="analysis_results.npz"):
    """
    results: list or dict of per-analysis dictionaries
    """
    out = {}

    if isinstance(results, list):
        for i, r in enumerate(results):
            out[f'analysis-{i}'] = r
    else:  # dict
        for k, r in results.items():
            out[str(k)] = r

    # Save each dict as an object
    np.savez(filename, **out, allow_pickle=True)

