import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import os
import cupix
from scipy.stats import chi2

class Minimizer(object):
    """Wrapper around an iminuit minimizer for the Posterior class"""

    def __init__(self, post, config={}):
        """Setup minimizer from likelihood."""

        self.verbose = config.get('verbose', False)
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
            print('Free parameters in minimizer')
            print(free_param_names)
            print('Inivial values set to', ini_values)

        # iminuit class itself
        self.minimizer = Minuit(self.minus_log_prob_interface, ini_values, name=free_param_names)

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


    def silence(self):
        """set verbose=False in all classes"""
        self.verbose=False
        self.post.verbose=False
        self.post.like.verbose=False
        self.post.like.theory.verbose=False
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
            if self.verbose: print('already minimized')
            if self.minimizer.covariance is None and compute_hesse == True:
                if self.verbose: print('compute Hessian')
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


    def plot_ellipses(self, pname_x, pname_y, nsig=2, 
                      true_vals=None, true_val_label="true value", 
                      xrange=None, yrange=None, ax=None, color='blue', extralabel=''):
        """Plot Gaussian contours for parameters (pname_x,pname_y)
        - nsig: number of sigma contours to plot. """

        from matplotlib.patches import Ellipse
        from numpy import linalg as LA

        # figure out order of parameters in free parameters list
        ix = self.get_param_index(pname_x)
        iy = self.get_param_index(pname_y)
        latexlabelx = self.post.free_params[ix].latex_label
        latexlabely = self.post.free_params[iy].latex_label
        # find out best-fit values, errors and covariance for parameters
        val_x = self.minimizer.values[ix]
        val_y = self.minimizer.values[iy]
        C = np.array([
            [self.minimizer.covariance[ix, ix],
            self.minimizer.covariance[ix, iy]],
            [self.minimizer.covariance[iy, ix],
            self.minimizer.covariance[iy, iy]]
        ])
        sig_x = self.minimizer.errors[ix]
        sig_y = self.minimizer.errors[iy]

        # shape of ellipse from eigenvalue decomposition of covariance
        w, v = LA.eig(
            np.array(C
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
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        for isig in range(1, nsig + 1):
            ell = Ellipse(
                (val_x, val_y), 2 * isig * a, 2 * isig * b, angle=alpha_deg, color=color
            )
            ell.set_alpha(0.6 / isig)
            # fig.add_artist(ell)
            ax.add_patch(ell)
        # plot a marker at the central value
        ax.plot(val_x, val_y, marker="o", color=color, label="best fit "+extralabel)
        if true_vals is not None:
            ax.axvline(true_vals[pname_x], color='grey', linestyle='--', label=true_val_label)
            ax.axhline(true_vals[pname_y], color='grey', linestyle='--')
            
        ax.set_xlabel(rf'${latexlabelx}$')
        ax.set_ylabel(rf'${latexlabely}$')
        if xrange==None or yrange==None:
            if true_vals is None:
                ax.set_xlim(val_x - (nsig + 1) * sig_x, val_x + (nsig + 1) * sig_x)
                ax.set_ylim(val_y - (nsig + 1) * sig_y, val_y + (nsig + 1) * sig_y)
            else:
                minx = min(val_x - (nsig + 1) * sig_x, true_vals[pname_x]-.1*abs(true_vals[pname_x]))
                maxx = max(val_x + (nsig + 1) * sig_x, true_vals[pname_x]+.1*abs(true_vals[pname_x]))
                miny = min(val_y - (nsig + 1) * sig_y, true_vals[pname_y]-.1*abs(true_vals[pname_y]))
                maxy = max(val_y + (nsig + 1) * sig_y, true_vals[pname_y]+.1*abs(true_vals[pname_y]))
                ax.set_ylim([miny,maxy])
                ax.set_xlim([minx,maxx])
        else:
            ax.set_ylim(yrange)
            ax.set_xlim(xrange)
            # show initial values (if asked for)
        
        ax.legend()
        return ax
    

    def plot_corner(self,
                nsig=2,
                show_truth=True,
                true_val_label="true value",
                figsize=None,
                color="C0",
                extralabel=""):
        """
        Gaussian corner plot from best-fit values and covariance.

        Parameters
        ----------
        nsig : int
            Number of sigma contours.
        true_vals : dict or None
            Dictionary of true parameter values keyed by parameter name.
        """

        from matplotlib.patches import Ellipse
        from numpy import linalg as LA

        npar = len(self.post.free_params)

        if figsize is None:
            figsize = (3*npar, 3*npar)

        fig, axes = plt.subplots(npar, npar, figsize=figsize)

        values = np.asarray(self.minimizer.values)
        errors = np.asarray(self.minimizer.errors)
        cov = np.asarray(self.minimizer.covariance)

        for i in range(npar):

            pname_i = self.post.free_params[i].name
            label_i = self.post.free_params[i].latex_label

            # -------------------------
            # Diagonal: 1D Gaussian
            # -------------------------
            ax = axes[i, i]

            x = np.linspace(values[i]-4*errors[i],
                            values[i]+4*errors[i], 300)

            y = np.exp(-(x-values[i])**2/(2*errors[i]**2))
            y /= y.max()

            ax.plot(x, y, color=color)

            if show_truth:
                ax.axvline(self.post.free_params[i].true_value,
                        ls="--",
                        color="grey",
                        label=true_val_label)

            ax.set_yticks([])
            ax.set_xlabel(rf"${label_i}$")

            # -------------------------
            # Lower triangle
            # -------------------------
            for j in range(i):

                ax = axes[i, j]

                C = cov[np.ix_([j, i], [j, i])]

                w, v = LA.eigh(C)

                # sort largest eigenvalue first
                order = np.argsort(w)[::-1]
                w = w[order]
                v = v[:, order]

                angle = np.degrees(np.arctan2(v[1,0], v[0,0]))

                a = np.sqrt(w[0])
                b = np.sqrt(w[1])
                # 1σ, 2σ, 3σ enclosed probabilities
                probs = [0.682689492, 0.954499736, 0.997300204]
                for p in probs[:nsig]:
                    scale = np.sqrt(chi2.ppf(p, df=2))
                

                    ell = Ellipse(
                        (values[j], values[i]),
                        width=2*a*scale,
                        height=2*b*scale,
                        angle=angle,
                        facecolor=color,
                        alpha=0.25/scale,
                        edgecolor=color
                    )

                    ax.add_patch(ell)

                ax.plot(values[j], values[i],
                        "o",
                        color=color,
                        label="best fit "+extralabel if (i,j)==(1,0) else None)

                if show_truth:
                    ax.axvline(self.post.free_params[j].true_value,
                            color="grey",
                            ls="--")
                    ax.axhline(self.post.free_params[i].true_value,
                            color="grey",
                            ls="--")

                sx = errors[j]
                sy = errors[i]

                xmin = values[j]-(nsig+1)*sx
                xmax = values[j]+(nsig+1)*sx
                ymin = values[i]-(nsig+1)*sy
                ymax = values[i]+(nsig+1)*sy

                if show_truth:
                    tx = self.post.free_params[j].true_value
                    ty = self.post.free_params[i].true_value

                    xmin = min(xmin, tx-0.1*abs(tx))
                    xmax = max(xmax, tx+0.1*abs(tx))
                    ymin = min(ymin, ty-0.1*abs(ty))
                    ymax = max(ymax, ty+0.1*abs(ty))

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                if i == npar-1:
                    ax.set_xlabel(rf"${self.post.free_params[j].latex_label}$")
                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(rf"${label_i}$")
                else:
                    ax.set_yticklabels([])

            # -------------------------
            # Upper triangle
            # -------------------------
            for j in range(i+1, npar):
                axes[i, j].axis("off")

        handles, labels = axes[1,0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

        fig.tight_layout()

        return fig, axes


    def plot_best_fit(self, multiply_by_k=True, every_other_theta=False, show=True, 
                      theorylabel=None, datalabel=None, plot_fname=None, 
                      ylim=None, xlim=None, ylim2=None, title=None, residual_to_theory=False,
                      extra_params=None, extra_label=None, include_chi2=False, include_probability=False,
                      outdir=None, out_fname=None
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
            include_chi2=include_chi2
        )
        if outdir is not None:
            if out_fname is None:
                plt.savefig(os.path.join(outdir, "best_fit_plot.png"))
            else:
                plt.savefig(os.path.join(outdir, out_fname))
        return


    def print_results(self):
        """Print to screen summary of minimizer results"""
        results = self.get_results_dict()
        print('chi2 = {:.3f} (ndf = {}) , prob = {:.3f}'.format(results['chi2'], results['ndf'], results['prob']))
        for par in self.post.free_params:
            pname = par.name
            info = '{} = {:.4f} +/- {:.4f}'.format(pname, results[pname], results[pname+'_err'])
            if par.true_value is not None:
                info += ' (true value = {:.4f})'.format(par.true_value)
            if par.gauss_prior_mean is not None:
                mean = par.gauss_prior_mean
                rms = par.gauss_prior_width
                info += ' (prior = {:.4f} +/- {:.4f})'.format(mean, rms)
            if par.min_value is not None and par.max_value is not None:
                info += ' (limits = [{:.4f}, {:.4f}])'.format(par.min_value, par.max_value)
            print(info)
        for key, val in self.post.fixed_params.items():
            print('{} = {:.4f} (fixed)'.format(key, val))


    def get_results_dict(self):
        """Return dictionary with best-fit results, errors, and covariance matrix."""
        results_dict = {}
        for par in self.post.free_params:
            parname = par.name
            bestfit, err = self.get_best_fit_value(parname, return_hesse=True)
            results_dict[parname] = bestfit
            results_dict[parname+'_err'] = err
        covariance = self.minimizer.covariance
        results_dict['cov'] = covariance
        prob = self.get_best_fit_probability()
        results_dict['prob'] = prob
        chi2 = self.get_best_fit_chi2()
        results_dict['chi2'] = chi2
        ndata = self.post.like.get_ndata()
        results_dict['ndata'] = ndata
        ndf = self.post.get_ndf()
        results_dict['ndf'] = ndf
        return results_dict


    def save_results(self, outfile=None, outdir=None):
        results_dict = self.get_results_dict()
        if outdir is None:
            repo = os.path.dirname(cupix.__path__[0])
            outdir = os.path.join(repo, "data", "fitter_results")
        if outfile is None:
            outfile = f"iminuit_results.npz"
        savepath = os.path.join(outdir, outfile)
        print("Saving results to", savepath)
        save_analysis_npz(results_dict, filename=savepath)

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


def plot_ellipses(val_x, val_y, pname_x, pname_y, sig_x, sig_y, cov, nsig=2, true_vals=None, true_val_label="true value", xrange=None, yrange=None):
    
    """Same as the class's plot_ellipses function, but for external use
     (e.g. for plotting saved results)
     Plot Gaussian contours for (val_x,val_y)
    - nsig: number of sigma contours to plot
    - cube_values: if True, will use unit cube values."""

    from matplotlib.patches import Ellipse
    from numpy import linalg as LA
    
    # shape of ellipse from eigenvalue decomposition of covariance
    w, v = LA.eig(
        np.array(cov
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
    plt.legend()



def plot_corner(results_dict,
            free_params,
            nsig=2,
            show_truth=True,
            true_val_label="true value",
            figsize=None,
            color="C0",
            extralabel=""):
    """
    Gaussian corner plot from best-fit values and covariance.

    Parameters
    ----------
    nsig : int
        Number of sigma contours.
    true_vals : dict or None
        Dictionary of true parameter values keyed by parameter name.
    """

    from matplotlib.patches import Ellipse
    from numpy import linalg as LA

    npar = len(free_params)

    if figsize is None:
        figsize = (3*npar, 3*npar)

    fig, axes = plt.subplots(npar, npar, figsize=figsize)

    values = np.asarray([results_dict[par.name] for par in free_params])
    errors = np.asarray([results_dict[par.name+'_err'] for par in free_params])
    cov = np.asarray(results_dict['cov'])

    for i in range(npar):

        pname_i = free_params[i].name
        label_i = free_params[i].latex_label

        # -------------------------
        # Diagonal: 1D Gaussian
        # -------------------------
        ax = axes[i, i]

        x = np.linspace(values[i]-4*errors[i],
                        values[i]+4*errors[i], 300)

        y = np.exp(-(x-values[i])**2/(2*errors[i]**2))
        y /= y.max()

        ax.plot(x, y, color=color)

        if show_truth:
            ax.axvline(free_params[i].true_value,
                    ls="--",
                    color="grey",
                    label=true_val_label)

        ax.set_yticks([])
        ax.set_xlabel(rf"${label_i}$")

        # -------------------------
        # Lower triangle
        # -------------------------
        for j in range(i):

            ax = axes[i, j]

            C = cov[np.ix_([j, i], [j, i])]

            w, v = LA.eigh(C)

            # sort largest eigenvalue first
            order = np.argsort(w)[::-1]
            w = w[order]
            v = v[:, order]

            angle = np.degrees(np.arctan2(v[1,0], v[0,0]))

            a = np.sqrt(w[0])
            b = np.sqrt(w[1])
            # 1σ, 2σ, 3σ enclosed probabilities
            probs = [0.682689492, 0.954499736, 0.997300204]
            for p in probs[:nsig]:
                scale = np.sqrt(chi2.ppf(p, df=2))
            

                ell = Ellipse(
                    (values[j], values[i]),
                    width=2*a*scale,
                    height=2*b*scale,
                    angle=angle,
                    facecolor=color,
                    alpha=0.25/scale,
                    edgecolor=color
                )

                ax.add_patch(ell)

            ax.plot(values[j], values[i],
                    "o",
                    color=color,
                    label="best fit "+extralabel if (i,j)==(1,0) else None)

            if show_truth:
                ax.axvline(free_params[j].true_value,
                        color="grey",
                        ls="--")
                ax.axhline(free_params[i].true_value,
                        color="grey",
                        ls="--")

            sx = errors[j]
            sy = errors[i]

            xmin = values[j]-(nsig+1)*sx
            xmax = values[j]+(nsig+1)*sx
            ymin = values[i]-(nsig+1)*sy
            ymax = values[i]+(nsig+1)*sy

            if show_truth:
                tx = free_params[j].true_value
                ty = free_params[i].true_value

                xmin = min(xmin, tx-0.1*abs(tx))
                xmax = max(xmax, tx+0.1*abs(tx))
                ymin = min(ymin, ty-0.1*abs(ty))
                ymax = max(ymax, ty+0.1*abs(ty))

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            if i == npar-1:
                ax.set_xlabel(rf"${free_params[j].latex_label}$")
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(rf"${label_i}$")
            else:
                ax.set_yticklabels([])

        # -------------------------
        # Upper triangle
        # -------------------------
        for j in range(i+1, npar):
            axes[i, j].axis("off")

    handles, labels = axes[1,0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.tight_layout()

    return fig, axes
