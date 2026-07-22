import os
import h5py as h5
import numpy as np
from scipy.stats.distributions import chi2
from matplotlib.patches import Ellipse
from numpy import linalg as LA
import matplotlib.pyplot as plt

from cupix.likelihood.config import Config
from cupix.inference.inference_config import InferenceConfig
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.inference.sampling_funcs import prepare_free_parameters
from lace.cosmo import cosmology


def load_mini_results(results_directory, filename='iminuit_results.npz', setup_config_fname=None, inf_config_fname=None):
    """ Load the chain, and all setup configs, from directory to be able to re-create or continue working
    with analysis."""
    # Martine note: I might change this to a class later, since there are a lot of items to be returned
    # load the chain
    minires_fname = os.path.join(results_directory, filename)
    
    # Load results file
    outfile = np.load(minires_fname)
    # turn outfile into a dictionary
    results_dict = {key: outfile[key] for key in outfile.files}

    # recreate the theory, posterior, and likelihood objects
    if setup_config_fname is None:
        setup_config = Config(os.path.join(results_directory, 'setup_config_mini.yaml'))
    else:
        setup_config = Config(os.path.join(results_directory, setup_config_fname))
    if inf_config_fname is None:
        inf_config = InferenceConfig(os.path.join(results_directory, 'inference_config_mini.yaml'))
    else:
        inf_config = InferenceConfig(os.path.join(results_directory, inf_config_fname))
    data = DESI_DR2(setup_config.data_config)
    iz = setup_config.like_config['iz']
    z = data.z[iz]
    cosmo = cosmology.Cosmology(cosmo_params_dict=setup_config.cosmo_config)
    theory = Theory(z=z, fid_cosmo=cosmo, config=setup_config.theory_config)
    like = Likelihood(data=data, theory=theory, iz=iz, 
                config=setup_config.like_config)
    free_param_names = list(inf_config.params_config.keys())
    free_params = prepare_free_parameters(free_param_names, theory, setup_config.theory_config, params_config=inf_config.params_config)

    return results_dict, free_params, data, cosmo, theory, like, setup_config, inf_config


def plot_corner(results_dict,
            free_params,
            nsig=2,
            fig=None,
            axes=None,
            show_truth=True,
            true_val_label="true value",
            figsize=None,
            color="C0",
            label="",
            title=None,
            outdir=None,
            outfile=None):
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

    if axes is None:
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
                    label=label if (i,j)==(1,0) else None)

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

    if outdir is not None:
        if outfile is None:
            plt.savefig(os.path.join(outdir, "corner.png"))
        else:
            plt.savefig(os.path.join(outdir, outfile))
        
    return fig, axes

def plot_ellipse(
    results_dict,
    pname_x,
    pname_y,
    nsig=2,
    ax=None,
    color="C0",
    label=None,
    true_vals=None,
    true_val_label="true value",
    xrange=None,
    yrange=None,
    outdir=None,
    outfile=None,
    title=None,
    latex_label_x=None,
    latex_label_y=None
):
    """
    Plot covariance ellipse using a flat results_dict.

    Expects:
    - pname_x, pname_y: parameter names
    - results_dict must contain:
        * pname -> best-fit value
        * pname + "_err" -> 1σ error
        * "cov" -> full covariance matrix (aligned with parameter order in dict)
    """

    if ax is None:
        fig, ax = plt.subplots()

    # -------------------------
    # extract parameter list
    # -------------------------
    # assumes covariance matches order of keys excluding meta fields
    param_names = [
        k for k in results_dict.keys()
        if not k.endswith("_err")
        and k not in ["cov", "prob", "chi2", "ndata", "ndf"]
    ]

    ix = param_names.index(pname_x)
    iy = param_names.index(pname_y)

    val_x = results_dict[pname_x]
    val_y = results_dict[pname_y]

    sig_x = results_dict[pname_x + "_err"]
    sig_y = results_dict[pname_y + "_err"]

    cov = np.array(results_dict["cov"])

    # -------------------------
    # 2x2 covariance extraction
    # -------------------------
    C = cov[np.ix_([ix, iy], [ix, iy])]

    # eigen decomposition
    w, v = LA.eig(C)

    # sort eigenvalues (important fix)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]

    a = np.sqrt(w[0])
    b = np.sqrt(w[1])

    angle = np.arctan2(v[1, 0], v[0, 0])
    angle_deg = np.degrees(angle)

    # -------------------------
    # draw ellipses
    # -------------------------
    for isig in range(1, nsig + 1):
        ell = Ellipse(
            (val_x, val_y),
            width=2 * isig * a,
            height=2 * isig * b,
            angle=angle_deg,
            color=color,
            alpha=0.6 / isig,
        )
        ax.add_patch(ell)

    ax.plot(val_x, val_y, "o", color=color, label=label)

    # truth values
    if true_vals is not None:
        ax.axvline(true_vals[pname_x], color="grey", linestyle="--", label=true_val_label)
        ax.axhline(true_vals[pname_y], color="grey", linestyle="--")

    # -------------------------
    # axis limits
    # -------------------------
    if xrange is None:
        ax.set_xlim(val_x - (nsig + 1) * sig_x, val_x + (nsig + 1) * sig_x)
    else:
        ax.set_xlim(xrange)
    if yrange is None:
        ax.set_ylim(val_y - (nsig + 1) * sig_y, val_y + (nsig + 1) * sig_y)
    else:
        ax.set_ylim(yrange)

    if latex_label_x is not None:
        print(latex_label_x)
        ax.set_xlabel(rf"${latex_label_x}$")
    else:
        ax.set_xlabel(pname_x)
    if latex_label_y is not None:
        ax.set_ylabel(rf"${latex_label_y}$")
    else:
        ax.set_ylabel(pname_y)
    # if there are any labels, set legend
    if label is not None or (true_vals is not None and true_val_label is not None):
        ax.legend()
    if title is not None:
        ax.set_title(title)
    if outdir is not None:
        if outfile is None:
            plt.savefig(os.path.join(outdir, f"{pname_x}_{pname_y}.png"))
        else:
            plt.savefig(os.path.join(outdir, outfile))
        
    return ax

def save_analysis_npz(results, filename="analysis_results.npz"):
    """
    results: list or dict of per-analysis dictionaries
    """
    out = {}

    if isinstance(results, list):
        for i, r in enumerate(results):
            out[f"analysis-{i}"] = r
    else:  # dict
        for k, r in results.items():
            out[str(k)] = r

    # Save each dict as an object
    np.savez(filename, **out, allow_pickle=True)



def plot_best_fit(
        results_dict,
        free_params,
        like,
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
        out_fname=None,
    ):
        """Plot best-fit PX vs data."""

        # obtain dictionary of best-fit parameters (will minimize if needed)
        params = {}
        
        for par in free_params:
            params[par.name] = results_dict[par.name]
        
        # add fixed parameters in the posterior
        # params.update(self.post.fixed_params)

        # use plotting tool in likelihood object to plot data and theory
        like.plot_px(
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
            if out_fname is None:
                plt.savefig(os.path.join(outdir, "best_fit_plot.png"))
            else:
                plt.savefig(os.path.join(outdir, out_fname))
        return