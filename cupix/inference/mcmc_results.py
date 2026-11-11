import os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import yaml
import copy
from getdist import MCSamples, plots
from lace.cosmo import cosmology
from cupix.inference.sampling_funcs import prepare_free_parameters, add_derived, apply_gaussian_weights_2d
from cupix.likelihood.config import Config
from cupix.inference.inference_config import InferenceConfig
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.px_data.data_DESI_DR2 import DESI_DR2



class MCMCResults:
    """
    Class to hold the MCMC results, with functions for plotting etc.
    Assumes directory is in format
    directory/
        chain_z*.h5
        setup_config_mcmc*_z*.yaml
        inference_config_mcmc*_z*.yaml
        mcmc_settings.yaml (optional)
    """
    def __init__(self, directory, iz):
        self.directory = directory
        dirfiles = os.listdir(directory)
        self.iz = np.atleast_1d(iz)
        # initialize 
        
        self.setup_files = {}
        self.inference_files = {}
        self.chain_files = {}
        self.tau_files = {}
        self.setup_configs = {}
        self.inf_configs = {}
        self.mcmc_settings_files = {}
        self.mcmc_settings = {}
        self.chains = {}
        self.log_probs = {}
        self.theories = {}
        self.likelihoods = {}
        self.freepars = {}
        self.mcsamples = {}

        # find all the setup and inference file, assume 1 per "z{iz}" pattern
        for iz in self.iz:
            found_chain = False
            for file in dirfiles:
                if f"z{iz}" in file and "setup_config_mcmc" in file:
                    self.setup_files[str(iz)] = os.path.join(directory, file)
                elif f"z{iz}" in file and "inference_config_mcmc" in file:
                    self.inference_files[str(iz)] = os.path.join(directory, file)
                elif file == f"chain_z{iz}.h5":
                    self.chain_files[str(iz)] = os.path.join(directory, file)
                    found_chain = True
                elif file == f"tau_estimates_z{iz}.txt":
                    self.tau_files[str(iz)] = os.path.join(directory, file)
                elif "mcmc_settings" in file and f"z{iz}" in file and file.endswith(".yaml"):
                    self.mcmc_settings_files[str(iz)] = os.path.join(directory, file)
            if not found_chain:
                # search for chain file with highest iteration, last one saved before crash / stop
                # pattern is chain_z{iz}_partial_it{iteration}.h5
                chain_files = [f for f in dirfiles if f.startswith(f"chain_z{iz}_partial")]
                if chain_files:
                    # sort by iteration number
                    chain_files.sort(key=lambda x: int(x.split("_it")[-1].split(".")[0]))
                    self.chain_files[str(iz)] = os.path.join(directory, chain_files[-1])
                    print(f"Found partial chain file for z{iz}: {chain_files[-1]}")
        
            self.setup_configs[str(iz)] = Config(self.setup_files[str(iz)])
        self.load_inf_config()
        self.load_theories_likelihoods()
        self.load_freepars()
        self.load_chain()

    def check_z(self, iz, z):
        
        if iz is not None and z is not None:
            raise ValueError("Must provide either iz or z, not both")
        if iz is None and z is None:
            raise ValueError("Must provide either iz or z")
        if iz is not None:
            if iz not in self.iz:
                raise ValueError(f"iz={iz} not found in self.iz={self.iz}")
            z = self.zs[str(iz)]
        else:
            if z not in self.zs.values():
                raise ValueError(f"z={z} not found in self.zs={self.zs}")
            iz = next(k for k, v in self.zs.items() if v == z)
        return iz, z
    
    def plot_tau_estimates(self, iz=None, z=None, save=False):
        iz, z = self.check_z(iz, z)
        tau_info = np.loadtxt(self.tau_files[str(iz)])
        plt.plot(tau_info[:,0], tau_info[:,1])
        F = self.inf_configs[str(iz)].samp_config.get("F_tau", 30)
        plt.plot(tau_info[:,0], tau_info[:,0]/F, label=r"$\tau = N/F$")
        # assumes we saved tau every 100 steps
        plt.ylabel(r"$\langle \tau \rangle$")
        plt.xlabel(r"Iteration")
        plt.legend()
        plt.title(f"Estimated autocorrelation time for z={z:.2f} (iz={iz})")
        save_fname = os.path.join(self.directory, f"tau_estimates_z{iz}.png")
        if save:
            plt.savefig(save_fname)
        plt.show()
        plt.clf()

    def prune_flatten_chain(self, iz, nburnin=300, thin=3):
        chain_new = self.chains[str(iz)][nburnin::thin]
        chain_new = chain_new.reshape(-1, chain_new.shape[-1]) #assumes chain is from get_chain
        return chain_new
    
    def build_samples(self, nburnin=300, thin=3):
        """MCSamples from chain_data, with loglikes if log_prob is available."""
        for iz in self.iz:
            # first check if samples already exists or if the nburnin / thinning requestsed have changed via shape
            nsteps, nwalkers, _ = self.chains[str(iz)].shape
            expected = len(range(nburnin, nsteps, thin)) * nwalkers

            if str(iz) not in self.mcsamples or self.mcsamples[str(iz)].samples.shape[0] != expected:
                print("Building MCSamples for z={:.2f} (iz={}) with nburnin={} and thin={}".format(self.zs[str(iz)], iz, nburnin, thin))
                chain = self.prune_flatten_chain(iz, nburnin, thin)
                kwargs = dict(names=[p.name for p in self.freepars[str(iz)]], labels=[p.latex_label for p in self.freepars[str(iz)]])
                log_prob = self.log_probs[str(iz)]
                if log_prob is not None:
                    log_prob = np.asarray(log_prob)[nburnin::thin].reshape(-1)
                    if log_prob.shape[0] != chain.shape[0]:
                        raise ValueError(
                            f"log_prob has {log_prob.shape[0]} entries but the pruned chain "
                            f"has {chain.shape[0]} samples; check burn-in/thinning/flattening")
                    kwargs["loglikes"] = -log_prob   # getdist expects -ln(posterior)
                self.mcsamples[str(iz)] = MCSamples(samples=chain, **kwargs)
    
    def load_chain(self):
        # load the chain
        for iz in self.iz:
            with h5.File(self.chain_files[str(iz)], "r") as f:
                gdnames = [g.decode() if isinstance(g, bytes) else g for g in f.attrs["gdnames"]]
                idx = [gdnames.index(p.name) for p in self.freepars[str(iz)]]
                self.chains[str(iz)] = f["chain"][:][..., idx]
                if "log_prob" in f.keys():
                    log_prob = f["log_prob"][:]
                else:
                    log_prob = None
                self.log_probs[str(iz)] = log_prob
            # make sure all gdnames are same as free_params names
            for par in self.freepars[str(iz)]:
                assert par.name in gdnames, f"Something is off! Parameter {par.name} not found in chain gdnames {gdnames} for z={self.zs[str(iz)]:.2f} (iz={iz})"
        
    def load_freepars(self):
        for iz in self.iz:
            free_param_names = list(self.inf_configs[str(iz)].params_config.keys())
            self.freepars[str(iz)] = prepare_free_parameters(
                free_param_names,
                self.theories[str(iz)],
                self.setup_configs[str(iz)].theory_config,
                params_config=self.inf_configs[str(iz)].params_config,
            )
            print(f"Free parameters for z={self.zs[str(iz)]}, iz={iz} were", [par.name for par in self.freepars[str(iz)]])
            
    def load_inf_config(self):
        for iz in self.iz:
            inf_config = InferenceConfig(self.inference_files[str(iz)])
            # open mcmc settings to see if anything needs replacing
            with open(self.mcmc_settings_files[str(iz)], "r") as f:
                self.mcmc_settings[str(iz)] = yaml.safe_load(f)
                
            for key, value in inf_config.params_config.items():
                # replace with key from mcmc_settings if it exists
                if key in self.mcmc_settings[str(iz)]:
                    inf_config.params_config[key] = self.mcmc_settings[str(iz)][key]
            self.inf_configs[str(iz)] = inf_config
            
    
    def load_theories_likelihoods(self):
        self.zs = {}
        for iz in self.iz:
            data = DESI_DR2(self.setup_configs[str(iz)].data_config)
            z = data.z[iz]
            self.zs[str(iz)] = z
            cosmo = cosmology.Cosmology(cosmo_params_dict=self.setup_configs[str(iz)].cosmo_config)
            theory = Theory(z=z, fid_cosmo=cosmo, config=self.setup_configs[str(iz)].theory_config)
            self.theories[str(iz)] = theory
            self.likelihoods[str(iz)] = Likelihood(data=data, theory=theory, iz=iz, config=self.setup_configs[str(iz)].like_config)

    def plot_chains(self, param_name,  iz=0, save=False, show=True, outdir=None):
        assert param_name in [self.freepars[str(iz)][i].name for i in range(len(self.freepars[str(iz)]))], f"Parameter {param_name} not found in free parameters for z={self.zs[str(iz)]:.2f} (iz={iz})"
        chain = self.chains[str(iz)]
        par_idx = [i for i, par in enumerate(self.freepars[str(iz)]) if par.name == param_name][0]
        
        if chain.ndim ==3:
            for w in range(chain.shape[1]):
                plt.plot(chain[:, w, par_idx], alpha=0.5) # plot each walker
        else:
            # chain has already been flattened
            plt.plot(chain[:, par_idx], alpha=0.5)
        plt.ylabel(rf"${self.freepars[str(iz)][par_idx].latex_label}$")
        plt.xlabel("step")
        plt.title(f"Full chain for parameter {param_name}, z={self.zs[str(iz)]:.2f} (iz={iz})")
        if save:
            if outdir is None:
                raise ValueError("outdir must be provided if save is True")
            plt.savefig(os.path.join(outdir, f"chain_full_{param_name}.png"))
        if show:
            plt.show()
        plt.clf()

    def plot_prob(self, iz=0, save=False, show=True, outdir=None):
        chain = self.log_probs[str(iz)]
        for w in range(chain.shape[1]):
            plt.semilogx(chain[:, w], alpha=0.5) # plot each walker
        plt.ylabel("Log probability")
        plt.xlabel("step")
        if save:
            if outdir is None:
                raise ValueError("outdir must be provided if save is True")
            plt.savefig(os.path.join(outdir, "chain_full_logprob.png"))
        if show:
            plt.show()
        plt.clf()

    def plot_contours(self, iz=0, title=None,
                    save=False, show_truth=False, show=True, outdir=None,
                    show_priors=False, show_bounds=False, show_best_fit=True,
                    params_to_plot=None,
                    remove_mean=False, gaussian_dict = {},
                    nburnin=300, thin=3):
        """Triangle plot of a chain.
    
        params_to_plot may contain free and/or derived parameter names; the panel
        order follows params_to_plot. If None, all free parameters plus all
        derived_pars are plotted. Priors, bounds and truth values are only drawn
        for free parameters (derived ones have none).
        Args:
        gaussian_dict: dictionary of parameter names to Gaussian parameters, 
                e.g. {"param_name": {"mean": 0.5, "std": 0.1, "corr": 0.5}}.
                These should be filled if you want to do importance sampling using external results.
        """
        
        if save and outdir is None:
            raise ValueError("outdir must be provided if save is True")
        self.build_samples(nburnin=nburnin, thin=thin)
        gdsamples = self.mcsamples[str(iz)]
    
        par_by_name = {par.name: par for par in self.freepars[str(iz)]}
        if params_to_plot is None:
            params_to_plot = [par.name for par in self.freepars[str(iz)]]
        params_to_plot = list(params_to_plot)
    
        # only compute the derived parameters that are actually plotted
        derived_needed = [d for d in params_to_plot if d not in par_by_name]
        if derived_needed:
            gdsamples = add_derived(gdsamples, self.freepars[str(iz)], derived_needed, self.theories[str(iz)])
            # set this as new samples object in case needed again
            self.mcsamples[str(iz)] = gdsamples
        
        # from here on do not modify the original samples object, so make a copy
        gdsamples = copy.deepcopy(self.mcsamples[str(iz)])
        if gaussian_dict:
            apply_gaussian_weights_2d(gdsamples, gaussian_dict)

        if remove_mean:
            gdsamples.setSamples(gdsamples.samples - np.average(gdsamples.samples, axis=0, weights=gdsamples.weights),
                                weights=gdsamples.weights, loglikes=gdsamples.loglikes)

        # None for derived parameters; everything below is indexed by plotted order
        plot_pars = [par_by_name.get(n) for n in params_to_plot]
        g = plots.get_subplot_plotter()
        g.triangle_plot([gdsamples], params=params_to_plot, filled=True)
    
        # ---- truth values -----------------------------------------------------
        if show_truth:
            truths = {p.name: p.true_value for p in plot_pars
                    if p is not None and p.true_value is not None}
            if truths:
                g.add_param_markers(truths)
    
        # ---- priors and bounds ------------------------------------------------
        for i, par_y in enumerate(plot_pars):
            # diagonal (1D posterior)
            ax = g.subplots[i, i]
            if show_priors:
                band = _prior_band(par_y)
                if band is not None:
                    ax.axvspan(*band, color="C1", alpha=0.3, zorder=0)
            if show_bounds and par_y is not None:
                for val in (par_y.min_value, par_y.max_value):
                    if val is not None:
                        ax.axvline(val, color="blue", ls="--", lw=1)
    
            # lower triangle (2D posteriors): x = params[j], y = params[i]
            for j in range(i):
                par_x = plot_pars[j]
                ax = g.subplots[i, j]
                if show_priors:
                    bx = _prior_band(par_x)
                    if bx is not None:
                        ax.axvspan(*bx, color="C1", alpha=0.15, zorder=0)
                    by = _prior_band(par_y)
                    if by is not None:
                        ax.axhspan(*by, color="C1", alpha=0.15, zorder=0)
                if show_bounds:
                    if par_x is not None:
                        for val in (par_x.min_value, par_x.max_value):
                            if val is not None:
                                ax.axvline(val, color="blue", ls="--", lw=1)
                    if par_y is not None:
                        for val in (par_y.min_value, par_y.max_value):
                            if val is not None:
                                ax.axhline(val, color="blue", ls="--", lw=1)
    
        # ---- best fit (computed once) ----------------------------------------
        if show_best_fit:
            bestfit_dict = self.chain_bestfit_dict(iz=iz, nburnin=nburnin,
                thin=thin)
            for i, name in enumerate(params_to_plot):
                if name in bestfit_dict:
                    ax = g.subplots[i, i]
                    val = bestfit_dict[name]
                    ax.axvline(val, color="red", ls="-", lw=1)
                    ax.text(0.95, 0.95, f"Best fit: {val:.3f}",
                            transform=ax.transAxes, fontsize=8,
                            verticalalignment="top", horizontalalignment="right",
                            color="red")
    
        g.finish_plot()  # applies tight_layout, so add the title afterwards
        if title is not None:
            g.fig.suptitle(title, y=1.02)
    
        if save:
            g.fig.savefig(os.path.join(outdir, "contours.png"), bbox_inches="tight")
        if show:
            plt.show()
        plt.close(g.fig)  # release it from pyplot; the returned figure is still intact
    


    def chain_bestfit_dict(self, iz=0, nburnin=300, thin=3):
        """Best-fit values (and marginalised stats) keyed by parameter name.

        If `samples` is given it must be an MCSamples built with build_samples
        (so it has loglikes) that already contains the derived parameters and has
        NOT been shifted (e.g. by remove_mean). Otherwise it is built here.
        """
        bestfit_dict = {}
        self.build_samples(nburnin=nburnin, thin=thin)
    
        samples = self.mcsamples[str(iz)]
        names = samples.getParamNames().list()
    
        if samples.loglikes is None:
            print("Warning: no log probabilities passed. Will use median of chain as bestfit.")
            median = np.median(samples.samples, axis=0)
            for n, v in zip(names, median):
                bestfit_dict[n] = v
        else:
            best = samples.getParamBestFitDict(best_sample=True, want_derived=True)
            margestats = samples.getMargeStats()
            for n in names:
                par = margestats.parWithName(n)
                bestfit_dict[n + "_mean"] = par.mean
                bestfit_dict[n + "_err"] = par.err
                bestfit_dict[n + "_err_low"] = par.limits[0].lower   # 68% limits
                bestfit_dict[n + "_err_up"] = par.limits[0].upper
                bestfit_dict[n] = best[n]

        return bestfit_dict





    def latex_table(self, nburnin=300, thin=3, derived_pars=[]):
        """Get the bestfit values from the chain, and return a dictionary with parameter names as keys and bestfit values as values."""
        # build samples if not already done
        self.build_samples(nburnin=nburnin, thin=thin)
        for iz in self.iz:
            p = self.mcsamples[str(iz)].getParams()
            # add derived parameters if not already stored
            derived_needed = [d for d in derived_pars if d not in p.names]
            if derived_needed:
                self.mcsamples[str(iz)] = add_derived(self.mcsamples[str(iz)], self.freepars[str(iz)], derived_needed, self.theories[str(iz)])
            # table = samples.getTable(include_bestfit=True)
            for param in self.mcsamples[str(iz)].getParamNames().list():
                print(self.mcsamples[str(iz)].getInlineLatex(param))
            
        return
    

    def plot_best_fit(self, iz, nburnin=300, thin=3, save=False, outdir=None,
                      include_chi2=False, include_probability=False,
                      multiply_by_k=True, xlim=None,
                      ylim=None, theorylabel="Best-fit theory", datalabel="Measurements",
                      title=None, ylim2=None, every_other_theta=False):
        """Plot the best fit theory and data."""
        bestfit_dict = self.chain_bestfit_dict(iz=iz, nburnin=nburnin, thin=thin)
        likelihood = self.likelihoods[str(iz)]
        bf_params = {p.name: bestfit_dict[p.name] for p in self.freepars[str(iz)]}
        likelihood.plot_px(params=bf_params, include_chi2=include_chi2, every_other_theta=every_other_theta, include_probability=include_probability, multiply_by_k=multiply_by_k, xlim=xlim, ylim=ylim, ylim2=ylim2, theorylabel=theorylabel, datalabel=datalabel, title=title,
                           n_free_p=len(self.freepars[str(iz)]))
        if save:
            if outdir is None:
                outdir=self.directory
            plt.savefig(os.path.join(outdir, f"best_fit_z{iz}.png"))




 
def _prior_band(par):
    """(lo, hi) of the Gaussian prior band, or None."""
    if par is None:
        return None
    if par.gauss_prior_mean is None or par.gauss_prior_width is None:
        return None
    return (par.gauss_prior_mean - par.gauss_prior_width,
            par.gauss_prior_mean + par.gauss_prior_width)
 
 




