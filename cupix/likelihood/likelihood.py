import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import math
from scipy.stats.distributions import chi2 as chi2_scipy
from scipy.optimize import minimize
from scipy.linalg import block_diag
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta
from cupix.utils.utils import is_number_string
from cupix.likelihood.likelihood_parameter import LikelihoodParameter, likeparam_from_dict
import warnings
import time

class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

    def __init__(
        self,
        data,
        theory,
        z,
        free_params=[],
        verbose=False,
        prior_Gauss_rms=None,
        min_log_like=-1e100,
    ):
        """Setup likelihood from theory and data. Options:
        - data (required) is the data to model
        - theory (required) instance of lya_theory
        - z is a single redshift value
        - free_params is a list of LikelihoodParameter instances that we want to vary in the likelihood. If empty, it 
            is assumed no parameters will be varied and no priors are applied.
        - if prior_Gauss_rms is None it will use uniform priors
        - min_log_like: use this instead of - infinity"""

        self.verbose = verbose
        self.prior_Gauss_rms = prior_Gauss_rms
        self.min_log_like = min_log_like
        self.data = data
        # ensure only one z passed to likelihood, and ensure that it exists in both the data and theory zs
        if np.atleast_1d(z).size != 1:
            raise ValueError("Likelihood computation not implemented for more than one redshift at once. Please pass a single redshift value.")
        self.z = z
        self.theory = theory
        self.theory_iz = np.where(self.theory.zs == self.z)[0][0]
        self.data_iz = np.where(self.data.z == self.z)[0][0]
        if self.verbose:
            print("Likelihood will evaluate redshift of", self.z, "which corresponds to iz =", self.theory_iz, "in theory and iz =", self.data_iz, "in data.")
        
        # for priors, we will keep it optional:
        # if free_param_names is empty, no priors; if it's not empty, set priors on those parameters.
        # Either Gaussians with widths given by prior_Gauss_rms, or uniform if prior_Gauss_rms is None.
        if free_params:
            self.set_Gauss_priors()
            self.free_param_names = [par.name for par in free_params]



    def get_convolved_Px_AA(self,
        theta_A,
        like_params=None):

        """Compute theoretical prediction for Pcross
        Args:
        iz (int): redshift bin index
        theta_A (int or list of int): index (or list of indices) of coarse theta_A bins
        like_params (list of LikelihoodParameter): parameters to use for theory

        Returns:
        Px_ZAM (np.ndarray): array of shape (len(theta_A), len(k_AA_binned))
            with convolved Pcross for each coarse-theta_A bin
        """
        # if theta_A is a single int, make it a list of one element for easier handling
        if type(theta_A) is not list and type(theta_A) is not np.ndarray:
            theta_A = [theta_A]
        z = np.atleast_1d(self.z)
        k_AA_fine = self.data.k_m
        # print("k_AA_fine shape", k_AA_fine.shape)
        theta_a_arcmin = (self.data.theta_min_a_arcmin + self.data.theta_max_a_arcmin)/2.
        if self.verbose:
            print("Computing Px simultaneously for thetas,", theta_a_arcmin, "arcmin")
        Px_Zam = self.theory.get_px_AA(
            zs = z,
            k_AA=k_AA_fine[self.data_iz],
            theta_arcmin=theta_a_arcmin,
            like_params=like_params,
            verbose=self.verbose
        )
        
        Px_Zam = np.squeeze(Px_Zam) # reduce from shape (1, ntheta_a, nk_AA) to (ntheta_a, nk_AA)
        # print("Px_Zam shape", Px_Zam.shape)
        # window convolution
        # loop through the large-theta bins from the data
        Px_ZAM_all = []
        for itheta_A in theta_A:
            # get the small-theta indices in this coarse-theta bin
            ind_in_theta = self.data.B_A_a.astype(bool)[itheta_A,:] # generally this could be different with redshift; assume it's not for now
            theta_a_inds = np.where(ind_in_theta)[0]
            
            k_AA_binned = (self.data.k_M_edges[self.data_iz][:-1] + self.data.k_M_edges[self.data_iz][1:]) / 2.
            Px_ZaM_all = np.zeros((len(theta_a_inds), len(k_AA_binned)))
            V_ZaM_all  = np.zeros((len(theta_a_inds), len(k_AA_binned)))
            for save_index, a in enumerate(theta_a_inds):
                # retrieve the window matrix for this small-theta bin
                if self.data.U_ZaMn is not None:
                    U_ZaMn = self.data.U_ZaMn[self.data_iz,a]
                    # print("U_ZaMn shape", U_ZaMn.shape)
                    # print("Px_ZaM shape to be convolved", Px_Zam[a].T.shape)
                    Px_ZaM = convolve_window(U_ZaMn,Px_Zam[a].T)
                Px_ZaM_all[save_index,:] = Px_ZaM
                V_ZaM = self.data.V_ZaM[self.data_iz,a]
                V_ZaM_all[save_index,:] = V_ZaM
            # rebin in theta
            Px_ZAM = rebin_theta(V_ZaM_all, Px_ZaM_all)
            Px_ZAM_all.append(Px_ZAM)
        print(np.asarray(Px_ZAM_all).shape)
        return np.asarray(Px_ZAM_all)



    def get_chi2(self, like_params=None, return_all=False):
        """Compute chi2 using data and theory, without adding
        emulator covariance"""
        
        log_like, log_like_all = self.get_log_like(
            like_params=like_params,
            ignore_log_det_cov=True
        )
        
        # print(-2 * log_like, -2 * log_like_all, -2 * np.sum(log_like_all))

        if return_all:
            return -2.0 * log_like, -2.0 * log_like_all
        else:
            return -2.0 * log_like

    def get_log_like(
        self,
        like_params=None,
        ignore_log_det_cov=True
    ):
        """Compute log(likelihood), including determinant of covariance
        unless you are setting ignore_log_det_cov=True."""
        start = time.time()
        # what to return if we are out of priors
        null_out = [-np.inf, -np.inf]
        data = self.data
        ntheta = len(data.theta_max_A_arcmin)
        # compute log like contribution from this redshift bin
        log_like_all = np.zeros(ntheta)
        log_like = 0
        
        
        theta_A = np.arange(ntheta)
        model_iz = self.get_convolved_Px_AA(theta_A,like_params)
        for itheta in range(ntheta):
            # compute chi2 for this redshift bin
            det_cov = np.linalg.det(self.data.cov_ZAM[self.data_iz,itheta,:,:])
            if det_cov == 0:
                # make a warning
                warnings.warn("Det(cov) appears to be 0: could be a singular covariance matrix!")
            icov_ZAM = np.linalg.inv(self.data.cov_ZAM[self.data_iz,itheta,:,:])
            data_izitheta  = data.Px_ZAM[self.data_iz,itheta,:]
            diff = np.squeeze(data_izitheta - model_iz[itheta])
            chi2_z = np.dot(np.dot(np.squeeze(icov_ZAM), diff), diff)
            # check whether to add determinant of covariance as well
            if ignore_log_det_cov:
                log_like_all[itheta] = -0.5 * chi2_z
            else:
                det_icov = np.linalg.det(icov_ZAM)
                assert np.isfinite(det_icov), "Non-finite determinant of inverse covariance!"
                log_det_cov = np.log(
                    np.abs(1 / np.linalg.det(icov_ZAM))
                )
                log_like_all[itheta] = -0.5 * (chi2_z + log_det_cov)
        log_like += np.sum(log_like_all)

        end = time.time()
        print(f"Log-likelihood computed in {end - start:.2f} seconds")
        if np.any(np.isnan(log_like)):
            return null_out
        
        out = [log_like, log_like_all]
        
        return out

    def regulate_log_like(self, log_like):
        """Make sure that log_like is not NaN, nor tiny"""
        if (log_like is None) or math.isnan(log_like):
            print("Returning min_log_like due to NaN or None")
            return self.min_log_like
        return max(self.min_log_like, log_like)

    def log_prob(
        self, like_params, ignore_log_det_cov=True
    ):
        """Compute log likelihood plus log priors for input values"""

        # translate like_params into array of values in cube
        for par in like_params:
            value = par.value_in_cube()
            # Always force parameter to be within range
            if (value > 1.0) or (value < 0.0):
                return self.min_log_like

        # compute log_prior
        if self.Gauss_priors is not None:
            log_prior = self.get_log_prior(values)
        else:
            log_prior = 0
        # compute log_like
        log_like, chi2_all = self.get_log_like(
            like_params=like_params,
            ignore_log_det_cov=ignore_log_det_cov
        )

        # regulate log-like (not NaN, not tiny)
        log_like = self.regulate_log_like(log_like)


        return log_like + log_prior




    def get_log_prior(self, values):
        """Compute logarithm of prior"""

        assert len(values) == len(self.free_param_names), "size mismatch"

        # Always force parameter to be within range (for now)
        if max(values) > 1:
            return self.min_log_like
        if min(values) < 0:
            return self.min_log_like
    
        # get the initial values
        fid_values = []
        for p in self.like_params:
            if p.name in self.free_param_names:
                fid_values.append(p.get_value_in_cube(p.ini_value))
        log_prior = -np.sum(
            (np.array(fid_values) - values) ** 2 / (2 * self.Gauss_priors**2)
        )
        if self.verbose:
            print("log prior is", log_prior)
        return log_prior

    def minus_log_prob(self, values):
        """Return minus log_prob (needed to maximise posterior)"""
        if self.verbose:
            print("values at beginning are", values)
        log_prob = self.log_prob(values)
        if not np.isfinite(log_prob):
            print("Non-finite value detected:", values, log_prob)
        print("log prob is", log_prob)
        return -1.0 * log_prob

    def maximise_posterior(
        self, initial_values=None, method="nelder-mead", tol=1e-4
    ):
        """Run scipy minimizer to find maximum of posterior"""

        if not initial_values:
            initial_values = np.ones(len(self.free_param_names)) * 0.5

        return minimize(
            self.minus_log_prob, x0=initial_values, method=method, tol=tol
        )


    def plot_px(self, z, like_params, multiply_by_k=True, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None, ylim=None, ylim2=None, xlim=None, title=None, residual_to_theory=False):
        import matplotlib.lines as mlines
        """Plot the Px data and theory.
        
        Args:
        z (int or np.ndarray): index of redshift to plot, or array or indices
        every_other_theta (bool): if True, plots only half the theta bins
        """
        
        plt.rcParams.update({'font.size': 20})
        # plot all theta on one, easily distinguishable colors
        colors = plt.cm.tab10(np.linspace(0,1,len(self.data.theta_min_A_arcmin)))
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        skip = 1
        if every_other_theta:
            skip = 2
        
        
        # if (type(z) is list or type(z) is np.ndarray) and len(z)!=1:
        assert(len(np.atleast_1d(z))<5), "Too many redshifts passed. Plot fewer."
        # set a linestyle for every z
        linestyles = ["solid", "dashed", "dotted", "dashdot"]
        markers    = ["o", "s", "^", "D"]
        theory = self.get_convolved_Px_AA(np.atleast_1d(z),np.arange(len(self.data.theta_min_A_arcmin)),like_params)
        
        # make iz array in any case
        if type(z) in [int, np.integer]:
            z = [z]
        for theory_redshift_element, iz in enumerate(z):
            k = (self.data.k_M_edges[iz][:-1]+self.data.k_M_edges[iz][1:])/2.
            if multiply_by_k:
                factor = k
                ylabel = r'$k P_\times$'
            else:
                factor = 1.0
                ylabel = r'$P_\times$ [$\AA$]'
            for itheta in range(0,len(self.data.theta_min_A_arcmin),skip):
                if theory_redshift_element==0:
                    label = r'$\theta_A={:.2f}^\prime$'.format(self.data.theta_centers_arcmin[itheta])
                else:
                    label = None
                errors = np.diag(np.squeeze(self.data.cov_ZAM[iz, itheta, :, :]))**0.5
                div = errors
                divname = 'errors'
                ax[0].errorbar(k, self.data.Px_ZAM[iz, itheta, :]*factor, errors*factor, label=label, color=colors[itheta], linestyle='none', marker=markers[theory_redshift_element], markersize=5)
                theory_iz_iA = theory[theory_redshift_element,itheta]
                if residual_to_theory:
                    div = theory_iz_iA
                    divname = 'theory'
                
                ax[0].plot(k, theory_iz_iA*factor, color=colors[itheta], linestyle=linestyles[theory_redshift_element], linewidth=2)
                ax[1].set_xlabel(r'$k [\AA^{-1}]$')
                ax[1].plot(k, (self.data.Px_ZAM[iz, itheta, :]-theory_iz_iA)/div, color=colors[itheta], marker='o', linestyle='none')
        # if more than 1 z plotted, add custom legend for the redshifts: "--, square: z=.., -., diamond: z=.." etc
        ax[0].legend()
        handles, labels = ax[0].get_legend_handles_labels()
        if len(np.atleast_1d(z))>1:
            
            for theory_redshift_element, iz in enumerate(z):
                line = mlines.Line2D([], [], color='black', linestyle=linestyles[theory_redshift_element], marker=markers[theory_redshift_element], label=f'z={self.data.z[iz]:.2f}')
                handles.append(line)
                labels.append(f'z={self.data.z[iz]:.2f}')

        ax[1].axhline(0, color='black', linestyle='dashed', linewidth=1)
        ax[1].set_xlabel(r'$k [\AA^{-1}]$')
        ax[0].set_ylabel(ylabel)
        
        ax[1].set_ylabel(f'(Data-Theory)/{divname}')
        # ax[1].legend()
        if ylim2 is None:
            ax[1].set_ylim([-3,3])
        else:
            ax[1].set_ylim(ylim2)
        if ylim is not None:
            ax[0].set_ylim(ylim)
        if xlim is not None:
            ax[1].set_xlim(xlim)
        
        
        if theorylabel is None:
            theorylabel = 'Theory prediction, windowed'
        if datalabel is None:
            datalabel='Data'
        if title is not None:
            plt.suptitle(title)
        handles.append(plt.Line2D([], [], color='black', linestyle='solid', label=theorylabel))
        handles.append(plt.Line2D([], [], color='black', marker='o', linestyle='none', label=datalabel))
        ax[0].legend(handles=handles, loc='upper right', fontsize='small')
        if plot_fname is not None:
            plt.savefig(plot_fname + ".pdf")
            plt.savefig(plot_fname + ".png")
        else:
            if show:
                plt.show()
        return

    def fit_probability(self, like_params=None, z=None, n_free_p=0):
        """Compute probability given number of degrees of freedom"""
        chi2 = self.get_chi2(
            like_params=like_params, return_all=False
        )
        if z is None:
            z = self.z
            iz = self.data_iz
        ndeg = self.ndeg(iz)
        print(ndeg, "ndeg", chi2,"chi2", n_free_p, "n_free_p")
        prob = chi2_scipy.sf(chi2, ndeg - n_free_p)
        return prob

    def ndeg(self, iz):
        """Compute number of degrees of freedom in data"""
        ndeg = np.sum(self.data.Px_ZAM[iz] != 0)
        return ndeg
    

    def plot_cov_terms(self, save_directory=None):
        npanels = int(np.round(np.sqrt(len(self.cov_Pk_AA))))
        fig, ax = plt.subplots(
            npanels + 1, npanels, sharex=True, sharey=True, figsize=(10, 8)
        )
        ax = ax.reshape(-1)
        for ii in range(len(self.cov_Pk_AA)):
            cov_stat = np.diag(self.data.covstat_Pk_AA[ii])
            cov_syst = np.diag(self.data.cov_Pk_AA[ii]) - cov_stat
            cov_emu = np.diag(self.covemu_Pk_AA[ii])
            cov_tot = np.diag(self.cov_Pk_AA[ii])
            ax[ii].plot(
                self.data.k_AA[ii], cov_stat / cov_tot, label=r"$x$ = Stat"
            )
            ax[ii].plot(
                self.data.k_AA[ii], cov_syst / cov_tot, label=r"$x$ = Syst"
            )
            ax[ii].plot(
                self.data.k_AA[ii], cov_emu / cov_tot, label=r"$x$ = Emu"
            )
            ax[ii].text(0.0, 0.1, "z=" + str(self.data.z[ii]))
        if len(ax) > len(self.cov_Pk_AA):
            for ii in range(len(self.cov_Pk_AA), len(ax)):
                ax[ii].axis("off")
        ax[0].legend()
        fig.supxlabel(r"$k\,[\mathrm{km}^{-1}\mathrm{s}]$")
        fig.supylabel(r"$\sigma^2_x/\sigma^2_\mathrm{total}$")
        plt.tight_layout()
        
        if save_directory is not None:
            name = os.path.join(save_directory, "cov_terms")
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_cov_to_pk(self, save_directory=None):
        npanels = int(np.round(np.sqrt(len(self.cov_Pk_AA))))

        fig, ax = plt.subplots(
            npanels + 1, npanels, sharex=True, sharey="row", figsize=(10, 8)
        )
        ax = ax.reshape(-1)
        for ii in range(len(self.cov_Pk_AA)):
            cov_stat = np.diag(self.data.covstat_Pk_AA[ii])
            cov_syst = np.diag(self.data.cov_Pk_AA[ii]) - cov_stat
            cov_emu = np.diag(self.covemu_Pk_AA[ii])
            cov_tot = np.diag(self.cov_Pk_AA[ii])
            pk = self.data.Pk_AA[ii].copy()
            ax[ii].plot(
                self.data.k_AA[ii], np.sqrt(cov_stat) / pk, label=r"$x$ = Stat"
            )
            ax[ii].plot(
                self.data.k_AA[ii], np.sqrt(cov_syst) / pk, label=r"$x$ = Syst"
            )
            ax[ii].plot(
                self.data.k_AA[ii], np.sqrt(cov_emu) / pk, label=r"$x$ = Emu"
            )
            ax[ii].plot(
                self.data.k_AA[ii], np.sqrt(cov_tot) / pk, label=r"$x$ = Total"
            )
            ax[ii].text(
                0.2,
                0.97,
                "z=" + str(self.data.z[ii]),
                ha="right",
                va="top",
                transform=ax[ii].transAxes,
            )
        if len(ax) > len(self.cov_Pk_AA):
            for ii in range(len(self.cov_Pk_AA), len(ax)):
                ax[ii].axis("off")
        ax[0].legend(ncols=2)
        fig.supxlabel(r"$k\,[\mathrm{km}^{-1}\mathrm{s}]$")
        fig.supylabel(r"$\sigma_x/P_\mathrm{1D}$")
        plt.tight_layout()

        if save_directory is not None:
            name = os.path.join(save_directory, "cov_to_pk")
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_correlation_matrix(self, save_directory=None):
        def correlation_from_covariance(covariance):
            v = np.sqrt(np.diag(covariance))
            outer_v = np.outer(v, v)
            correlation = covariance / outer_v
            correlation[covariance == 0] = 0
            return correlation

        def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) > 0)

        plt.imshow(correlation_from_covariance(self.full_cov_Pk_AA))
        plt.colorbar()

        if save_directory is not None:
            name = os.path.join(save_directory, "correlation")
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()


    def set_Gauss_priors(self):
        """
        Sets Gaussian priors on the parameters
        """
        
        self.Gauss_priors = {}
        for par in self.free_params:
            self.Gauss_priors[par] = 1
        
        for par_like in self.free_params:
            if par_like.Gauss_priors_width is not None:
                _fid = par_like.ini_value
                _width = par_like.Gauss_priors_width
                _low = par_like.get_value_in_cube(_fid - 0.5 * _width) # bring these back when want to normalize
                _high = par_like.get_value_in_cube(_fid + 0.5 * _width)
                _prior = _high - _low
            else:
                print(par_like.name + " no prior")
                _prior = 1e4  # so we get zero
            self.Gauss_priors[par_like.name] = _prior
         
        if np.any(self.Gauss_priors != 1e4):
            pass
        else:
            self.Gauss_priors = None


    def like_parameter_by_name(self, pname):
        """Find parameter in list of likelihood free parameters"""
        return [p for p in self.like_params if p.name == pname][0]

    def index_by_name(self, pname):
        """Find parameter index in list of likelihood free parameters"""

        return [
            i for i, parname in enumerate(self.free_param_names) if parname == pname
        ][0]


    def sampling_point_from_parameters(self):
        """Translate likelihood parameters to array of values (in cube)"""

        values = np.zeros(len(self.free_param_names))
        for ii, par_name in enumerate(self.free_param_names):
            par = self.like_parameter_by_name(par_name)
            values[ii] = par.value_in_cube()

        return values

    def parameters_from_sampling_point(self, values):
        """Translate input array of values (in cube) to likelihood parameters"""

        if values is None:
            return []

        assert len(values) == len(self.free_param_names), "size mismatch"
        Npar = len(values)
        like_params = []
        for ip in range(Npar):
            par = self.like_parameter_by_name(self.free_param_names[ip]).get_new_parameter(values[ip])
            like_params.append(par)

        return like_params



