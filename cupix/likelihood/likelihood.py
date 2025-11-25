import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.stats.distributions import chi2 as chi2_scipy
from scipy.optimize import minimize
from scipy.linalg import block_diag
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta
import lace
from lace.cosmo import camb_cosmo
from cupix.utils.utils import is_number_string
from cupix.likelihood.likelihood_parameter import LikelihoodParameter

def get_bin_coverage(xmin_o, xmax_o, xmin_n, xmax_n):
    """Trick to accelerate rebinning"""
    # check out https://stcorp.github.io/harp/doc/html/algorithms/regridding.html
    cover = np.zeros((len(xmin_n), len(xmin_o)))
    for jj in range(len(xmin_n)):
        cover[jj] = np.fmax(
            (np.fmin(xmax_o, xmax_n[jj]) - np.fmax(xmin_o, xmin_n[jj]))
            / (xmax_o - xmin_o),
            0,
        )
    return cover


class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

    def __init__(
        self,
        data,
        theory,
        iz_choice,
        like_params,
        free_param_names=None,
        free_param_limits=None,
        verbose=False,
        prior_Gauss_rms=None,
        min_log_like=-1e100,
    ):
        """Setup likelihood from theory and data. Options:
        - data (required) is the data to model
        - theory (required) instance of lya_theory
        - free_param_names is a list of param names, in any order
        - free_param_limits list of tuples, same order than free_param_names
        - if prior_Gauss_rms is None it will use uniform priors
        - min_log_like: use this instead of - infinity"""

        self.verbose = verbose
        self.prior_Gauss_rms = prior_Gauss_rms
        self.min_log_like = min_log_like
        self.data = data
        self.free_param_names = free_param_names
        self.free_param_limits = free_param_limits
        self.iz_choice = iz_choice
        self.like_params = like_params
        
        # we only do this for latter save all relevant after fitting the model
        # self.args = {}
        # for attr, value in args.__dict__.items():
        #     if attr not in ["archive", "emulator"]:
        #         self.args[attr] = value

        
        self.theory = theory
        
        self.set_Gauss_priors()

        # store also fiducial model
        # self.set_fid()




    def get_convolved_Px_AA(self,
        iz,
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

        squeeze_result = False
        if type(theta_A) is not list and type(theta_A) is not np.ndarray:
            theta_A = [theta_A]
            squeeze_result = True
        zs = np.atleast_1d(self.data.z)
        k_AA_fine = self.data.k_m
        # Compute Px in vectorized way (all theta at once)
        theta_a_arcmin = (self.data.theta_min_a_arcmin + self.data.theta_max_a_arcmin)/2.
        if self.verbose:
            print("Computing Px simultaneously for", theta_a_arcmin)
        Px_Zam = self.theory.get_px_AA(
            zs = zs[iz],
            k_AA=k_AA_fine,
            theta_arcmin=theta_a_arcmin,
            like_params=like_params,
            verbose=self.verbose
        )

        # loop through the large-theta bins from the data
        Px_ZAM_all = []
        for itheta_A in theta_A:
            # get the small-theta indices in this coarse-theta bin
            ind_in_theta = self.data.B_A_a.astype(bool)[itheta_A,:]
            theta_a_inds = np.where(ind_in_theta)[0]
            
            k_AA_binned = (self.data.k_M_edges[:-1] + self.data.k_M_edges[1:]) / 2.
            
            Px_ZaM_all = np.zeros((len(theta_a_inds), len(k_AA_binned)))
            V_ZaM_all  = np.zeros((len(theta_a_inds), len(k_AA_binned)))
            

            for save_index, a in enumerate(theta_a_inds):
                # retrieve the window matrix for this small-theta bin
                if self.data.U_ZaMn is not None:
                    U_ZaMn = self.data.U_ZaMn[iz,a]
                    Px_ZaM = convolve_window(U_ZaMn,Px_Zam[a].T) # check later if the window convolution can also be vectorized

                Px_ZaM_all[save_index,:] = Px_ZaM
                V_ZaM = self.data.V_ZaM[iz,a]
                V_ZaM_all[save_index,:] = V_ZaM
            # rebin in theta
            Px_ZAM = rebin_theta(V_ZaM_all, Px_ZaM_all)
            Px_ZAM_all.append(Px_ZAM)
        if squeeze_result:
            return np.squeeze(np.asarray(Px_ZAM_all))
        else:
            return np.asarray(Px_ZAM_all)



    def get_chi2(self, values, return_all=False):
        """Compute chi2 using data and theory, without adding
        emulator covariance"""
        
        log_like, log_like_all = self.get_log_like(
            values=values,
            ignore_log_det_cov=True
        )

        # print(-2 * log_like, -2 * log_like_all, -2 * np.sum(log_like_all))

        if return_all:
            return -2.0 * log_like, -2.0 * log_like_all
        else:
            return -2.0 * log_like

    def get_log_like(
        self,
        values,
        ignore_log_det_cov=True
    ):
        """Compute log(likelihood), including determinant of covariance
        unless you are setting ignore_log_det_cov=True."""
        iz = self.iz_choice
        # what to return if we are out of priors
        null_out = [-np.inf, -np.inf]
        data = self.data
        ntheta = len(data.theta_max_A_arcmin)
        # compute log like contribution from each redshift bin
        log_like_all = np.zeros((1, ntheta))
        log_like = 0
        # get the parameters for this iteration
        params = self.like_params.copy()
        
        # fill in the free parameters
        # free parameter names must match the order of values
        fp_index = 0
        for i,p in enumerate(params):
            if p.name in self.free_param_names:
                # translate the value in [0,1] to the actual parameter value
                free_param_value = p.value_from_cube(values[fp_index])
                params[i] = LikelihoodParameter(
                    name=p.name,
                    min_value=p.min_value,
                    max_value=p.max_value,
                    value=free_param_value)
                fp_index += 1
        for p in params:
            print(p.name, p.value)
        
        theta_A = np.arange(ntheta)
        model_iz = self.get_convolved_Px_AA(iz,theta_A,params)
        for itheta in range(ntheta):
            # compute chi2 for this redshift bin
            det_cov = np.linalg.det(self.data.cov_ZAM[iz,itheta,:,:])
            assert det_cov != 0, "Singular covariance matrix!"
            icov_ZAM = np.linalg.inv(self.data.cov_ZAM[iz,itheta,:,:])
            data_izitheta  = data.Px_ZAM[iz,itheta,:]
            
            diff = data_izitheta - model_iz[itheta]
            chi2_z = np.dot(np.dot(icov_ZAM, diff), diff)
            # check whether to add determinant of covariance as well
            if ignore_log_det_cov:
                log_like_all[iz, itheta] = -0.5 * chi2_z
            else:
                det_icov = np.linalg.det(icov_ZAM)
                assert np.isfinite(det_icov), "Non-finite determinant of inverse covariance!"
                log_det_cov = np.log(
                    np.abs(1 / np.linalg.det(icov_ZAM))
                )
                log_like_all[iz, itheta] = -0.5 * (chi2_z + log_det_cov)
        log_like += np.sum(log_like_all)

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

    def compute_log_prob(
        self, values, ignore_log_det_cov=True
    ):
        """Compute log likelihood plus log priors for input values"""

        # Always force parameter to be within range (for now)
        # if (max(values) > 1.0) or (min(values) < 0.0):
        #     return self.min_log_like

        # compute log_prior
        log_prior = self.get_log_prior(values)
        # compute log_like (option to ignore emulator covariance)
        
        log_like, chi2_all = self.get_log_like(
            values,
            ignore_log_det_cov=ignore_log_det_cov
        )

        # regulate log-like (not NaN, not tiny)
        log_like = self.regulate_log_like(log_like)


        return log_like + log_prior

    def log_prob(self, values, ignore_log_det_cov=True):
        """Return log likelihood plus log priors"""

        return self.compute_log_prob(
            values,
            ignore_log_det_cov=ignore_log_det_cov
        )



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
        print("values at beginning are", values)
        if not np.isfinite(self.log_prob(values)):
            print("Non-finite value detected:", values, self.log_prob(values))

        return -1.0 * self.log_prob(values)

    def maximise_posterior(
        self, initial_values=None, method="nelder-mead", tol=1e-4
    ):
        """Run scipy minimizer to find maximum of posterior"""

        if not initial_values:
            initial_values = np.ones(len(self.free_param_names)) * 0.5

        return minimize(
            self.minus_log_prob, x0=initial_values, method=method, tol=tol
        )


    def plot_px(self, z, like_params, multiply_by_k=True, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None, ylim=None, xlim=None):
        """Plot the Px data and theory.
        
        Args:
        z (int or np.ndarray): index of redshift to plot, or array or indices
        every_other_theta (bool): if True, plots only half the theta bins
        """
        
        plt.rcParams.update({'font.size': 20})
        # plot all theta on one, easily distinguishable colors
        colors = plt.cm.tab10(np.linspace(0,1,len(self.data.theta_min_A_arcmin)))
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        k = (self.data.k_M_edges[:-1]+self.data.k_M_edges[1:])/2.
        skip = 1
        if every_other_theta:
            skip = 2
        if multiply_by_k:
            factor = k
            ylabel = r'$k P_\times$'
        else:
            factor = 1.0
            ylabel = r'$P_\times$ [$\AA$]'
        if (type(z) is list or type(z) is np.ndarray) and len(z)!=1:
            assert(len(z)<5), "Too many redshifts passed. Plot fewer."
            # set a linestyle for every z
            linestyles = ["solid", "dashed", "dotted", "dashdot"]

            for iz in z:
                for itheta in range(0,len(self.data.theta_min_A_arcmin),skip):
                    errors = np.diag(self.data.cov_ZAM[iz, itheta, :, :])**0.5
                    ax[0].errorbar(k, self.data.Px_ZAM[iz, itheta, :]*factor, errors*factor, label=r'$\theta_A={:.2f}^\prime$'.format(self.data.theta_centers_arcmin[itheta]), color=colors[itheta], marker='o', linestyle=linestyles[itheta])
                    theory_iA_iz = self.get_convolved_Px_AA(iz,itheta,like_params)
                    ax[0].plot(k, theory_iA_iz*factor, color=colors[itheta], linestyle='solid')
                    ax[1].set_xlabel(r'$k [\AA^{-1}]$')
                    ax[1].plot(k, (self.data.Px_ZAM[iz, itheta, :]-theory_iA_iz)/errors, color=colors[itheta], marker='o', linestyle=linestyles[itheta])
        else:
            for itheta in range(0,len(self.data.theta_min_A_arcmin),skip):
                errors = np.diag(self.data.cov_ZAM[z, itheta, :, :])**0.5
                ax[0].errorbar(k, self.data.Px_ZAM[z, itheta, :]*factor, errors*factor, label=r'$\theta_A={:.2f}^\prime$'.format(self.data.theta_centers_arcmin[itheta]), color=colors[itheta], linestyle='none', marker='o')
                theory_iA_iz = self.get_convolved_Px_AA(z,itheta,like_params)
                ax[0].plot(k, theory_iA_iz*factor, color=colors[itheta], linestyle='solid')
                ax[1].plot(k, (self.data.Px_ZAM[z, itheta, :]-theory_iA_iz)/errors, color=colors[itheta], marker='o', linestyle='none')
            ax[1].axhline(0, color='black', linestyle='dashed', linewidth=1)
            ax[1].set_xlabel(r'$k [\AA^{-1}]$')
            ax[0].set_ylabel(ylabel)
            ax[0].legend()
            ax[1].set_ylabel('(Data-Theory)/error')
            # ax[1].legend()
            ax[1].set_ylim([-3,3])
            if ylim is not None:
                ax[0].set_ylim(ylim)
            if xlim is not None:
                ax[1].set_xlim(xlim)
        
        handles, labels = ax[0].get_legend_handles_labels()
        if theorylabel is None:
            theorylabel = 'Theory prediction, windowed'
        if datalabel is None:
            datalabel='Data'
        
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

    def fit_probability(self, values, Z=None, n_free_p=None):
        """Compute probability given number of degrees of freedom"""
        if n_free_p is None:
            n_free_p = len(self.free_param_names)
        chi2 = self.get_chi2(
            values=values, return_all=False
        )
        if Z is None:
            Z = self.iz_choice
        ndeg = self.ndeg(Z)
        prob = chi2_scipy.sf(chi2, ndeg - n_free_p)
        return prob

    def ndeg(self, Z):
        """Compute number of degrees of freedom in data"""

        ndeg = 0
        if type(Z) in [int, np.integer]:
            Z = [Z]
        for iz in range(len(Z)):
            ndeg += np.sum(self.data.Px_ZAM[iz] != 0)
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

    def plot_hull_fid(self):
        emu_call, M_of_z = self.theory.get_emulator_calls(self.data.z)
        p1 = np.zeros(
            (
                self.theory.hull.nz,
                len(self.theory.hull.params),
            )
        )
        for jj, key in enumerate(self.theory.hull.params):
            p1[:, jj] = emu_call[key]

        self.theory.hull.plot_hulls(p1)

    def set_Gauss_priors(self):
        """
        Sets Gaussian priors on the parameters
        """

        self.Gauss_priors = np.ones((len(self.free_param_names)))
        ii = 0
        for par_like in self.like_params:
            if par_like.name in self.free_param_names:
                if par_like.Gauss_priors_width is not None:
                    _fid = par_like.ini_value
                    _width = par_like.Gauss_priors_width
                    # _low = _fid - 0.5 * _width
                    # _high = _fid + 0.5 * _width
                    _low = par_like.get_value_in_cube(_fid - 0.5 * _width) # bring these back when want to normalize
                    _high = par_like.get_value_in_cube(_fid + 0.5 * _width)
                    _prior = _high - _low
                else:
                    print(par_like.name + " no prior")
                    _prior = 1e4  # so we get zero
                self.Gauss_priors[ii] = _prior
                ii += 1
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
