import numpy as np
import math
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta

class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

    def __init__(
        self,
        data,
        theory,
        iz,
        verbose=False,
    ):
        """Setup likelihood from theory and data. Options:
        - data (required) is the data to model
        - theory (required) instance of lya_theory
        - iz is the index of the redshift bin to use
        """

        self.verbose = verbose
        self.data = data
        self.iz = iz
        self.theory = theory
        assert self.data.z[iz] == self.theory.z, "inconsistent redshifts"
        if self.verbose:
            print("Likelihood will evaluate redshift bin", self.iz, "corresponds to z =", self.theory.z)
        

    def get_convolved_px(self, params={}):
        # array with discrete k values (inverse Angstroms)
        k_m = self.data.k_m[self.iz]

        # array with centers of the original theta bins (in arcmin)
        theta_a = (self.data.theta_min_a_arcmin + self.data.theta_max_a_arcmin)/2.

        # evaluate theory at these values (no rebinning, no convolution)
        Px_Zam = self.theory.get_px_obs(theta_arc=theta_a, k_AA=k_m, params=params)

        # number of rebinned k_M and theta_A bins
        N_M = len(self.data.k_M_edges[self.iz]) - 1
        N_A = len(self.data.theta_max_A_arcmin)
        if self.verbose:
            print('Working with {} k bins, and {} theta bins'.format(N_M, N_A))

        # window convolution
        # loop through the large-theta bins from the data
        Px_ZAM_all = []
        for it_A in range(N_A):
            # get the small-theta indices in this coarse-theta bin
            ind_in_theta = self.data.B_A_a.astype(bool)[it_A,:] # generally this could be different with redshift; assume it's not for now
            theta_a_inds = np.where(ind_in_theta)[0]
            # collect Px and V from each narrow bin
            Px_ZaM_all = np.zeros((len(theta_a_inds), N_M))
            V_ZaM_all  = np.zeros((len(theta_a_inds), N_M))
            for save_index, a in enumerate(theta_a_inds):
                # retrieve the window matrix for this small-theta bin
                if self.data.U_ZaMn is not None:
                    U_ZaMn = self.data.U_ZaMn[self.iz, a]
                    # print("U_ZaMn shape", U_ZaMn.shape)
                    # print("Px_ZaM shape to be convolved", Px_Zam[a].T.shape)
                    Px_ZaM = convolve_window(U_ZaMn, Px_Zam[a].T)
                Px_ZaM_all[save_index,:] = Px_ZaM
                V_ZaM = self.data.V_ZaM[self.iz, a]
                V_ZaM_all[save_index,:] = V_ZaM
            # rebin in theta
            Px_ZAM = rebin_theta(V_ZaM_all, Px_ZaM_all)
            Px_ZAM_all.append(Px_ZAM)

        return np.asarray(Px_ZAM_all)


    def get_chi2(self, params={}, return_info=False):
        log_like, info = self._compute_log_like(params)
        chi2 = -2.0 * log_like
        if return_info:
            return chi2, info
        else:
            return chi2


    def get_log_like(self, params={}, return_info=False):
        log_like, info = self._compute_log_like(params=params)
        if return_info:
            return log_like, info
        else:
            return log_like


    def _compute_log_like(self, params={}):

        # get model prediction, including convolution
        model_px = self.get_convolved_px(params=params)
        if self.verbose:
            print('shape model_px', model_px.shape)
        N_A, N_M = model_px.shape

        # compute log like contributions from each theta bin
        log_like_all = np.zeros(N_A)
        for it_A in range(N_A):
            det_cov = np.linalg.det(self.data.cov_ZAM[self.iz, it_A,:,:])
            if det_cov == 0:
                print("Det(cov) appears to be 0: could be a singular covariance matrix!")
            # compute chi2 for this theta bin
            icov_ZAM = np.linalg.inv(self.data.cov_ZAM[self.iz, it_A,:,:])
            data_A = self.data.Px_ZAM[self.iz, it_A,:]
            diff = np.squeeze(data_A - model_px[it_A])
            chi2 = np.dot(np.dot(np.squeeze(icov_ZAM), diff), diff)
            log_like_all[it_A] = -0.5*chi2

        log_like = np.sum(log_like_all)
        info = {'log_like_all': log_like_all}

        return log_like, info


    def plot_px(self, params={}, multiply_by_k=True, every_other_theta=False, show=True,
                theorylabel=None, datalabel=None, plot_fname=None,
                ylim=None, ylim2=None, xlim=None, title=None, residual_to_theory=False):
        """Plot the Px data and theory."""
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines

        # get theory prediction
        model_px = self.get_convolved_px(params=params)
        Nt_A, Nk_M = model_px.shape

        # plot all theta on one, easily distinguishable colors
        plt.rcParams.update({'font.size': 20})
        colors = plt.cm.tab10(np.linspace(0, 1, Nt_A))
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        skip = 1
        if every_other_theta:
            skip = 2

        # central value of k bins
        k_M = (self.data.k_M_edges[self.iz][:-1] + self.data.k_M_edges[self.iz][1:])/2.
        if multiply_by_k:
            factor = k_M
            ylabel = r'$k P_\times$'
        else:
            factor = 1.0
            ylabel = r'$P_\times$ [$\AA$]'

        # loop over theta bins (skipping some if needed)
        for it_A in range(0, Nt_A, skip):
            label = r'$\theta_A={:.2f}^\prime$'.format(self.data.theta_centers_arcmin[it_A])
            errors = np.diag(np.squeeze(self.data.cov_ZAM[self.iz, it_A, :, :]))**0.5
            div = errors
            divname = 'errors'
            ax[0].errorbar(k_M, self.data.Px_ZAM[self.iz, it_A, :]*factor, errors*factor, label=label, color=colors[it_A], linestyle='none', marker='^', markersize=5)
            theory_iA = model_px[it_A]
            if residual_to_theory:
                div = theory_iA
                divname = 'theory'

            ax[0].plot(k_M, theory_iA*factor, color=colors[it_A], linewidth=2)
            ax[1].set_xlabel(r'$k [\AA^{-1}]$')
            ax[1].plot(k_M, (self.data.Px_ZAM[self.iz, it_A, :] - theory_iA)/div, color=colors[it_A], marker='o', linestyle='none')

        # if more than 1 z plotted, add custom legend for the redshifts: "--, square: z=.., -., diamond: z=.." etc
        ax[0].legend()
        handles, labels = ax[0].get_legend_handles_labels()
        ax[1].axhline(0, color='black', linestyle='dashed', linewidth=1)
        ax[1].set_xlabel(r'$k [\AA^{-1}]$')
        ax[0].set_ylabel(ylabel)
        ax[1].set_ylabel(f'(Data-Theory)/{divname}')
        # ax[1].legend()

        # set range limits
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
