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

