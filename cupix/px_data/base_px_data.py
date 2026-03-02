import os, sys
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
import h5py
from cupix.utils.utils import get_path_repo


class BaseDataPx(object):
    """Base class to store measurements of the cross power spectrum"""

    BASEDIR = os.path.join(get_path_repo("cupix"), "data", "px_measurements")


    def __init__(
        self,
        Px_ZAM,
        cov_ZAM,
        z,
        k_M_edges_AA,
        theta_min_A_arcmin,
        theta_max_A_arcmin,
        N_fft,
        L_fft,
        has_theta_rebinning=True,
        has_k_rebinning=True,
        k_m_AA=None,
        theta_min_a_arcmin=None,
        theta_max_a_arcmin=None,
        B_A_a=None,
        U_ZaMn=None,
        V_ZaM=None,
        filepath=None,
        theta_min_cut_arcmin=None,
        theta_max_cut_arcmin=None,
        kmin_cut_AA=None,
        kmax_cut_AA=None
    ):
        """Construct base Px class, from measured power and covariance"""

        self.z = np.array(z)
        self.theta_min_a_arcmin = theta_min_a_arcmin
        self.theta_min_A_arcmin = theta_min_A_arcmin
        self.theta_max_A_arcmin = theta_max_A_arcmin
        self.N_fft = N_fft
        self.L_fft = L_fft
        self.Nz = len(self.z)

        if filepath is not None:
            self.filepath = filepath
        else:
            self.filepath = None

        if has_k_rebinning:
            assert k_m_AA is not None, "k_m_AA must be provided if has_k_rebinning is True"
            if self.Nz>1 and k_m_AA.shape[0]!=self.Nz:
                # set k_centers one time for easy plotting
                self.k_M_centers_AA = 0.5 * (k_M_edges_AA[:-1] + k_M_edges_AA[1:])
                # duplicate to make 1 k for each z
                self.k_m = np.tile(k_m_AA,(self.Nz,1))
                self.k_M_edges = np.tile(k_M_edges_AA,(self.Nz,1))
            elif Nz==1:
                self.k_m = [k_m_AA]
                self.k_M_edges = [k_M_edges_AA]
                self.k_M_centers_AA = 0.5 * (k_M_edges_AA[:-1] + k_M_edges_AA[1:])
            else:
                self.k_m = k_m_AA
                self.k_M_edges = k_M_edges_AA
                self.k_M_centers_AA = [0.5 * (k_M_edges_AA[iz][:-1] + k_M_edges_AA[iz][1:]) for iz in range(self.Nz)]
                
            self.has_k_rebinning = True
        if has_theta_rebinning:
            assert theta_min_a_arcmin is not None, "theta_min_a_arcmin must be provided if has_theta_rebinning is True"
            assert theta_max_a_arcmin is not None, "theta_max_a_arcmin must be provided if has_theta_rebinning is True"
            assert B_A_a is not None, "B_A_a must be provided if has_theta_rebinning is True"
            self.theta_min_a_arcmin = theta_min_a_arcmin
            self.theta_max_a_arcmin = theta_max_a_arcmin
            self.B_A_a = B_A_a
            self.has_theta_rebinning = True
            self.Px_ZAM = Px_ZAM
            self.cov_ZAM = cov_ZAM
            self.U_ZaMn = U_ZaMn
            self.V_ZaM = V_ZaM

        if theta_min_cut_arcmin is not None or theta_max_cut_arcmin is not None:
            self.limit_theta_range(theta_min_arcmin=theta_min_cut_arcmin, theta_max_arcmin=theta_max_cut_arcmin)
        if kmin_cut_AA is not None or kmax_cut_AA is not None:
            self.limit_k_range(k_min_AA=kmin_cut_AA, k_max_AA=kmax_cut_AA)
        # set theta centers
        self.theta_centers_arcmin = 0.5 * (self.theta_min_A_arcmin + self.theta_max_A_arcmin)
        return
    

    def limit_theta_range(self, theta_min_arcmin=None, theta_max_arcmin=None):
        """Limit the theta range of the data to [theta_min_arcmin, theta_max_arcmin]"""

        if (theta_min_arcmin is None) and (theta_max_arcmin is None):
            warn("No theta limits provided. No changes made.")
            return

        if theta_min_arcmin is None:
            theta_min_arcmin = self.theta_min_A_arcmin[0]
        if theta_max_arcmin is None:
            theta_max_arcmin = self.theta_max_A_arcmin[-1]


        # find indices of theta bins within the specified range
        indices_A = np.where((self.theta_min_A_arcmin >= theta_min_arcmin) & (self.theta_max_A_arcmin <= theta_max_arcmin))[0]
        if len(indices_A) == 0:
            raise ValueError("No theta bins found within the specified range.")
        if self.has_theta_rebinning:
            indices_a = np.where(
                (self.theta_min_a_arcmin >= theta_min_arcmin) & (self.theta_max_a_arcmin <= theta_max_arcmin)
            )[0]
            if len(indices_a) == 0:
                raise ValueError("No theta bins found within the specified range.")
            self.theta_min_a_arcmin = self.theta_min_a_arcmin[indices_a]
            self.theta_max_a_arcmin = self.theta_max_a_arcmin[indices_a]
            self.U_ZaMn = self.U_ZaMn[:, indices_a, :, :]
            self.V_ZaM = self.V_ZaM[:, indices_a, :]
            self.B_A_a = self.B_A_a[indices_A, :][:, indices_a]



        # update theta bin edges and rebinning matrix
        self.theta_min_A_arcmin = self.theta_min_A_arcmin[indices_A]
        self.theta_max_A_arcmin = self.theta_max_A_arcmin[indices_A]

        # update Px_ZAM and cov_ZAM
        self.Px_ZAM = self.Px_ZAM[:, indices_A, :]
        self.cov_ZAM = self.cov_ZAM[:, indices_A, :, :]

        return
    
    def limit_k_range(self, k_min_AA=None, k_max_AA=None):
        """Limit the k range of the data to [k_min_AA, k_max_AA]"""
        """Eventually, I can have this more flexible to allow different k limits for each redshift"""

        if (k_min_AA is None) and (k_max_AA is None):
            warn("No k limits provided. No changes made.")
            return
        new_k_M_edges = []
        new_Px_ZAM = []
        new_cov_ZAM = []
        new_U_ZaMn = []
        new_V_ZaM = []
        for iz in range(self.Nz):
            if k_min_AA is None:
                indices_M = np.where(self.k_M_edges[iz][:-1] <= k_max_AA)[0]
            elif k_max_AA is None:
                indices_M = np.where(self.k_M_edges[iz][:-1] >= k_min_AA)[0]
            else:
                indices_M = np.where((self.k_M_edges[iz][:-1] >= k_min_AA) & (self.k_M_edges[iz][1:] <= k_max_AA))[0]
            if len(indices_M) == 0:
                raise ValueError("No k bins found within the specified range.")        
            new_k_M_edges.append(self.k_M_edges[iz][np.concatenate(([indices_M[0]], indices_M + 1))])
            # update Px_ZAM and cov_ZAM
            new_Px_ZAM.append(np.take(self.Px_ZAM[iz], indices_M, axis=1))
            tmp = np.take(self.cov_ZAM[iz], indices_M, axis=1)
            tmp = np.take(tmp, indices_M, axis=2)
            new_cov_ZAM.append(tmp)
            if self.has_theta_rebinning:
                # update U_ZaMn
                new_U_ZaMn.append(np.take(self.U_ZaMn[iz], indices_M, axis=1))
                new_V_ZaM.append(np.take(self.V_ZaM[iz], indices_M, axis=1))
            if self.k_M_centers_AA.shape[0] == self.Nz:
                self.k_M_centers_AA[iz] = self.k_M_centers_AA[iz][indices_M]
            else:
                if iz==0:
                    self.k_M_centers_AA = self.k_M_centers_AA[indices_M]
        self.k_M_edges = np.array(new_k_M_edges)
        self.Px_ZAM = np.array(new_Px_ZAM)
        self.cov_ZAM = np.array(new_cov_ZAM)
        self.U_ZaMn = np.array(new_U_ZaMn)
        self.V_ZaM = np.array(new_V_ZaM)
        
        return