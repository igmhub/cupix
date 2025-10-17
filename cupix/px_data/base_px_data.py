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
        theta_max_cut_arcmin=None
    ):
        """Construct base Px class, from measured power and covariance"""

        self.z = np.array(z)
        self.k_M_edges = k_M_edges_AA
        self.theta_min_a_arcmin = theta_min_a_arcmin
        self.theta_min_A_arcmin = theta_min_A_arcmin
        self.theta_max_A_arcmin = theta_max_A_arcmin
        self.z = z
        self.N_fft = N_fft
        self.L_fft = L_fft

        # set theta centers
        self.theta_centers_arcmin = 0.5 * (self.theta_min_A_arcmin + self.theta_max_A_arcmin)

        if filepath is not None:
            self.filepath = filepath
        else:
            self.filepath = None

        if has_k_rebinning:
            assert k_m_AA is not None, "k_m_AA must be provided if has_k_rebinning is True"
            self.k_m = k_m_AA
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

        if (theta_min_arcmin < self.theta_min_A_arcmin[0]) or (theta_max_arcmin > self.theta_max_A_arcmin[-1]):
            raise ValueError("Provided theta limits are out of bounds of the current data.")

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