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
    ):
        """Construct base Px class, from measured power and covariance"""

        self.z = np.array(z)
        self.k_M_edges = k_M_edges_AA
        self.theta_min_a = theta_min_a_arcmin
        self.theta_min_A = theta_min_A_arcmin
        self.theta_max_A = theta_max_A_arcmin
        self.z = z
        self.N_fft = N_fft
        self.L_fft = L_fft

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
            self.theta_min_a = theta_min_a_arcmin
            self.theta_max_a = theta_max_a_arcmin
            self.B_A_a = B_A_a
            self.has_theta_rebinning = True
            self.Px_ZAM = Px_ZAM
            self.cov_ZAM = cov_ZAM
            self.U_ZaMn = U_ZaMn
            self.V_ZaM = V_ZaM
        return