import numpy as np
import os
import h5py

from cupix.px_data.base_px_data import BaseDataPx


class Px_Lyacolore(BaseDataPx):
    """Class containing Px from Lyacolore"""

    def __init__(self, filename):
        """Read measured Px."""

        # folder storing P1D measurements
        datadir = BaseDataPx.BASEDIR + "/Lyacolore/"
        self.filepath = os.path.join(datadir, filename)
        # read redshifts, wavenumbers, power spectra and covariance matrices
        k_m, k_M_edges, theta_min_a, theta_max_a, theta_min_A, theta_max_A, zbin_centers, N_fft, L_fft, B_A_a = self.read_from_file()
        Nz = len(zbin_centers)
        Nk_M = len(k_M_edges)-1
        Ntheta_A = len(theta_min_A)
        Ntheta_a = len(theta_min_a)
        # store the data as a 3D array of (Nz, Ntheta, Nk)
        Px_ZAM = np.zeros((Nz, Ntheta_A, Nk_M))
        cov_ZAM = np.zeros((Nz, Ntheta_A, Nk_M, Nk_M))
        for iz in range(Nz):
            for A in range(Ntheta_A):
                Px_ZAM[iz, A, :] = self.get_Px_z_T(iz, A)
                cov_ZAM[iz, A, :, :] = self.get_cov_matrix_z_T(iz, A)
        
        # store the window matrices
        Nk_m = len(k_m)
        U_ZaMn = np.zeros((Nz, Ntheta_a, Nk_M, Nk_m))
        V_ZaM = np.zeros((Nz, Ntheta_a, Nk_M))
        for iz in range(Nz):
            for a in range(Ntheta_a):
                U_ZaMn[iz, a, :, :] = self.get_window_matrix_z_t(iz, a)
                V_ZaM[iz, a, :] = self.get_V_Z_aM(iz, a)

        super().__init__(Px_ZAM,
                        cov_ZAM,
                        zbin_centers,
                        k_M_edges,
                        theta_min_A,
                        theta_max_A,
                        N_fft,
                        L_fft,
                        has_theta_rebinning=True,
                        has_k_rebinning=True,
                        k_m_AA=k_m,
                        theta_min_a_arcmin=theta_min_a,
                        theta_max_a_arcmin=theta_max_a,
                        B_A_a=B_A_a,
                        U_ZaMn=U_ZaMn,
                        V_ZaM=V_ZaM,
                        filepath=self.filepath)
        return



    def read_from_file(self):
        with h5py.File(self.filepath,'r') as f:
        # Read metadata
            k_m = f['metadata'].attrs['k_m']
            k_M_edges = f['metadata'].attrs['k_M_edges']
            theta_min_a = f['metadata'].attrs['theta_min_a']
            theta_max_a = f['metadata'].attrs['theta_max_a']
            theta_min_A = f['metadata'].attrs['theta_min_A']
            theta_max_A = f['metadata'].attrs['theta_max_A']
            zbin_centers = f['metadata'].attrs['z_centers']
            N_fft = f['metadata'].attrs['N_fft']
            L_fft = f['metadata'].attrs['L_fft']
            B_A_a = f['metadata/B_A_a'][:]
        return k_m, k_M_edges, theta_min_a, theta_max_a, theta_min_A, theta_max_A, zbin_centers, N_fft, L_fft, B_A_a

    def get_Px_z_T(self,zbin_ind,theta_bin_ind):
        with h5py.File(self.filepath,'r') as f:
            P_Z_AM = f['P_Z_AM/z_{}/theta_rebin_{}/'.format(zbin_ind,theta_bin_ind)][:] # shape (NK,)
            return P_Z_AM
  
    def get_cov_matrix_z_T(self,zbin_ind,theta_bin_ind):
        with h5py.File(self.filepath,'r') as f:
            cov_matrix = f['C_Z_AMN/z_{}/theta_rebin_{}/'.format(zbin_ind,theta_bin_ind)][:] # shape (NK,NK)
            return cov_matrix
  
    def get_window_matrix_z_t(self,zbin_ind,theta_bin_ind):
        with h5py.File(self.filepath,'r') as f:
            window_matrix = f['U_Z_aMn/z_{}/theta_{}/'.format(zbin_ind,theta_bin_ind)][:] # shape (NK,Nk)
            return window_matrix

    def get_V_Z_aM(self,zbin_ind,theta_bin_ind):
        with h5py.File(self.filepath,'r') as f:
            norm_V = f['V_Z_aM/z_{}/theta_{}/'.format(zbin_ind,theta_bin_ind)][:] # shape (NK,)
        return norm_V
