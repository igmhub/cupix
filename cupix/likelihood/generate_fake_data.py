# generate fake data from Arinyo model, for testing sensitivity of the data to parameters

import numpy as np
import h5py
# fake_data object will covariance matrix, Arinyo model etc

class FakeData(object):
    """Class to generate fake data from Arinyo model"""

    def __init__(self, likelihood):
        """Initialize fake data generator with theory and real data"""
        self.like = likelihood

    def add_noise(self, theory_datavector, cov):
        """Add noise to theory datavector, given covariance matrix"""
        L = np.linalg.cholesky(cov)
        n = np.random.normal(size=theory_datavector.shape)
        noisy_datavector = theory_datavector + np.dot(L, n)
        return noisy_datavector


    def generate_px(self, iz_choice, theta_A_ind, like_params=None):
        # set up the Arinyo P3D model
        # do we want to apply the noise before or after the window convolution?
        # has to be after because I have the covariance matrix for the post-rebinned data
        if like_params is None:
            # use the default parameters from the likelihood
            Px_theory = self.like.get_convolved_Px_AA(iz_choice, theta_A_ind, self.like.like_params)
        else:
            Px_theory = self.like.get_convolved_Px_AA(iz_choice, theta_A_ind, like_params)
        # add noise
        if type(theta_A_ind) in [int, np.int64, np.int32]:
            cov = self.like.data.cov_ZAM[iz_choice,theta_A_ind,:,:]
            noisy_Px = self.add_noise(Px_theory, cov)
        elif type(theta_A_ind) in [np.ndarray, list]:
            noisy_Px = []
            for theta_A_i in theta_A_ind:
                cov = self.like.data.cov_ZAM[iz_choice,theta_A_i,:,:]
                Px_theory_i = Px_theory[theta_A_i]
                noisy_Px_i = self.add_noise(Px_theory_i, cov)
                noisy_Px.append(noisy_Px_i)
            noisy_Px = np.array(noisy_Px)
        return noisy_Px
    
    def write_to_file(self, filepath, like_params=None):
        with h5py.File(filepath,'w') as f:
            metadata = f.create_group('metadata')
            # Reproduce metadata from data file
            metadata.attrs['k_m'] = self.like.data.k_m
            metadata.attrs['k_M_edges'] = self.like.data.k_M_edges
            metadata.attrs['theta_min_a'] = self.like.data.theta_min_a_arcmin
            metadata.attrs['theta_max_a'] = self.like.data.theta_max_a_arcmin
            metadata.attrs['theta_min_A'] = self.like.data.theta_min_A_arcmin
            metadata.attrs['theta_max_A'] = self.like.data.theta_max_A_arcmin
            metadata.attrs['z_centers'] = self.like.data.z
            metadata.attrs['N_fft'] = self.like.data.N_fft
            metadata.attrs['L_fft'] = self.like.data.L_fft
            metadata['B_A_a'] = self.like.data.B_A_a
            
            
            pxgroup = f.create_group('P_Z_AM')
            covgroup = f.create_group('C_Z_AMN')
            Ugroup = f.create_group('U_Z_aMn')
            Vgroup = f.create_group('V_Z_aM')
            for zbin_ind in range(len(self.like.data.z)):
                print(zbin_ind)
                pxgroup_z = pxgroup.create_group(f'z_{zbin_ind}')
                covgroup_z = covgroup.create_group(f'z_{zbin_ind}')
                Ugroup_z = Ugroup.create_group(f'z_{zbin_ind}')
                Vgroup_z = Vgroup.create_group(f'z_{zbin_ind}')
                for theta_rebin_ind in range(len(self.like.data.theta_min_A_arcmin)):
                    if like_params is None:
                        # use the default parameters from the likelihood
                        noisy_Px = self.generate_px(zbin_ind, theta_rebin_ind, self.like.like_params)
                    else:
                        noisy_Px = self.generate_px(zbin_ind, theta_rebin_ind, like_params)
                    pxgroup_z[f'theta_rebin_{theta_rebin_ind}/'] = noisy_Px
                    covgroup_z[f'theta_rebin_{theta_rebin_ind}/'] = self.like.data.cov_ZAM[zbin_ind,theta_rebin_ind,:,:]
  
                for theta_bin_ind in range(len(self.like.data.theta_min_a_arcmin)):
                    window_matrix = self.like.data.U_ZaMn[zbin_ind, theta_bin_ind]
                    Ugroup_z[f'theta_{theta_bin_ind}/'] = window_matrix
                    Vweights = self.like.data.V_ZaM[zbin_ind, theta_bin_ind]
                    Vgroup_z[f'theta_{theta_bin_ind}/'] = Vweights
        return
    