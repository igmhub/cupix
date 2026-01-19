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


    def generate_px(self, iz_choice, theta_A_ind, like_params=None, add_noise=True):
        # set up the Arinyo P3D model
        # do we want to apply the noise before or after the window convolution?
        # has to be after because I have the covariance matrix for the post-rebinned data
        if like_params is None:
            # use the default parameters from the likelihood
            Px_theory = self.like.get_convolved_Px_AA(iz_choice, theta_A_ind, self.like.like_params)
        else:
            
            Px_theory = self.like.get_convolved_Px_AA(iz_choice, theta_A_ind, like_params)
        # add noise
        if add_noise:
            if type(theta_A_ind) in [int, np.int64, np.int32]:
                cov = self.like.data.cov_ZAM[iz_choice,theta_A_ind,:,:]
                Px_out = self.add_noise(Px_theory, cov)
            elif type(theta_A_ind) in [np.ndarray, list]:
                Px_out = []
                for theta_A_i in theta_A_ind:
                    cov = self.like.data.cov_ZAM[iz_choice,theta_A_i,:,:]
                    Px_theory_i = Px_theory[theta_A_i]
                    Px_out_i = self.add_noise(Px_theory_i, cov)
                    Px_out.append(Px_out_i)
                Px_out = np.array(Px_out)
        else:
            Px_out = Px_theory
        return Px_out
    
    def write_to_file(self, filepath, like_params=None, add_noise=True):
        with h5py.File(filepath,'w') as f:
            metadata = f.create_group('metadata')
            # Reproduce metadata from data file
            metadata.attrs['k_m'] = self.like.data.k_m[0,:]
            metadata.attrs['k_M_edges'] = self.like.data.k_M_edges[0,:]
            metadata.attrs['theta_min_a'] = self.like.data.theta_min_a_arcmin
            metadata.attrs['theta_max_a'] = self.like.data.theta_max_a_arcmin
            metadata.attrs['theta_min_A'] = self.like.data.theta_min_A_arcmin
            metadata.attrs['theta_max_A'] = self.like.data.theta_max_A_arcmin
            metadata.attrs['z_centers'] = self.like.data.z[self.like.iz_choice]
            metadata.attrs['N_fft'] = self.like.data.N_fft
            metadata.attrs['L_fft'] = self.like.data.L_fft
            metadata['B_A_a'] = self.like.data.B_A_a
            
            pxgroup = f.create_group('P_Z_AM')
            covgroup = f.create_group('C_Z_AMN')
            Ugroup = f.create_group('U_Z_aMn')
            Vgroup = f.create_group('V_Z_aM')

            
            if like_params is None:
                Px = self.generate_px(self.like.iz_choice, np.arange(len(self.like.data.theta_min_A_arcmin)), self.like.like_params, add_noise=add_noise)
            else:
                Px = self.generate_px(self.like.iz_choice, np.arange((self.like.data.theta_min_A_arcmin)), like_params, add_noise=add_noise)
            print("checkpoint in generate fake data")
            if Px.ndim == 3:
                Nz = Px.shape[0]
            else:
                Nz = 1
                Px = [Px]
            print(f"Intuited that Nz = {Nz}")
            for zbin_ind in range(Nz):
                print(f"writing to z bin {zbin_ind}")
                pxgroup_z = pxgroup.create_group(f'z_{zbin_ind}')
                covgroup_z = covgroup.create_group(f'z_{zbin_ind}')
                Ugroup_z = Ugroup.create_group(f'z_{zbin_ind}')
                Vgroup_z = Vgroup.create_group(f'z_{zbin_ind}')
                for theta_rebin_ind in range(len(self.like.data.theta_min_A_arcmin)):
                    Px_theta = Px[zbin_ind][theta_rebin_ind]
                    pxgroup_z[f'theta_rebin_{theta_rebin_ind}/'] = Px_theta
                    covgroup_z[f'theta_rebin_{theta_rebin_ind}/'] = self.like.data.cov_ZAM[zbin_ind,theta_rebin_ind,:,:]
                print("Made it past the large theta bin writing in generate_fake_data")
                for theta_bin_ind in range(len(self.like.data.theta_min_a_arcmin)):
                    window_matrix = self.like.data.U_ZaMn[zbin_ind, theta_bin_ind]
                    # window_matrix = np.eye(self.like.data.U_ZaMn[zbin_ind, theta_bin_ind].shape[0])
                    Ugroup_z[f'theta_{theta_bin_ind}/'] = window_matrix
                    Vweights = self.like.data.V_ZaM[zbin_ind, theta_bin_ind]
                    Vgroup_z[f'theta_{theta_bin_ind}/'] = Vweights
                print("Made it past the small theta bin writing in generate_fake_data")
        return
    