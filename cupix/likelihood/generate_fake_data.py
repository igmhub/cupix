# generate fake data from Arinyo model, for testing sensitivity of the data to parameters

import numpy as np
import h5py
import copy
from cupix.likelihood.likelihood_parameter import likeparam_from_dict, LikelihoodParameter, dict_from_likeparam, format_like_params_dict
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


    def generate_px(self, theta_A_ind, like_params=None, add_noise=True):
        
        # set up the Arinyo P3D model
        # do we want to apply the noise before or after the window convolution?
        # has to be after because I have the covariance matrix for the post-rebinned data
        Px_theory, arinyo_coeffs = self.like.get_convolved_Px_AA(theta_A_ind, like_params, return_arinyo_coeffs=True)
        print(arinyo_coeffs)
        # add noise
        if add_noise:
            if type(theta_A_ind) in [int, np.int64, np.int32]:
                cov = self.like.data.cov_ZAM[self.like.data_iz,theta_A_ind,:,:]
                Px_out = self.add_noise(Px_theory, cov)
            elif type(theta_A_ind) in [np.ndarray, list]:
                Px_out = []
                for theta_A_i in theta_A_ind:
                    cov = self.like.data.cov_ZAM[self.like.data_iz,theta_A_i,:,:]
                    Px_theory_i = Px_theory[theta_A_i]
                    Px_out_i = self.add_noise(Px_theory_i, cov)
                    Px_out.append(Px_out_i)
                Px_out = np.array(Px_out)
        else:
            Px_out = Px_theory
        return Px_out, arinyo_coeffs
    
    def write_to_file(self, filepath, like_params=None, add_noise=True):
        with h5py.File(filepath,'w') as f:
            metadata = f.create_group('metadata')
            # Reproduce metadata from data file
            metadata.attrs['k_m'] = self.like.data.k_m[self.like.data_iz,:]
            metadata.attrs['k_M_edges'] = self.like.data.k_M_edges[self.like.data_iz,:]
            metadata.attrs['theta_min_a'] = self.like.data.theta_min_a_arcmin
            metadata.attrs['theta_max_a'] = self.like.data.theta_max_a_arcmin
            metadata.attrs['theta_min_A'] = self.like.data.theta_min_A_arcmin
            metadata.attrs['theta_max_A'] = self.like.data.theta_max_A_arcmin
            metadata.attrs['z_centers'] = [self.like.z]
            metadata.attrs['N_fft'] = self.like.data.N_fft
            metadata.attrs['L_fft'] = self.like.data.L_fft
            metadata['B_A_a'] = self.like.data.B_A_a
            metadata.attrs['true_lya_theory'] = self.like.theory.default_lya_theory
            pxgroup = f.create_group('P_Z_AM')
            covgroup = f.create_group('C_Z_AMN')
            Ugroup = f.create_group('U_Z_aMn')
            Vgroup = f.create_group('V_Z_aM')
            params_group = f.create_group('like_params')
            arinyo_group = f.create_group('arinyo_pars')
            cosmo = f.create_group('cosmo_params')
            print("Here")
            # no matter the inputs, make sure like_params_obj is a list of LikelihoodParameter objects, and like_params is a dictionary of parameter values for easier handling below
            print(self.like.data_iz)
            like_params = format_like_params_dict(self.like.data_iz, like_params)
            print("After")   
            print(like_params)
            print("arguments going in", np.arange(len(self.like.data.theta_min_A_arcmin)), like_params, add_noise)
            Px, arinyo = self.generate_px(np.arange(len(self.like.data.theta_min_A_arcmin)), like_params, add_noise=add_noise)
            print(f"writing to z bin {self.like.data_iz}")
            pxgroup_z = pxgroup.create_group(f'z_{self.like.data_iz}')
            covgroup_z = covgroup.create_group(f'z_{self.like.data_iz}')
            Ugroup_z = Ugroup.create_group(f'z_{self.like.data_iz}')
            Vgroup_z = Vgroup.create_group(f'z_{self.like.data_iz}')
            for theta_rebin_ind in range(len(self.like.data.theta_min_A_arcmin)):
                
                Px_theta = Px[theta_rebin_ind]
                pxgroup_z[f'theta_rebin_{theta_rebin_ind}/'] = np.squeeze(Px_theta)
                covgroup_z[f'theta_rebin_{theta_rebin_ind}/'] = np.squeeze(self.like.data.cov_ZAM[self.like.data_iz,theta_rebin_ind,:,:])
            print("Made it past the large theta bin writing in generate_fake_data")
            for theta_bin_ind in range(len(self.like.data.theta_min_a_arcmin)):
                window_matrix = self.like.data.U_ZaMn[self.like.data_iz, theta_bin_ind]
                # window_matrix = np.eye(self.like.data.U_ZaMn[self.like.data_iz, theta_bin_ind].shape[0])
                Ugroup_z[f'theta_{theta_bin_ind}/'] = np.squeeze(window_matrix)
                Vweights = self.like.data.V_ZaM[self.like.data_iz, theta_bin_ind]
                Vgroup_z[f'theta_{theta_bin_ind}/'] = np.squeeze(Vweights)
            print("Made it past the small theta bin writing in generate_fake_data")
            # get input parameters. First, get all the defaults
            theory_inputs = copy.deepcopy(self.like.theory.default_param_dict.copy())
            # replace with any user-provided values
            if like_params:
                for par in like_params:
                    theory_inputs[par] = like_params[par]
            for par in theory_inputs:
                print(par, theory_inputs[par])
                params_group.attrs[par] = theory_inputs[par]
            for par in self.like.theory.cosmo_dict:
                print(par, self.like.theory.cosmo_dict[par])
                cosmo.attrs[par] = self.like.theory.cosmo_dict[par]
            for par in arinyo:
                arinyo_group.attrs[par+f'_{self.like.theory_iz}'] = arinyo[par][self.like.theory_iz]
        return
    