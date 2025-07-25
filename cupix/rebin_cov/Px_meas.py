import h5py
from cupix.rebin_cov.rebin_cov import calculate_estnorm,bin_power,compute_cov
from collections import defaultdict
import numpy as np


# make a class for the output data
class Px_meas:
    """ Class to store the Px measurements """
    def __init__(self):
        #self.p1d = None
        self.k_arr = None
        self.N_fft = None
        self.N_skewers = None
        self.px = {}
        self.px_weights = {}
        self.theta_bins = []
        self.z_bins = []
        self.num_pairs = {}
        self.avg_weights = {}
        self.p1d = {}
        self.p1d_avg = {}
        self.px_avg = {}
        self.covariance = {}
        self.px_avg_bin = {}
        self.covariance_bin={}
        self.k_bins = {}
        self.px_theta_binned = {}
        self.vm_theta_binned = {}
        return

    def read_hdf5(self, path, nmodes=None):
        with h5py.File(path, 'r') as f:
            print(f.keys())
            # Load shared datasets
            self.k_arr = f['k_arr'][:]
            # self.p1d_weights = f['p1d_weights'][:]

            # Load attributes
            self.N_fft = f.attrs['N_fft']
            self.pw_A = f.attrs['pixel_width_A']
            
            group_names = [
            (group_name, float(group_name.split('_')[3]))  # (name, theta_min)
            for group_name in f.keys()
            if group_name.startswith('z_')
            ]

            # Sort by theta_min
            group_names.sort(key=lambda x: x[1])  # sorts by the second item (theta_min)

            # Now loop over the sorted group names
            for group_name, theta_min in group_names:
                parts = group_name.split('_')
                z_bin = float(parts[1])
                theta_max = float(parts[4])
                
                group = f[group_name]
                
                key = (z_bin, theta_min, theta_max)

                self.px[key] = group['px'][:]
                self.px_weights[key] = group['px_weights'][:]
                self.p1d[key[0]] = group['p1d'][:]
                self.num_pairs[key] = group['no_of_pairs'][:]

                self.theta_bins.append((theta_min, theta_max))
                self.z_bins.append(z_bin)

    def compute_stat(self):
        """Computes the covariance matrix for the Px measurements
    
        Args:
            px_mocks: Px_meas object
                Contains the Px measurements and their weights
    
        Returns:
            px_avg_weights: dictionary with average weights per healpix
            p1d_avg: dictionary with average P1D measurements
            px_avg: dictionary with average Px measurements
            covariance: dictionary with covariance matrices for each key
        """

        for key in self.px.keys():
                p1d_all = self.p1d
                px_all = self.px
                px_weights_all = self.px_weights
                pw_A = self.pw_A
                N_fft = self.N_fft
                no_of_pairs = self.num_pairs
    
                # stack by healpix
                
                stacked_px = np.stack(px_all[key])  # shape (nhp, Nk)
                
                stacked_weights = np.stack(px_weights_all[key])
    
                #print('shape of stacked_px:', np.shape(stacked_px))
                #print('shape of stacked_weights:', np.shape(stacked_weights))
                
                fft_avg_res = np.ones(N_fft)
                L = N_fft * pw_A  # length of the spectra in Angstroms
                stacked_V_m = np.stack([calculate_estnorm(w,fft_avg_res , L) for w in stacked_weights])        
                
                #print('shape of stacked_V_m:', np.shape(stacked_V_m))
                
                stacked_px_hat = np.zeros_like(stacked_px)
                ind = stacked_V_m > 0.0
                stacked_px_hat[ind] = stacked_px[ind] / stacked_V_m[ind] # normalised Px = F_m/V_m
                
                valid_rows = ind[:,0]
                
                assert stacked_px_hat.shape[0] == len(valid_rows)
                
                #print(np.where(valid_rows))
                #print(np.shape(np.where(valid_rows)))
                
                valid_hp = np.asarray(np.where(valid_rows)[0])  
                #print(valid_hp)
    
                stacked_px_hat_valid = np.empty((len(valid_hp),N_fft))
                stacked_V_m_valid = np.empty((len(valid_hp),N_fft))
            
                for l in range(len(valid_hp)):
                    stacked_px_hat_valid[l] = stacked_px_hat[valid_hp[l]]
                    stacked_V_m_valid[l] = stacked_V_m[valid_hp[l]]
   
                self.p1d_avg[key[0]] = np.nanmean(np.stack(p1d_all[key[0]]), axis=0)
    
                mean_px , self.covariance[key] = compute_cov(stacked_px_hat_valid, stacked_V_m_valid)  # covariance matrix of Px arrays
                self.px_avg[key] = mean_px
  
    def compute_thetabinned_px(self,theta_min_rebin,theta_max_rebin):
        """Computes the theta-binned mean and covariance matrix for the Px measurements

        Args:
            px_mocks: Px_meas object
                Contains the Px measurements and their weights    
       
        Adds the following instances to Px_meas object:
            px_theta_binned: dictionary with theta-binned px F_m_A in each healpix
            vm_theta_binned: dictionary with theta-binned weights W_m_A in each healpix 
        
        """
        all_px = {}
        all_vm = {}
        key = self.px.keys()
        zbin_to_keys = defaultdict(list)
        for keys in self.px.keys():
            zbin = keys[0]  # Extract the z_bin  from the key tuple
            zbin_to_keys[zbin].append(keys)

        print('Proceeding to bin in theta...')    
        
        for key in self.px.keys():
            p1d_all = self.p1d
            px_all = self.px
            px_weights_all = self.px_weights
            pw_A = self.pw_A
            N_fft = self.N_fft
            
            # stack by healpix
            stacked_px = np.stack(px_all[key])
            stacked_weights = np.stack(px_weights_all[key])
        
            #print(np.shape(stacked_px))
            #print(stacked_px.shape[0])
            nhp = stacked_px.shape[0]
            

            fft_avg_res = np.ones(N_fft)
            L = N_fft * pw_A  # length of the spectra in Angstroms
            stacked_V_m = np.stack([calculate_estnorm(w,fft_avg_res , L) for w in stacked_weights])        
            
            #print('shape of stacked_V_m:', np.shape(stacked_V_m))
            stacked_px_hat = np.zeros_like(stacked_px)
            
            #all_px_hat[key] = stacked_px/stacked_V_m
            
            all_px[key] = stacked_px  #F_m
            all_vm[key] = stacked_V_m

            z_bins = sorted(set([key[0] for key in all_px.keys()]))

            for z_bin in z_bins:
                for i in range(len(theta_min_rebin)):
                    theta_lo = theta_min_rebin[i]
                    theta_hi = theta_max_rebin[i]
                    rebinned_cen = 0.5 * (theta_lo + theta_hi)
        
                    # Find all original theta bins in this rebin range
                
                    matching_keys = [
                        key for key in all_px.keys()
                        if key[0] == z_bin and not (key[2] <= theta_lo or key[1] >= theta_hi)
                    ]
        
                    #print(matching_keys)
        
                     # Collect the (nhp, Nk) arrays
                    px_arrays = [all_px[key] for key in matching_keys if key in all_px]
                    vm_arrays = [all_vm[key] for key in matching_keys if key in all_vm]
                    
                    if not px_arrays:
                        nhp = all_px[zbin_to_keys[z_bin][0]].shape[0]
                        
                        #print(nhp)
                        self.px_theta_binned[(z_bin, theta_lo,theta_hi)] = np.full((nhp, N_fft), np.nan)
                        self.vm_theta_binned[(z_bin, theta_lo, theta_hi)] = np.full((nhp, N_fft), 0.0)
                        continue
            
                    # Stack and average over the theta dimension
                    stacked = np.stack(px_arrays)  # shape: (n_theta, nhp, Nk)
                    stacked_vm = np.stack(vm_arrays)
        
                    px_bin = np.nanmean(stacked, axis=0)  # shape: (nhp, Nk) 
                    vm_bin = np.nanmean(stacked_vm,axis=0)
        
                    self.px_theta_binned[(z_bin, theta_lo,theta_hi)] = px_bin # theta-binned F_m
                    self.vm_theta_binned[(z_bin,theta_lo,theta_hi)] = vm_bin  #theta-binned V_m
                    
        print('Done')            
    
    def compute_binned_cov(self,bin_info,theta_binning):
         """Computes the k- and theta-binned mean and covariance matrix for the Px measurements

        Args:
            px_mocks: Px_meas object
                Contains the Px measurements and their weights    
       
        Adds the following instances to Px_meas object:
            k_bins: dictionary with the coarse k bins 
            px_avg_bin: dictionary with k- and theta-binned px averaged over all healpix pixels <F_M_A>
            covariance_bin: dictionary with k- and theta-binned normalised weights averaged over all healpix pixels <V_M_A>
            
        """
        all_px = {}
        all_vm = {}
            
        Nk =  int(bin_info['Nk'])
 
        print('Binning in k and computing stats..')
        
        if theta_binning == False:
            for key in self.px.keys():
                p1d_all = self.p1d
                px_all = self.px
                px_weights_all = self.px_weights
                pw_A = self.pw_A
                N_fft = self.N_fft
            
                # stack by healpix
                stacked_px = np.stack(px_all[key])
                stacked_weights = np.stack(px_weights_all[key])
            
                #print(np.shape(stacked_px))
                print(stacked_px.shape[0])
                nhp = stacked_px.shape[0]
                
           
                fft_avg_res = np.ones(N_fft)
                L = N_fft * pw_A  # length of the spectra in Angstroms
                stacked_V_m = np.stack([calculate_estnorm(w,fft_avg_res , L) for w in stacked_weights])        
                
                #print('shape of stacked_V_m:', np.shape(stacked_V_m))
                stacked_px_hat = np.zeros_like(stacked_px)
                
                #all_px_hat[key] = stacked_px/stacked_V_m
                
                all_px[key] = stacked_px  #F_m
                all_vm[key] = stacked_V_m

                ind = stacked_V_m > 0.0
                stacked_px_hat[ind] = stacked_px[ind] / stacked_V_m[ind]        
                valid_rows = ind[:,0]
            
                assert stacked_px_hat.shape[0] == len(valid_rows)
            
         
                valid_hp = np.asarray(np.where(valid_rows)[0])
            
                stacked_px_hat_binned = np.empty((len(valid_hp),Nk))
                stacked_V_m_binned = np.empty((len(valid_hp),Nk))
        
                for l in range(len(valid_hp)):
                    px_k = stacked_px_hat[valid_hp[l]]
                    w_k = stacked_V_m[valid_hp[l]]
                    k_bins, stacked_px_hat_binned[l] = bin_power(self.k_arr,px_k,bin_info) #binned F_m/V_m
                    k_bins, stacked_V_m_binned[l] = bin_power(self.k_arr,w_k,bin_info)

                self.k_bins[key] = k_bins
                self.px_avg_bin[key] , self.covariance_bin[key] = compute_cov(stacked_px_hat_binned, stacked_V_m_binned)  # covariance matrix of Px arrays
                             
        elif theta_binning == True:
            for key in self.px_theta_binned.keys():
                px_all = self.px_theta_binned
                px_weights_all = self.vm_theta_binned
                
                stacked_px = np.stack(px_all[key])
                stacked_V_m = np.stack(px_weights_all[key])

                stacked_px_hat = np.zeros_like(stacked_px)

                
                all_px[key] = stacked_px  #F_m
                all_vm[key] = stacked_V_m

                ind = stacked_V_m > 0.0
                stacked_px_hat[ind] = stacked_px[ind] / stacked_V_m[ind]        
                
            
                valid_rows = ind[:,0]
                
                assert stacked_px_hat.shape[0] == len(valid_rows)
                
                valid_hp = np.asarray(np.where(valid_rows)[0])
                #print(valid_hp)
                
                stacked_px_hat_binned = np.empty((len(valid_hp),Nk))
                stacked_V_m_binned = np.empty((len(valid_hp),Nk))
            
                for l in range(len(valid_hp)):
                    px_k = stacked_px_hat[valid_hp[l]]
                    w_k = stacked_V_m[valid_hp[l]]
                    k_bins, stacked_px_hat_binned[l] = bin_power(self.k_arr,px_k,bin_info)
                    k_bins, stacked_V_m_binned[l] = bin_power(self.k_arr,w_k,bin_info)
     
                self.k_bins = k_bins
                self.px_avg_bin[key] , self.covariance_bin[key] = compute_cov(stacked_px_hat_binned, stacked_V_m_binned)  # covariance matrix of Px arrays            
        print('Done')
        return    




