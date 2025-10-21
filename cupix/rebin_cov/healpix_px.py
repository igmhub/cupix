import h5py
from collections import defaultdict
import numpy as np

# make a class for the output data
class Px_meas(object):
    """ Class to store the Px measurements """
    def __init__(self,path):
        #self.p1d = None
        self.k_arr = None
        self.N_fft = None
        self.N_skewers = None
        self.L_fft = None
        self.k_fund = None
        self.k_Nyq = None
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
         
        with h5py.File(path, 'r') as f:
            #print(f.keys())
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
        
        self.z_bin_centers = np.unique(self.z_bins)
        self.theta_bin_min = np.unique(np.array([tb[0] for tb in self.theta_bins]))
        self.theta_bin_max = np.unique(np.array([tb[1] for tb in self.theta_bins]))
        
        # minimum and maximum frequencies 
        self.L_fft=self.N_fft*self.pw_A
        self.k_fund=2*np.pi/self.L_fft        
        self.k_Nyq = np.pi/self.pw_A
        return None
            
    def unpack_healpix(self):
        """
        Pack dicts into dense arrays for einsum ops.
        Returns: P, W, k_out with shapes (Nz, Nθ, Nhp, Nk_out)
        """
        # Shapes
        Nhp, N_k = next(iter(self.px.values())).shape
        Nz = len(self.z_bin_centers)
        Nt = len(self.theta_bin_min)

        P = np.zeros((Nz, Nt, Nhp, N_k), dtype=float)
        W = np.zeros_like(P)
        theta_pairs = list(zip(self.theta_bin_min, self.theta_bin_max))

        # Fill
        for i, z in enumerate(self.z_bin_centers):
            for j, (tmin, tmax) in enumerate(theta_pairs):
                key = (z, tmin, tmax)
                arr = self.px[key]
                wgt = self.px_weights[key] 
                if arr is None:
                    print('No Px measurements found')
                    continue
                if wgt is None:
                    print('No weights found')
                    wgt = np.ones_like(arr)
                P[i,j] = arr
                W[i,j] = wgt
                
        # replace NaNs
        P = np.nan_to_num(P,nan=0.0,posinf=0.0,neginf=0.0)
        W = np.nan_to_num(W,nan=0.0,posinf=0.0,neginf=0.0)

        k_out = np.array(self.k_arr, copy=False)

        # Keep only positive frequencies (first half)
        if positive_frequencies:
            half = N_k // 2
            P = P[..., :half]
            #W = W[..., :half]
            k_out = k_out[:half]

        return P, W, k_out


