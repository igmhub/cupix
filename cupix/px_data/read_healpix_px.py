import numpy as np
import h5py

from cupix.px_data import px_binning, px_ztk, px_window


class HealpixPxReader(object):
    '''Class to read current data model by Sindhu, with healpixels, before rebinning'''

    def __init__(self, fname, verbose=False):
        '''Read PX file and store information'''

        print(f'will read healpixel PX from {fname}')
    
        with h5py.File(fname, 'r') as f:
            if verbose: print(f.keys())

            # Load shared datasets
            self.k_arr = f['k_arr'][:]

            # Load attributes
            self.N_fft = f.attrs['N_fft']
            self.pw_A = f.attrs['pixel_width_A']
            self.L_A = self.N_fft * self.pw_A
            if verbose: print(f'N_fft = {self.N_fft}, pw_A = {self.pw_A}')
            
            group_names = [
                (group_name, float(group_name.split('_')[3]))  # (name, theta_min)
                for group_name in f.keys()
                if group_name.startswith('z_')
            ]

            # Sort by theta_min
            group_names.sort(key=lambda x: x[1])  # sorts by the second item (theta_min)

            self.px = {}
            self.px_weights = {}
            self.theta_bins = []
            self.z_bins = []

            # Now loop over the sorted group names
            for group_name, theta_min in group_names:
                if verbose: print('loop over', group_name)

                parts = group_name.split('_')
                z_bin = float(parts[1])
                theta_max = float(parts[4])
                
                group = f[group_name]
                
                key = (z_bin, theta_min, theta_max)

                self.px[key] = group['px'][:]
                # this is what we would call W_m (not V_m)
                self.px_weights[key] = group['px_weights'][:]

                self.theta_bins.append((theta_min, theta_max))
                self.z_bins.append(z_bin)

        print('finished reading PX file')
        # get first PX to get number of healpixels
        test_px = next(iter(self.px.values()))
        self.N_hp, N_fft = test_px.shape
        assert self.N_fft == N_fft

        return


    def get_z_bins(self):
        '''Get list of redshift bins (Bin_z objects)'''

        zs = np.array(sorted(set(self.z_bins)))
        #print(zs)
        dz = zs[1] - zs[0]
        #print(dz)
        z_bins = []
        for z in zs:
            z_bin = px_binning.Bin_z(min_z=z-0.5*dz,max_z=z+0.5*dz)
            z_bins.append(z_bin)
        return z_bins


    def get_t_bins(self):
        '''Get list of theta bins (Bin_t objects)'''

        theta_min = [theta_bin[0] for theta_bin in self.theta_bins]
        min_t = np.array(sorted(set(theta_min)))
        N_t = len(min_t)
        theta_max = [theta_bin[1] for theta_bin in self.theta_bins]
        max_t = np.array(sorted(set(theta_max)))
        assert N_t == len(max_t)
        t_bins = []
        for it in range(N_t):
            t_bin = px_binning.Bin_t(min_t=min_t[it], max_t=max_t[it])        
            t_bins.append(t_bin)
        return t_bins


    def get_k_bins(self):
        '''Get list of wavenumbers (Discrete_k objects)'''
        k_bins = []
        for k in self.k_arr:
            k_bin = px_binning.Discrete_k(k)
            k_bins.append(k_bin)
        return k_bins


    def get_list_px(self, verbose=False):
        '''Get a list of PX (Px_w objects), one per healpixel'''

        # get list of bins 
        z_bins = self.get_z_bins()
        t_bins = self.get_t_bins()
        k_bins = self.get_k_bins()

        # collect list of healpixels and their px
        list_hp = range(self.N_hp)
        list_px = []
        for pix in list_hp:
            if verbose: print('setting PX for healpixel', pix)
            list_px_z = []
            for z_bin in z_bins:
                if verbose: print('setting PX for z', z_bin.label)
                list_px_zt = []
                for t_bin in t_bins:
                    if verbose: print('setting PX for theta', t_bin.label)
                    # try to find PX for this bin and healpixel
                    mean_z = 0.5*(z_bin.min_z+z_bin.max_z)
                    key = (mean_z, t_bin.min_t, t_bin.max_t)
                    if verbose: print(key)
                    F_m = self.px[key][pix]
                    W_m = self.px_weights[key][pix]
                    R2_m = np.ones_like(self.k_arr)
                    # set NaNs to zero
                    F_m[np.isnan(F_m)] = 0
                    W_m[np.isnan(W_m)] = 0
                    T_m = W_m * R2_m
                    # create Px_zt_w object from these
                    px_zt = px_window.Px_zt_w.from_unnormalized(
                            z_bin, t_bin, k_bins, 
                            F_m=F_m, W_m=W_m, T_m=T_m, L=self.L_A)
                    list_px_zt.append(px_zt)
                # create Px_z object from these
                px_z = px_ztk.Px_z(t_bins, list_px_zt)
                list_px_z.append(px_z)
            # create Px_w object from these
            px = px_window.Px_w(z_bins, list_px_z)
            list_px.append(px)

        return list_hp, list_px

