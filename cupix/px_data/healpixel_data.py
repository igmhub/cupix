import numpy as np
import h5py

import px_ztk
import px_binning



class Px_reader(object):
    '''Class to read current data model by Sindhu, with healpixels, before rebinning'''

    def __init__(self, fname):
        '''Read PX file and store informatio'''

        print(f'will read healpixel PX from {fname}')
    
        with h5py.File(fname, 'r') as f:
            print(f.keys())

            # Load shared datasets
            self.k_arr = f['k_arr'][:]

            # Load attributes
            self.N_fft = f.attrs['N_fft']
            self.pw_A = f.attrs['pixel_width_A']
            print(f'N_fft = {N_fft}, pw_A = {pw_A}')
            
            group_names = [
                (group_name, float(group_name.split('_')[3]))  # (name, theta_min)
                for group_name in f.keys()
                if group_name.startswith('z_')
            ]

            # Sort by theta_min
            group_names.sort(key=lambda x: x[1])  # sorts by the second item (theta_min)

            # Now loop over the sorted group names
            for group_name, theta_min in group_names:
                print('loop over', group_name)

                parts = group_name.split('_')
                z_bin = float(parts[1])
                theta_max = float(parts[4])
                
                group = f[group_name]
                
                key = (z_bin, theta_min, theta_max)

                self.px[key] = group['px'][:]
                # is this W_m or V_m? I'm guessing V_m?
                self.px_weights[key] = group['px_weights'][:]
                self.p1d[key[0]] = group['p1d'][:]
                self.num_pairs[key] = group['no_of_pairs'][:]

                self.theta_bins.append((theta_min, theta_max))
                self.z_bins.append(z_bin)


    def get_list_px(self):
        '''Return list of PX objects, one per healpixel'''

        # I need to check with Sindhu how to do this
        healpixels = list_of_healpixels
        list_px = []

        for pix in healpixels:
            print('setting PX for healpixel', pix)
            
            z_bins = []
            list_px_z = []

            for z in self.z_bins:
                print('setting PX for z', z)
                z_bin = px_binning.Bin_z(min_z=z-0.5*dz,max_z=z+0.5*dz)
                z_bins.append(z_bin)

                t_bins = []
                list_px_zt = []

                for (min_t, max_t) in self.theta_bins:
                    print('setting PX for theta', min_t, max_t)
                    t_bin = px_binning.Bin_t(min_t=min_t, max_t=max_t)
                    t_bins.append(t_bin)

                    # try to find PX for this bin and healpixel
                    key = (z, min_t, max_t)
                    P_m = self.px[key][pix]
                    V_m = self.px_weights[key][pix]

                    # define k bins (discrete values here)
                    k_bins = []
                    for k in self.k_arr:
                        print('setting PX for k', k)
                        k_bin = px_binning.Discrete_k(k)
                        k_bins.append(k_bin)

                    # create Px_zt object from these
                    px_zt = px_ztk.Px_zt(z_bin, t_bin, k_bins, px_ztk=P_m, V_ztk=V_m)
                    list_px_zt.append(px_zt)

                # create Px_z object from these
                px_z = px_ztk.Px_z(t_bins, list_px_ztk)
                list_px_z.append(px_z)

            # create BasePx object from these
            px = px_ztk.BaseDataPx(z_bins, list_px_z)
            list_px.append(px)
                
        return healpixels, list_px
