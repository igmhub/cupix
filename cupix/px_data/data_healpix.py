import numpy as np
import h5py

from cupix.px_data import read_healpix_px, px_ztk, px_window


class HealpixPxArchive(object):
    '''Collection of PX measurements from healpixels'''

    def __init__(self, fname, list_hp=None, list_px=None):
        '''Setup collection, from file or by hand'''

        if fname is not None:
            assert (list_hp is None) and (list_px is None)
            self.fname = fname
            print('read healpix info from', fname)
            # read information from file (Sindhu's format)      
            reader = read_healpix_px.HealpixPxReader(fname)
            # use information to construct list_px
            self.list_hp, self.list_px = reader.get_list_px()
        else:
            self.fname = None
            assert len(list_hp) == len(list_px)
            self.list_hp = list_hp
            self.list_px = list_px

        print(f'Got Px for {len(self.list_hp)} healpixels')

        # get binning information
        self.z_bins = self.list_px[0].z_bins
        self.t_bins = self.list_px[0].t_bins
        self.k_bins = self.list_px[0].k_bins
        for px in self.list_px:
            assert len(self.z_bins) == len(px.z_bins)
            assert len(self.t_bins) == len(px.t_bins)
            assert len(self.k_bins) == len(px.k_bins)

        return


    def get_mean_px(self):
        list_px_z = []
        for iz, z_bin in enumerate(self.z_bins):
            list_px_zt = []
            for it, t_bin in enumerate(self.t_bins):
                # this should be properly computed, not hard-coded
                pw_A = 0.8
                Nk = len(self.k_bins)
                N_fft = Nk
                L_A = pw_A * N_fft
                # add the contribution from each healpixel
                F_m = np.zeros(Nk, dtype='complex')
                W_m = np.zeros(Nk, dtype='complex')
                T_m = np.zeros(Nk, dtype='complex')
                for px in self.list_px:
                    px_zt = px.list_px_z[iz].list_px_zt[it]
                    F_m += px_zt.F_m
                    W_m += px_zt.W_m
                    T_m += px_zt.T_m
                px_zt = px_window.Px_zt_w.from_unnormalized(
                                z_bin, t_bin, self.k_bins,
                                F_m=F_m, W_m=W_m, T_m=T_m, L=L_A)
                list_px_zt.append(px_zt)
            px_z = px_ztk.Px_z(self.t_bins, list_px_zt)
            list_px_z.append(px_z)
        mean_px = px_window.Px_w(self.z_bins, list_px_z)
        return mean_px


