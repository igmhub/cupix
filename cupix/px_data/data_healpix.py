import numpy as np
import h5py

from cupix.px_data import read_healpix_px, px_window


class HealpixPxArchive(object):
    '''Collection of PX measurements from healpixels'''

    def __init__(self, fname, list_hp=None, list_px=None, 
                    compute_window=False):
        '''Setup collection, from file or by hand'''

        if fname is not None:
            assert (list_hp is None) and (list_px is None)
            self.fname = fname
            print('read healpix info from', fname)
            # read information from file (Sindhu's format)      
            reader = read_healpix_px.HealpixPxReader(fname)
            # use information to construct list_px
            self.list_hp, self.list_px = reader.get_list_px(compute_window)
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

        # length in Angstroms
        self.L_A = self.list_px[0].L_A
        # mean pixel resolution in Angstroms (could be z dependent eventually)
        self.sig_A = self.list_px[0].sig_A

        for px in self.list_px:
            assert len(self.z_bins) == len(px.z_bins)
            assert len(self.t_bins) == len(px.t_bins)
            assert len(self.k_bins) == len(px.k_bins)
            assert self.L_A == px.L_A
            assert self.sig_A == px.sig_A

        return


    def get_mean_px(self):
        list_px_z = []
        for iz, z_bin in enumerate(self.z_bins):
            list_px_zt = []
            for it, t_bin in enumerate(self.t_bins):
                # this should be properly computed, not hard-coded
                Nk = len(self.k_bins)
                # add the contribution from each healpixel
                F_m = np.zeros(Nk)
                W_m = np.zeros(Nk)
                for px in self.list_px:
                    px_zt = px.list_px_z[iz].list_px_zt[it]
                    F_m += px_zt.F_m
                    W_m += px_zt.W_m
                px_zt = px_window.Px_zt_w.from_unnormalized(
                                z_bin, t_bin, self.k_bins,
                                F_m=F_m, W_m=W_m, 
                                L_A=self.L_A, sig_A=self.sig_A,
                                compute_window=False)
                list_px_zt.append(px_zt)
            px_z = px_window.Px_z_w(self.t_bins, list_px_zt)
            list_px_z.append(px_z)
        mean_px = px_window.Px_w(self.z_bins, list_px_z)
        return mean_px


    def get_mean_and_cov(self, compute_window=False):
        list_px_z = []
        for iz, z_bin in enumerate(self.z_bins):
            list_px_zt = []
            for it, t_bin in enumerate(self.t_bins):
                # collect PX from each healpixel into (N_hp, Nk) array
                Nk = len(self.k_bins)
                N_hp = len(self.list_hp)
                all_P_m = np.empty((N_hp,Nk))
                all_V_m = np.empty((N_hp,Nk))
                for ipix in range(N_hp):
                    px_zt = self.list_px[ipix].list_px_z[iz].list_px_zt[it]
                    all_P_m[ipix] = px_zt.P_m
                    all_V_m[ipix] = px_zt.V_m
                # compute weighted mean and covariance (from Picca)
                mean_P_m = (all_P_m * all_V_m).sum(axis=0)
                sum_V_m = all_V_m.sum(axis=0)
                w = sum_V_m > 0.
                mean_P_m[w] /= sum_V_m[w]
                meanless_P_V_m = all_V_m * (all_P_m - mean_P_m)
                C_mn = meanless_P_V_m.T.dot(meanless_P_V_m)
                sum_V2 = sum_V_m * sum_V_m[:, None]
                w = sum_V2 > 0.
                C_mn[w] /= sum_V2[w]
                # check if we need to compute the mean window matrix
                if compute_window:
                    mean_U_mn = self.get_mean_window_zt(iz, it, all_V_m)
                else:
                    mean_U_mn = None
                # setup object (normalized, with covariance)
                px_zt = px_window.Px_zt_w(z_bin, t_bin, self.k_bins,
                                P_m=mean_P_m, V_m=sum_V_m, C_mn=C_mn,
                                F_m=None, W_m=None, U_mn=mean_U_mn)
                list_px_zt.append(px_zt)
            px_z = px_window.Px_z_w(self.t_bins, list_px_zt)
            list_px_z.append(px_z)
        mean_px = px_window.Px_w(self.z_bins, list_px_z)
        return mean_px


    def get_mean_window_zt(self, iz, it):
        '''Compute mean window matrix, for a given (z,t) bin'''

        # collect PX from each healpixel into (N_hp, Nk) array
        Nk = len(self.k_bins)
        N_hp = len(self.list_hp)
        all_P_m = np.empty((N_hp, Nk))
        all_V_m = np.empty((N_hp, Nk))
        # collect individual window matrices (per healpixel)
        all_U_mn = np.empty((N_hp, Nk, Nk))
        for ipix in range(N_hp):
            px_zt = self.list_px[ipix].list_px_z[iz].list_px_zt[it]
            all_P_m[ipix] = px_zt.P_m
            all_V_m[ipix] = px_zt.V_m
            if px_zt.U_mn is None:
                # compute U_mn for each healpixel
                R2_m = px_window.compute_R2_m(px_zt.k_bins, px_zt.sig_A)
                U_mn = px_window.compute_U_mn(px_zt.W_m, R2_m, px_zt.L_A)
                all_U_mn[ipix] = U_mn
            else:
                all_U_mn[ipix] = px_zt.U_mn
        # compute weighted mean
        mean_P_m = (all_P_m * all_V_m).sum(axis=0)
        sum_V_m = all_V_m.sum(axis=0)
        w = sum_V_m > 0.
        mean_P_m[w] /= sum_V_m[w]
        # weighted sum of all the windows
        sum_U_mn = np.einsum('im,imn->mn', all_V_m, all_U_mn)
        mean_U_mn = sum_U_mn / sum_V_m[:,None]
      
        return sum_V_m, mean_P_m, mean_U_mn


    def rebin(self, rebin_t_factor, rebin_k_factor, include_k_0=True):
        '''Return a new Px archive, after rebinning in theta (t) and k'''

        new_list_px = []
        for px in self.list_px:
            new_px = px.rebin(rebin_t_factor, rebin_k_factor, include_k_0)
            new_list_px.append(new_px)

        new_archive = HealpixPxArchive(fname=None,
                            list_hp=self.list_hp, list_px=new_list_px)

        return new_archive
