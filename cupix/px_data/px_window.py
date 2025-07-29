import numpy as np
from cupix.px_data import px_binning, px_ztk

class Px_w(px_ztk.BasePx):
    '''Derived BasePx object, with information related to window matrix.'''

    def __init__(self, z_bins, list_px_z):
        super().__init__(z_bins, list_px_z)

        return


    def rebin_t(self, rebin_factor):
        '''Return a new Px_w object, rebinned in theta'''

        # for each z bin, rebin in theta
        new_list_px_z = []
        for px_z in self.list_px_z:
            new_px_z = px_z.rebin_t(rebin_factor)
            new_list_px_z.append(new_px_z)

        new_px = Px_w(self.z_bins, new_list_px_z)

        return new_px


    def rebin_k(self, rebin_factor, include_k_0=True):
        '''Return a new Px_w object, rebinned in k'''

        # for each z bin, rebin in k
        new_list_px_z = []
        for px_z in self.list_px_z:
            new_px_z = px_z.rebin_k(rebin_factor, include_k_0)
            new_list_px_z.append(new_px_z)

        new_px = Px_w(self.z_bins, new_list_px_z)

        return new_px


    def rebin(self, rebin_t_factor, rebin_k_factor, include_k_0=True):
        '''Return a new Px_w object, rebinned in theta and k'''

        # for each z bin, rebin in theta
        new_list_px_z = []
        for px_z in self.list_px_z:
            new_px_z = px_z.rebin(rebin_t_factor, rebin_k_factor, include_k_0)
            new_list_px_z.append(new_px_z)

        new_px = Px_w(self.z_bins, new_list_px_z)

        return new_px



class Px_z_w(px_ztk.Px_z):
    '''Derived Px_z object, with information related to window matrix.'''

    def __init__(self, t_bins, list_px_zt):
        super().__init__(t_bins, list_px_zt)

        return


    def rebin_t(self, rebin_factor):
        '''Return a new Px_z_w object, rebinned in theta'''

        new_t_bins = px_binning.get_coarser_t_bins(self.t_bins, rebin_factor) 
        new_list_px_zt = []
        for new_t_bin in new_t_bins:
            #print('new_t', new_t_bin.min_t, new_t_bin.max_t)
            new_F_am = np.zeros(len(self.k_bins))
            new_V_am = np.zeros_like(new_F_am)
            # should rebin window matrix U_mn here as well
            for in_t_bin, in_px_zt in zip(self.t_bins, self.list_px_zt):
                t_a = in_t_bin.mean()
                B_a = new_t_bin.B_t(t_a)
                if B_a > 0:
                    #print(t_a, B_a)
                    new_F_am += B_a * in_px_zt.F_m
                    new_V_am += B_a * in_px_zt.V_m

            # normalize Px (for bins measured)
            mask = new_V_am>0
            new_P_am = np.zeros_like(new_F_am)
            new_P_am[mask] = new_F_am[mask] / new_V_am[mask]
            new_px_zt = Px_zt_w.from_rebinning(self.z_bin, new_t_bin, self.k_bins,
                                            P_m=new_P_am, V_m=new_V_am, U_mn=None)
            new_list_px_zt.append(new_px_zt)

        new_px_z = Px_z_w(new_t_bins, new_list_px_zt)

        return new_px_z


    def rebin_k(self, rebin_factor, include_k_0=True):
        '''Return a new Px_z_w object, rebinned in k'''

        # for each theta bin, rebin in k
        new_list_px_zt = []
        for px_zt in self.list_px_zt:
            new_px_zt = px_zt.rebin_k(rebin_factor, include_k_0)
            new_list_px_zt.append(new_px_zt)

        new_px_z = Px_z_w(self.t_bins, new_list_px_zt)

        return new_px_z


    def rebin(self, rebin_t_factor, rebin_k_factor, include_k_0=True):
        '''Return a new Px_w_z object, rebinned in theta and k'''

        # get first a new object, rebinned in k
        new_px_z = self.rebin_k(rebin_k_factor, include_k_0)

        # ask the new object to rebin in theta
        return new_px_z.rebin_t(rebin_t_factor)



class Px_zt_w(px_ztk.Px_zt):
    '''Derived Px_zt object, with information related to window matrix'''

    def __init__(self, z_bin, t_bin, k_bins, P_m, V_m, F_m, W_m, T_m, U_mn, C_mn=None):
        '''Provide extra information related to window / weights'''

        super().__init__(
                z_bin=z_bin, 
                t_bin=t_bin, 
                k_bins=k_bins,
                P_m=P_m,
                V_m=V_m,
                C_mn=C_mn)

        # sum of the product of FFT of delta * weights
        self.F_m = F_m
        # sum of the product of FFT of weights
        self.W_m = W_m
        # sum of the product of FFT of resolution * weights
        self.T_m = T_m
        # window matrix
        self.U_mn = U_mn

        return


    @classmethod
    def from_unnormalized(cls, z_bin, t_bin, k_bins, F_m, W_m, T_m, L):
        '''Construct object from unnormalized quantities'''

        P_m, V_m = normalize_Px(F_m, W_m, T_m, L)
        C_mn = None
        # should compute the window matrix here
        U_mn = None

        return Px_zt_w(z_bin, t_bin, k_bins, P_m, V_m, F_m, W_m, T_m, U_mn, C_mn)


    @classmethod
    def from_rebinning(cls, z_bin, t_bin, k_bins, P_m, V_m, U_mn):
        '''Construct object from rebinning thinner bins'''

        # these might be used in further rebinning
        F_m = P_m * V_m
        # for now, these are not needed
        W_m = None
        T_m = None
        C_mn = None

        return Px_zt_w(z_bin, t_bin, k_bins, P_m, V_m, F_m, W_m, T_m, U_mn, C_mn)


    def rebin_k(self, rebin_factor, include_k_0=True):
        '''Return a new Px_zt_w object, rebinned in k'''

        in_k_bins = self.k_bins
        in_Nk = len(in_k_bins)
        new_k_bins = px_binning.get_coarser_k_bins(in_k_bins, rebin_factor, include_k_0)
        new_Nk = len(new_k_bins)
        new_F_m = np.zeros(new_Nk)
        new_V_m = np.zeros(new_Nk)
        # should rebin window matrix U_mn here as well
        for new_ik in range(new_Nk):
            new_k_bin = new_k_bins[new_ik]
            #print('new_ik', new_ik, new_k_bin.min_k, new_k_bin.max_k)
            for in_ik in range(in_Nk):
                in_k = in_k_bins[in_ik].k
                B_m = new_k_bin.B_k(in_k)
                if B_m > 0:
                    #print('in_ik', in_ik, in_k, B_m)
                    new_F_m[new_ik] += B_m * self.F_m[in_ik]
                    new_V_m[new_ik] += B_m * self.V_m[in_ik]

        # normalize Px (for bins measured)
        mask = new_V_m>0
        new_P_m = np.zeros_like(new_F_m)
        new_P_m[mask] = new_F_m[mask] / new_V_m[mask]
        new_px_tz = Px_zt_w.from_rebinning(self.z_bin, self.t_bin, new_k_bins,
                                            P_m=new_P_m, V_m=new_V_m, U_mn=None)
        return new_px_tz



def normalize_Px(F_m, W_m, T_m, L):
    '''Compute P_m and V_m given unnormalized measurements'''

    # compute normalization factors, used 
    V_m = compute_V_m(W_m, T_m, L)
    P_m = np.zeros_like(V_m)
    P_m[V_m>0] = (F_m[V_m>0] / V_m[V_m>0]).real

    return P_m, V_m


def compute_V_m(W_m, T_m, L):
    '''Compute normalization factor for Px'''

    # effective resolution kernel (squared)
    R2_m = np.zeros_like(W_m)
    R2_m[W_m>0] = T_m[W_m>0] / W_m[W_m>0]
 
    # convolve W and R2 arrays 
    W_R2 = np.fft.ifft(np.fft.fft(W_m)* np.fft.fft(R2_m))

    return np.abs(W_R2) / L
