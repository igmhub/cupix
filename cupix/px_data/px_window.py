import numpy as np
from numba import njit

from cupix.px_data import px_binning, px_ztk


class Px_w(px_ztk.BasePx):
    '''Derived BasePx object, with information related to window matrix.'''

    def __init__(self, z_bins, list_px_z):
        super().__init__(z_bins, list_px_z)

        # check consistency of FFT length and pixel resolution
        self.L_A = list_px_z[0].L_A
        self.sig_A = list_px_z[0].sig_A
        for px_z in list_px_z:
            assert self.L_A == px_z.L_A
            assert self.sig_A == px_z.sig_A

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

        # check consistency of FFT length and pixel resolution
        self.L_A = list_px_zt[0].L_A
        self.sig_A = list_px_zt[0].sig_A
        for px_zt in list_px_zt:
            assert self.L_A == px_zt.L_A
            assert self.sig_A == px_zt.sig_A

        return


    def rebin_t(self, rebin_factor):
        '''Return a new Px_z_w object, rebinned in theta'''

        new_t_bins = px_binning.get_coarser_t_bins(self.t_bins, rebin_factor) 
        new_list_px_zt = []
        for new_t_bin in new_t_bins:
            #print('new_t', new_t_bin.min_t, new_t_bin.max_t)
            new_F_am = np.zeros(len(self.k_bins))
            new_V_am = np.zeros_like(new_F_am)
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


    def rebin_model(self, raw_model, raw_px_z, convolve=True):
        '''Rebin model, and convolve with window matrix'''

        # theoretical prediction, in raw bins (2D array)
        raw_Nt, raw_Nk = raw_model.shape
        print(f'input raw model, shape = {raw_Nt}, {raw_Nk}')
        assert raw_Nt == len(raw_px_z.t_bins), 'size mismatch'
        assert raw_Nk == len(raw_px_z.k_bins), 'size mismatch'

        # get window matrix (and weights) for each (original) theta bin
        list_V_m = [ px_zt.V_m for px_zt in raw_px_z.list_px_zt]
        list_U_mn = [ px_zt.U_mn for px_zt in raw_px_z.list_px_zt]
        print(f'got {len(list_U_mn)} window matrices')

        # for each (original) theta bin, construct rectangular window
        rb_Nk = len(self.k_bins) 
        print(f'will rebin windows to shape {rb_Nk} x {raw_Nk}')

        # for each rebinned k bin, figure out contributions
        B_M_m = []
        for iM in range(rb_Nk):
            k_bin = self.k_bins[iM]
            B_m = k_bin.B_k_bins(raw_px_z.k_bins)
            print(iM, np.nonzero(B_m)[0])
            B_M_m.append(B_m)

        # compute rectangular window matrix (one per original theta bin)
        list_U_Mn = []
        list_V_M = []
        list_P_M = []
        for raw_it in range(raw_Nt):
            V_m = list_V_m[raw_it]
            U_mn = list_U_mn[raw_it]  
            # V_M = sum_m (V_m B_M_m) 
            V_M = np.zeros(rb_Nk)
            # U_Mn = sum_m (V_m B_M_m U_mn) / V_M
            U_Mn = np.zeros([rb_Nk, raw_Nk])
            # P_M = sum_n U_Mn P_n
            P_M = np.zeros(rb_Nk)
            list_P_M.append(P_M)
            list_V_M.append(V_M)
            list_U_Mn.append(U_Mn)


        # rebin model into coarser theta bins
        rb_Nt = len(self.t_bins) 
        B_A_a = []
        for iA in range(rb_Nt):
            t_bin = self.t_bins[iA]
            B_a = t_bin.B_t_bins(raw_px_z.t_bins)
            print(iA, np.nonzero(B_a)[0])
            B_A_a.append(B_a)


        # convolve here with the window, rebin, etc
        rb_model = np.zeros([len(self.t_bins), len(self.k_bins)])

        return rb_model



class Px_zt_w(px_ztk.Px_zt):
    '''Derived Px_zt object, with information related to window matrix'''

    def __init__(self, z_bin, t_bin, k_bins, P_m, V_m, F_m, W_m, U_mn, 
                    C_mn=None, L_A=None, sig_A=None):
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
        # window matrix
        self.U_mn = U_mn
        # length of FFT grid (in Angstroms)
        self.L_A = L_A
        # mean pixel resolution (in Angstroms)
        self.sig_A = sig_A

        return


    @classmethod
    def from_unnormalized(cls, z_bin, t_bin, k_bins, 
                F_m, W_m, L_A, sig_A, compute_window=False):
        '''Construct object from unnormalized quantities'''

        R2_m = compute_R2_m(k_bins, sig_A)
        P_m, V_m = normalize_Px(F_m, W_m, R2_m, L_A)
        C_mn = None
        if compute_window:
            R2_m = compute_R2_m(k_bins, sig_A)
            U_mn = compute_U_mn(W_m, R2_m, L_A)
        else:
            U_mn = None

        return cls(z_bin, t_bin, k_bins, P_m, V_m, F_m, W_m, U_mn, C_mn, L_A, sig_A)


    @classmethod
    def from_rebinning(cls, z_bin, t_bin, k_bins, P_m, V_m, U_mn):
        '''Construct object from rebinning thinner bins'''

        # these might be used in further rebinning
        F_m = P_m * V_m
        # for now, these are not needed
        W_m = None
        C_mn = None
        L_A = None
        sig_A = None

        return cls(z_bin, t_bin, k_bins, P_m, V_m, F_m, W_m, U_mn, C_mn, L_A, sig_A)


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



def normalize_Px(F_m, W_m, R2_m, L_A):
    '''Compute P_m and V_m given unnormalized measurements'''
    # compute normalization factors, used 
    V_m = compute_V_m(W_m, R2_m, L_A)
    P_m = np.zeros_like(V_m)
    P_m[V_m>0] = (F_m[V_m>0] / V_m[V_m>0]).real
    return P_m, V_m


@njit
def compute_U_mn_fast(W_m, R2_m, V_m_L):
    '''numba function to speed up the matrix operation'''
    Nk = len(W_m)
    U_mn = np.zeros((Nk, Nk))
    for m in range(Nk):
        U_mn[m, m] = W_m[0] * R2_m[m] / V_m_L[m]
        for n in range(m):
            diff = m - n
            U_mn[m, n] = W_m[diff] * R2_m[n] / V_m_L[m]
        for n in range(m+1, Nk):
            diff = Nk + m - n
            U_mn[m, n] = W_m[diff] * R2_m[n] / V_m_L[m]
    return U_mn


def compute_R2_m(k_bins, sig_A):
    '''Resolution kernel'''
    k_m = np.array( [ k_bin.k for k_bin in k_bins ] )
    R2_m = np.exp(-(k_m * sig_A)**2)   
    return R2_m


def compute_U_mn(W_m, R2_m, L_A):
    '''Compute window matrix'''
    # normalization
    V_m = compute_V_m(W_m, R2_m, L_A)
    Nk = len(V_m)
    if np.all(V_m == 0):
        return np.zeros((Nk, Nk))
    V_m_L = V_m * L_A
    return compute_U_mn_fast(W_m, R2_m, V_m_L)


def compute_V_m(W_m, R2_m, L_A):
    '''Compute normalization factor for Px'''
    # convolve W and R2 arrays 
    W_R2 = np.fft.ifft(np.fft.fft(W_m)* np.fft.fft(R2_m))
    return np.abs(W_R2) / L_A
