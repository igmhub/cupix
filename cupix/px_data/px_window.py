import numpy as np
from cupix.px_data import px_binning, px_ztk

class Px_w(px_ztk.BasePx):
    '''Derived BasePx object, with information related to window matrix.
       This object should only be used without rebinning. '''

    def __init__(self, z_bins, list_px_z):
        super().__init__(z_bins, list_px_z)

        # if doing rebinning, do not use this object (ill-defined)
        assert isinstance(self.k_bins[0], px_binning.Discrete_k)

        return


class Px_zt_w(px_ztk.Px_zt):
    '''Derived Px_zt object, with information related to window matrix.
       This object should only be used without rebinning. '''

    def __init__(self, z_bin, t_bin, k_bins, P_m, V_m, F_m, W_m, T_m, U_mn, C_mn=None):
        '''Provide extra information related to window / weights'''

        # if doing rebinning, do not use this object (ill-defined)
        assert isinstance(k_bins[0], px_binning.Discrete_k)

        super().__init__(
                z_bin=z_bin, 
                t_bin=t_bin, 
                k_bins=k_bins,
                P_m=P_m,
                V_m=V_m,
                C_mn=C_mn)

        # (complex) sum of the product of FFT of delta * weights
        self.F_m = F_m
        # (complex) sum of the product of FFT of weights
        self.W_m = W_m
        # (complex) sum of the product of FFT of resolution * weights
        self.T_m = T_m
        # window matrix
        self.U_mn = U_mn

        return


    @classmethod
    def from_unnormalized(cls, z_bin, t_bin, k_bins, F_m, W_m, T_m, L):
        '''Construct object from unnormalized quantities'''

        P_m, V_m = normalize_Px(F_m, W_m, T_m, L)
        C_mn = None
        # do not compute the window matrix for now
        U_mn = None

        return Px_zt_w(z_bin, t_bin, k_bins, P_m, V_m, F_m, W_m, T_m, U_mn, C_mn)


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
