import numpy as np


class Bin_z(object):
    '''Description of a particular redshift bin'''

    def __init__(self, min_z, max_z, label=None):
        '''Construct by providing edges, and plotting label'''

        self.min_z = min_z
        self.max_z = max_z

        if label is None:
            self.label = f'{min_z} < z < {max_z}'
        else:
            self.label = label


    def mean(self):
        return 0.5*(self.min_z+self.max_z)



class Bin_t(object):
    '''Description of a particular theta bin'''

    def __init__(self, min_t, max_t, label=None):
        '''Construct by providing edges, and plotting label'''

        self.min_t = min_t
        self.max_t = max_t
        if label is None:
            self.label = f'{min_t} < theta [arcmin] < {max_t}'
        else:
            self.label = label


    def mean(self):
        return 0.5*(self.min_t+self.max_t)


    def B_t(self, t):
        '''Contribution of theta value to this bin'''

        # this could be made more efficient by allowing arrays, if needed
        if (t >= self.min_t) and (t < self.max_t):
            return 1.0
        else:
            return 0.0



class Discrete_k(object):
    '''Description of a particular wavenumber k (no binning)'''

    def __init__(self, k, label=None):
        '''Construct by providing value and plotting label'''

        self.k = k
        if label is None:
            self.label = f'k = {k} 1/A'
        else:
            self.label = label



class Bin_k(object):
    '''Description of a particular k bin'''

    def __init__(self, min_k, max_k, label=None):
        '''Construct by providing edges, and plotting label'''

        self.min_k = min_k
        self.max_k = max_k
        if label is None:
            self.label = f'{min_k} < k [1/A] < {max_k}'
        else:
            self.label = label


    def mean(self):
        return 0.5*(self.min_k+self.max_k)


    def B_k(self, k):
        '''Contribution of k value to this k-bin'''

        # this could be made more efficient by allowing arrays, if needed
        if (k >= self.min_k) and (k < self.max_k):
            return 1.0
        else:
            return 0.0


def get_coarser_k_bins(in_k_bins, rebin_factor, include_k_0=True):
    '''Return coarser bins, rebin_factor coarser than in_k_bins.
       If include_k_0=True, include k=0 instead of k_Ny.'''

    # this function for now should be called on Discrete_k bins
    assert isinstance(in_k_bins[0], Discrete_k)
    in_k = [ k_bin.k for k_bin in in_k_bins ]
    in_dk = in_k[1]-in_k[0]
    in_Nk = len(in_k)
    in_k_max = np.max(np.abs(in_k))
    #print(f'input Nk = {in_Nk}, dk = {in_dk:.4f}, k_max = {in_k_max:.4f}')

    # extra factor of 2 here since in_k_bins include negative frequencies
    out_Nk = int(in_Nk / 2 / rebin_factor)
    if include_k_0:
        # include k=0 in first bin
        k_shift = -0.5*in_dk
    else:
        # include k_Ny in last bin
        k_shift = +0.5*in_dk
    out_k_edges = np.linspace(k_shift, in_k_max+k_shift, num = out_Nk+1, endpoint=True)
    out_dk = out_k_edges[1] - out_k_edges[0]
    out_k_max = np.max(np.abs(out_k_edges))
    #print(f'out Nk = {out_Nk}, dk = {out_dk:.4f}, k_max = {out_k_max:.4f}')
    out_k_bins = []
    for ik in range(out_Nk):
        k_bin = Bin_k(out_k_edges[ik], out_k_edges[ik+1])
        out_k_bins.append(k_bin)
    return out_k_bins


def get_coarser_t_bins(in_t_bins, rebin_factor):
    '''Return coarser bins, rebin_factor coarser than in_t_bins'''

    in_Nt = len(in_t_bins)
    assert in_Nt % rebin_factor == 0, 'size mismatch'
    in_min_t = [ t_bin.min_t for t_bin in in_t_bins ]
    in_max_t = [ t_bin.max_t for t_bin in in_t_bins ]
    out_min_t = in_min_t[::rebin_factor]
    out_max_t = in_max_t[rebin_factor-1::rebin_factor]
    out_Nt = len(out_min_t)
    out_t_bins = []
    for it in range(out_Nt):
        #print(f'{it}, {out_min_t[it]} < t < {out_max_t[it]}')
        t_bin = Bin_t(out_min_t[it], out_max_t[it])
        out_t_bins.append(t_bin)
    return out_t_bins

