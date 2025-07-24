import numpy as np


class Bin_z(object):
    '''Description of a particular redshift bin'''

    def __init__(self, min_z, max_z, mean_z=None, label=None):
        '''Construct by providing edges, and plotting label'''

        self.min_z = min_z
        self.max_z = max_z

        if label is None:
            self.label = f'{min_z} < z < {max_z}'
        else:
            self.label = label

        # compute mean redshift in bin if needed
        if mean_z is None:
            self.mean_z = 0.5*(self.min_z + self.max_z)
        else:
            self.mean_z = mean_z



class Bin_t(object):
    '''Description of a particular theta bin'''

    def __init__(self, min_t, max_t, mean_t=None, label=None):
        '''Construct by providing edges, and plotting label'''

        self.min_t = min_t
        self.max_t = max_t
        if label is None:
            self.label = f'{tmin} < theta [arcmin] < {tmax}'
        else:
            self.label = label

        # compute mean separation in bin if needed
        if mean_t is None:
            self.mean_t = 0.5*(self.min_t + self.max_t)
        else:
            self.mean_t = mean_t


    def B_t(self, t):
        '''Contribution of theta value to this bin'''

        # this could be made more efficient by allowing arrays, if needed
        if (t > self.min_t) and (t <= self.max_t):
            return 1.0
        else:
            return 0.0



class Discrete_k(object):
    '''Description of a particular wavenumber k (no binning)'''

    def __init__(self, k, label=None):
        '''Construct by providing value and plotting label'''

        self.mean_k = k
        if label is None:
            self.label = f'k = {mean_k} 1/A'
        else:
            self.label = label



class Bin_k(object):
    '''Description of a particular k bin'''

    def __init__(self, min_k, max_k, mean_k=None, label=None):
        '''Construct by providing edges, and plotting label'''

        self.min_k = min_k
        self.max_k = max_k
        if label is None:
            self.label = f'{min_k} < k [1/A] < {max_k}'
        else:
            self.label = label

        # compute mean wavenumber in bin if needed
        if mean_k is None:
            self.mean_k = 0.5*(self.min_k + self.max_k)
        else:
            self.mean_k


    def B_k(self, k):
        '''Contribution of k value to this k-bin'''

        # this could be made more efficient by allowing arrays, if needed
        if (k > self.min_k) and (k <= self.max_k):
            return 1.0
        else:
            return 0.0

