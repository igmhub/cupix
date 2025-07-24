import numpy as np


class Px_zt(object):
    '''Px measurement at one (z,t), including many k bins'''

    def __init__(self, z_bin, t_bin, k_bins, px_ztk, cov_px_ztk=None, V_ztk=None):
        '''Construct Px measurement at bin (z,t), for different k-bins'''

        # px_ztk and cov_px_ztk should numpy (nd)arrays
        self.Nk = len(k_bins)
        assert self.Nk == px_ztk.size, 'size mismatch'
        self.k_bins = k_bins
        self.px = px_ztk

        # before combining different healpixels, we do not have a covariance
        if cov_px_ztk is not None:
            assert cov_px_ztk.size == self.Nk**2
            self.cov = cov_px_ztk
        else:
            self.cov = None

        # this is used to weight differnet bins when rebinning
        if V_ztk is not None:
            assert V_ztk.size == self.Nk
            self.V_ztk = V_ztk
        else:
            self.V_ztk = None

        # eventually, we should also keep track of W_ztk for the window matrix
        # self.W_ztk = None

        # store information about this particular bin (z, t)
        self.z_bin = z_bin
        self.t_bin = t_bin



class Px_z(object):
    '''Px measurement at one z, including many (t,k) bins'''

    def __init__(self, t_bins, list_px_zt):
        '''Construct from list of theta, Px(z, theta)'''

        # store information about theta bins, and their Px
        assert len(t_bins) == len(list_px_zt), 'size mismatch'
        self.t_bins = t_bins
        self.list_px_zt = list_px_zt

        # store information about this z-bin
        self.z_bin = list_px_zt[0].z_bin
        for px_zt in list_px_zt:
            assert px_zt.z_bin == self.z_bin, 'inconsistent binning'

        # finally get k bins, and check consistency
        self.k_bins = list_px_zt[0].k_bins
        for px_z in list_px_zt:
            assert px_z.k_bins == self.k_bins, 'inconsistent binning'


class BaseDataPx(object):
    '''Base class to store measurements of the cross power spectrum'''

    def __init__(self, z_bins, list_px_z):
        '''Construct from lists of z bins, Px(z)'''

        # store information about z bins, and their Px
        assert len(z_bins) == len(list_px_z), 'size mismatch'
        self.z_bins = z_bins
        self.list_px_z = list_px_z

        # get also theta bins, and check consistency
        self.t_bins = list_px_z[0].t_bins
        for px_z in list_px_z:
            assert px_z.t_bins == self.t_bins, 'inconsistent binning'

        # finally get k bins, and check consistency
        self.k_bins = list_px_z[0].k_bins
        for px_z in list_px_z:
            assert px_z.k_bins == self.k_bins, 'inconsistent binning'
