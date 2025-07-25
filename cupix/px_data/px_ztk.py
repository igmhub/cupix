import numpy as np


class Px_zt(object):
    '''Collection of Px measurements at many k bins, same (z,t)'''

    def __init__(self, z_bin, t_bin, k_bins, P_m, V_m=None, C_mn=None):
        '''Construct Px measurement at bin (z,t), for different k-bins'''

        # P_m, V_m and C_mn should be numpy arrays
        self.Nk = len(k_bins)
        assert self.Nk == P_m.size, 'size mismatch'
        self.k_bins = k_bins
        self.P_m = P_m

        # before combining different healpixels, we do not have a covariance
        if C_mn is not None:
            assert C_mn.size == self.Nk**2
            self.C_mn = C_mn
        else:
            self.C_mn = None

        # this is used to weight different bins when rebinning
        if V_m is not None:
            assert V_m.size == self.Nk
            self.V_m = V_m
        else:
            self.V_m = None

        # store information about this particular bin (z, t)
        self.z_bin = z_bin
        self.t_bin = t_bin


class Px_z(object):
    '''Collection of Px measurements at many (t,k) bins, same z'''

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


class BasePx(object):
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
