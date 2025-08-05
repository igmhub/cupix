import numpy as np
import os
import h5py

from cupix.px_data.base_px_data import BaseDataPx, normalize


class Px_AK25(BaseDataPx):
    """Class containing Px from Lyacolore"""

    def __init__(self):
        """Read measured Px."""

        # folder storing P1D measurements
        datadir = BaseDataPx.BASEDIR + "/Lyacolore/"

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z, k, px, px_covar, px_weights, theta_bins = read_from_file(datadir)

        super().__init__(z, k, px, px_covar, px_weights, theta_bins)

        return

def read_from_file(datadir):
    # tbd
    return