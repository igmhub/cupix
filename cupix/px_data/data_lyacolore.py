import numpy as np
import os
import h5py

from cupix.px_data.base_px_data import BaseDataPx
from cupix.window.window import window_matrix_from_fft_weights

class Px_Lyacolore(BaseDataPx):
    """Class containing Px from Lyacolore"""

    def __init__(self):
        """Read measured Px."""

        # folder storing P1D measurements
        datadir = BaseDataPx.BASEDIR + "/Lyacolore/"

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z, k, px, px_covar, px_weights, theta_bins, window = read_from_file(datadir)

        super().__init__(z, k, px, px_covar, px_weights, theta_bins, window)

        return

def read_from_file(datadir):
    # assumptions: all k are the same for every theta. If this changes, would need to update
            
    px_file = datadir + "/px-nhp_1147_zbin_2.2.hdf5"
    # assumes single redshift for now
    with h5py.File(px_file, 'r') as f:
        z = np.array([f.attrs['z']])
        dz = f.attrs['dz']
        dx = f.attrs['pixel_width_A']
        k = f['k_arr'][:]
        k_keep = k>0 # only keep positive k for now
        k_lim = k[k_keep]
        N  = k.size
        L  = dx * N
        R = np.ones(N) # perfect resolution
        N_keep = k_keep.sum()  # number of k values to keep

        # Loop over all theta groups
        theta_keys = sorted([key for key in f.keys() if key.startswith('theta_')],
                        key=lambda k: float(k.split('_')[1]))  # sort by theta_min in arcmin
        
        ntheta = len(theta_keys)
        px = np.zeros((ntheta, N_keep))
        px_covar = np.zeros((ntheta, N_keep, N_keep))
        px_weights = np.zeros((ntheta, N_keep))
        theta_bins = np.zeros((ntheta, 2))  # to store (theta_min, theta_max) in radians
        window = np.zeros((ntheta, N_keep, N_keep))  # window matrix for each theta
        for i, key in enumerate(theta_keys):
            g = f[key]
            # normalize and calculate the window matrix # norm = normalize(px_weights[i,:], R, L)
            px_weights_i = g['px_weights']
            window_matrix, norm = window_matrix_from_fft_weights(px_weights_i, R, L)
            window[i,:,:] = window_matrix[np.ix_(k_keep, k_keep)]
            px_weights[i,:] = g['px_weights'][k_keep]
            px_var = (g['px_var'] * norm)[k_keep]
            # make diagonal covariance matrix
            px_covar[i,:,:] = np.diag(px_var)
            px[i,:] = (g['px'] * norm)[k_keep]
            theta_bins[i,:] = [g.attrs['theta_min'], g.attrs['theta_max']]
    theta_bins = np.degrees(theta_bins) # convert from radian to deg
    mean_ang_sep = np.average(theta_bins, axis=1)
    if z.size == 1:
        # return arrays as shape [Nz, Ntheta, Nk]
        px = px[np.newaxis, :, :]
        theta_bins = theta_bins[np.newaxis, :, :]
        k_lim = k_lim[np.newaxis, :]
        px_covar = px_covar[np.newaxis, :, :, :]
        px_weights = px_weights[np.newaxis, :]
        window = window[np.newaxis, :, :, :]
    return z, k_lim, px, px_covar, px_weights, theta_bins, window

# def read_from_file(datadir, velunits):
#     """Reconstruct covariance matrix from files."""


#     px_file = datadir + "/p1d_measurement.txt"

#     inz, ink, inPk = np.loadtxt(
#         px_file,
#         unpack=True,
#         usecols=range(
#             3,
#         ),
#     )
#     # store unique values of redshift and wavenumber
#     z = np.unique(inz)
#     Nz = len(z)

#     mask = inz == z[0]
#     k = ink[mask]
#     Nk = len(k)

#     # re-shape matrices, and compute variance (statistics only for now)
#     if velunits:
#         Pk = []
#         for i in range(len(z)):
#             mask = inz == z[i]
#             Pk.append(inPk[mask][:Nk] * np.pi / k)
#         Pk = np.array(Pk)
#     else:
#         Pk = np.reshape(inPk * np.pi / k, [Nz, Nk])

#     # now read correlation matrices
#     if velunits:
#         cov_file = datadir + "/covariance_matrix_kms.txt"
#     else:
#         cov_file = datadir + "/covariance_matrix.txt"

#     inzcov, ink1, _, incov = np.loadtxt(
#         cov_file,
#         unpack=True,
#         usecols=range(
#             4,
#         ),
#     )
#     if velunits:
#         cov_Pk = []
#         for i in range(Nz):
#             mask = inzcov == z[i]
#             k1 = np.unique(ink1[mask])
#             cov_Pk_z = []
#             for j in range(Nk):
#                 mask_k = mask & (ink1 == k1[j])
#                 cov_Pk_z.append(incov[mask_k][:Nk])
#             cov_Pk.append(cov_Pk_z)
#         cov_Pk = np.array(cov_Pk)
#     else:
#         cov_Pk = np.reshape(
#             incov,
#             [Nz, Nk, Nk],
#         )

#     return z, k, Pk, cov_Pk

