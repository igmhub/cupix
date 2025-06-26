import numpy as np
import os
import h5py

from cupix.px_data.base_px_data import BaseDataPx, normalize


class Px_Lyacolore(BaseDataPx):
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
    px_file = datadir + "/px_measurement.h5"
    # assumes single redshift for now
    with h5py.File(px_file, 'r') as f:
        z = f.attrs['z']
        k = f['k_arr'][:]
        dz = f.attrs['dz']
        dx = f.attrs['pixel_width_A']
        N  = len(k)
        L  = dx * N
        R = np.ones(N) # perfect resolution

        # Loop over all theta groups
        theta_keys = sorted([key for key in f.keys() if key.startswith('theta_')],
                        key=lambda k: float(k.split('_')[1]))  # sort by theta_min in arcmin
        
        ntheta = len(theta_keys)
        px = np.zeros((ntheta, k.size))
        px_covar = np.zeros((ntheta, k.size, k.size))
        px_weights = np.zeros((ntheta, k.size))
        theta_bins = np.zeros((ntheta, 2))  # to store (theta_min, theta_max) in radians
        for i, key in enumerate(theta_keys):
            g = f[key]
            px_var = g['px_var'][:]
            # make diagonal covariance matrix
            px_covar[i,:,:] = np.diag(px_var)
            px_weights[i,:] = g['px_weights'][:]
            # normalize (assuming all k are the same for every theta. If this changes, would need to update)
            norm = normalize(px_weights[i,:], R, L)
            px[i,:] = g['px'][:] * norm
            theta_bins[i,:] = [g.attrs['theta_min'], g.attrs['theta_max']]
    theta_bins = np.degrees(theta_bins) # convert from radian to deg
    mean_ang_sep = np.average(theta_bins, axis=1)
    # return arrays as shape [Nz, Ntheta, Nk]
    px = px[np.newaxis, :, :]  # add redshift dimension
    return z, k, px, px_covar, px_weights, theta_bins

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

