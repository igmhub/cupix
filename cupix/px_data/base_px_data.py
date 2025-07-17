import os, sys
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

from cupix.utils.utils import get_path_repo


def _drop_zbins(
    z_in,
    k_in,
    Pk_in,
    cov_in,
    z_min,
    z_max,
    full_zs=None,
    full_Pk_AA=None,
    full_cov_AA=None,
    Pksmooth_AA=None,
    cov_stat=None,
    kmin_in=None,
    kmax_in=None,
):
    """Drop redshift bins below z_min or above z_max"""

    z_in = np.array(z_in)
    ind = np.argwhere((z_in >= z_min) & (z_in <= z_max))[:, 0]
    z_out = z_in[ind]

    k_out = []
    kmin_out = []
    kmax_out = []
    Pk_out = []
    cov_out = []
    cov_stat_out = []
    if Pksmooth_AA is None:
        Pksmooth_out = None
    else:
        Pksmooth_out = []
    if Pksmooth_AA is None:
        Pksmooth_out = None
    for jj in ind:
        # remove tailing zeros
        ind = np.argwhere(Pk_in[jj] != 0)[:, 0]
        k_out.append(k_in[jj][ind])
        kmin_out.append(kmin_in[jj][ind])
        kmax_out.append(kmax_in[jj][ind])
        Pk_out.append(Pk_in[jj][ind])
        if Pksmooth_AA is not None:
            Pksmooth_out.append(Pksmooth_AA[jj][ind])
        cov_out.append(cov_in[jj][ind, :][:, ind])
        if cov_stat is not None:
            cov_stat_out.append(cov_stat[jj][ind, :][:, ind])

    if full_zs is not None:
        ind = np.argwhere((full_zs >= z_min) & (full_zs <= z_max))[:, 0]
        full_zs = full_zs[ind]
        full_Pk_AA = full_Pk_AA[ind]
        full_cov_AA = full_cov_AA[ind, :][:, ind]

    return (
        z_out,
        k_out,
        Pk_out,
        cov_out,
        full_zs,
        full_Pk_AA,
        full_cov_AA,
        Pksmooth_out,
        cov_stat_out,
        kmin_out,
        kmax_out,
    )


class BaseDataPx(object):
    """Base class to store measurements of the cross power spectrum"""

    BASEDIR = os.path.join(get_path_repo("cupix"), "data", "px_measurements")

    def __init__(
        self,
        z,
        _k_AA,
        Pk_AA,
        cov_Pk_AA,
        weights,
        thetabin_deg,
        window=None,
        zmin=0.0,
        zmax=10.0,
        k_AA_min=None,
        k_AA_max=None,
        full_zs=None,
        full_Pk_AA=None,
        full_cov_AA=None,
    ):
        """Construct base Px class, from measured power and covariance"""

        ## if multiple z, ensure that k_AA for each redshift
        # more than one z, and k_AA is different for each z
        if (len(z) > 1) & (len(np.atleast_1d(_k_AA[0])) != 1):
            k_AA = []
            for iz in range(len(z)):
                k_AA.append(_k_AA[iz])
        # more than one z, and AA is the same for all z
        elif (len(z) > 1) & (len(np.atleast_1d(_k_AA[0])) == 1):
            k_AA = []
            for iz in range(len(z)):
                k_AA.append(_k_AA)
        # only one z
        else:
            k_AA = _k_AA

        self.z = np.array(z)
        self.k_AA = k_AA
        self.Pk_AA = Pk_AA
        self.cov_Pk_AA = cov_Pk_AA
        self.weights = weights
        self.thetabin_deg = thetabin_deg
        self.window = window
        self.k_AA_min = 0.001 # placeholder for now
        self.k_AA_max = 10 # placeholder for now
        self.full_Pk_AA = full_Pk_AA
        self.full_cov_Pk_AA = full_cov_AA
        self.full_zs = full_zs
        # decide if applying blinding
        self.apply_blinding = False
        if hasattr(self, "blinding"):
            if self.blinding is not None:
                self.apply_blinding = True

    def get_Pk_iz(self, iz):
        """Return P1D in units of km/s for redshift bin iz"""

        return self.Pk_AA[iz]

    def get_cov_iz(self, iz):
        """Return covariance of P1D in units of (km/s)^2 for redshift bin iz"""

        return self.cov_Pk_AA[iz]

    def get_icov_iz(self, iz):
        """Return covariance of P1D in units of (km/s)^2 for redshift bin iz"""

        return self.icov_Pk_AA[iz]

    def cull_data(self, kmin_AA=0, kmax_AA=10):
        """Remove bins with wavenumber k < kmin_AA and k > kmin_AA"""

        if (kmin_AA is None) & (kmax_AA is None):
            return

        for iz in range(len(self.z)):
            ind = np.argwhere(
                (self.k_AA[iz] >= kmin_AA) & (self.k_AA[iz] <= kmax_AA)
            )[:, 0]
            sli = slice(ind[0], ind[-1] + 1)
            self.k_AA[iz] = self.k_AA[iz][sli]
            self.Pk_AA[iz] = self.Pk_AA[iz][sli]
            self.cov_Pk_AA[iz] = self.cov_Pk_AA[iz][sli, sli]
            self.icov_Pk_AA[iz] = self.icov_Pk_AA[iz][sli, sli]

    def plot_p1d(
        self, use_dimensionless=True, xlog=False, ylog=True, fname=None
    ):
        """Plot P1D mesurement. If use_dimensionless, plot k*P(k)/pi."""

        N = len(self.z)
        for i in range(N):
            k_AA = self.k_AA[i]
            Pk_AA = self.get_Pk_iz(i)
            err_Pk_AA = np.sqrt(np.diagonal(self.get_cov_iz(i)))
            if use_dimensionless:
                fact = k_AA / np.pi
            else:
                fact = 1.0
            plt.errorbar(
                k_AA,
                fact * Pk_AA,
                yerr=fact * err_Pk_AA,
                label="z = {}".format(np.round(self.z[i], 3)),
            )

        plt.legend(ncol=4)
        if ylog:
            plt.yscale("log", nonpositive="clip")
        if xlog:
            plt.xscale("log")
        plt.xlabel("k [s/km]")
        if use_dimensionless:
            plt.ylabel(r"$k P(k)/ \pi$")
        else:
            plt.ylabel("P(k) [km/s]")

        if fname is not None:
            plt.savefig(fname)
        else:
            plt.show()

def normalize(W, R, L):
    '''
    W (np.ndarray): vector length N, average FFT of the weights
    R (np.ndarray): vector length N, resolution in Fourier space
    L (float): length of the spectra (in physical units, e.g. Angstroms or Mpc)
    Returns:
    norm (np.ndarray): vector length N, to be multiplied by every Px mode of the measurement
    '''
    R2 = R.real**2 + R.imag**2
    denom = np.absolute(np.fft.ifft(np.fft.fft(W)* np.fft.fft(R2)))
    norm = np.absolute(L/denom)
    return norm