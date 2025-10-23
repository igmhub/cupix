# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import numpy as np
import camb

# specify global settings to CAMB calls
camb_kmin_Mpc = 1.0e-4
camb_npoints = 1000
camb_fluid = 8
# no need to go beyond this k_Mpc when fitting linear power only
camb_fit_kmax_Mpc = 1.5
# set kmax in transfer function beyond what you need (avoid warnings)
camb_extra_kmax = 1.001
clight_kms = 299792.458


def get_cosmology(
    H0=67.66,
    mnu=0.0,
    omch2=0.119,
    ombh2=0.0224,
    omk=0.0,
    As=2.105e-09,
    ns=0.9665,
    nrun=0.0,
    pivot_scalar=0.05,
    w=-1,
    wa=0,
):
    """Given set of cosmological parameters, return CAMB cosmology object.

    Fiducial values for Planck 2018
    """

    pars = camb.CAMBparams()
    # set background cosmology
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, mnu=mnu)
    # set DE
    pars.set_dark_energy(w=w, wa=wa)
    # set primordial power
    pars.InitPower.set_params(
        As=As, ns=ns, nrun=nrun, pivot_scalar=pivot_scalar
    )

    return pars



# %%
cosmo_fid = get_cosmology()
camb_results = camb.get_results(cosmo_fid)

# %%
camb_results.comoving_radial_distance(3)


# %%
from lace.cosmo import camb_cosmo
from astropy.cosmology import FlatLambdaCDM
P18 = FlatLambdaCDM(H0=cosmo_fid.H0, Om0=cosmo_fid.omegam, Tcmb0=cosmo_fid.TCMB)
import astropy.units as u

# %%
z = .7
angle_deg = np.array([0,5,10,20])
angle_Mpc_ap = (P18.kpc_comoving_per_arcmin(z)).to(u.Mpc/u.deg)*angle_deg*u.deg


# %%
angle_Mpc_ap

# %%
1/camb_cosmo.ddeg_dMpc(cosmo_fid, z) * angle_deg

# %%
