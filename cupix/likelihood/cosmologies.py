import os
import numpy as np

import lace
from lace.cosmo import camb_cosmo


def get_cosmology_from_label(cosmo_label="default"):
    if cosmo_label == "default":
        return camb_cosmo.get_cosmology()
    elif cosmo_label == "low_omch2":
        return camb_cosmo.get_cosmology(omch2=0.11)
    elif cosmo_label == "high_omch2":
        return camb_cosmo.get_cosmology(omch2=0.13)
    elif cosmo_label == "omch2_0115":
        return camb_cosmo.get_cosmology(omch2=0.115)
    elif cosmo_label == "omch2_0125":
        return camb_cosmo.get_cosmology(omch2=0.125)
    elif cosmo_label == "mnu_03":
        return camb_cosmo.get_cosmology(mnu=0.3)
    elif cosmo_label == "mnu_06":
        return camb_cosmo.get_cosmology(mnu=0.6)
    elif cosmo_label == "SHOES":
        return camb_cosmo.get_cosmology(H0=73.0)
    else:
        raise ValueError("implement cosmo_label " + cosmo_label)


def set_cosmo(
    cosmo_label="mpg_central",
    return_all=False,
    nyx_version="models_Nyx_Mar2025_with_CGAN_val_3axes",
):
    """Set fiducial cosmology

    Parameters
    ----------
    cosmo_label : str

    Returns
    -------
    cosmo : object
    """
    if (cosmo_label[:3] == "mpg") | (cosmo_label[:3] == "nyx"):
        if cosmo_label[:3] == "mpg":
            repo = os.path.dirname(lace.__path__[0]) + "/"
            fname = repo + ("data/sim_suites/Australia20/mpg_emu_cosmo.npy")
            get_cosmo = camb_cosmo.get_cosmology_from_dictionary
        elif cosmo_label[:3] == "nyx":
            fname = (
                os.environ["NYX_PATH"] + "nyx_emu_cosmo_" + nyx_version + ".npy"
            )
            get_cosmo = camb_cosmo.get_Nyx_cosmology

        try:
            data_cosmo = np.load(fname, allow_pickle=True).item()
        except:
            raise ValueError(f"{fname} not found")

        if cosmo_label in data_cosmo.keys():
            # print(data_cosmo[cosmo_label]["cosmo_params"])
            cosmo = get_cosmo(data_cosmo[cosmo_label]["cosmo_params"])
        else:
            raise ValueError(f"Cosmo not found in {fname} for {cosmo_label}")

    elif cosmo_label == "Planck18":
        cosmo = camb_cosmo.get_cosmology(
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
        )
    elif cosmo_label == "Planck18_nyx":
        cosmo = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=0.0,
            omch2=0.119,
            ombh2=0.0224,
            omk=0.0,
            As=2.24e-09,
            ns=0.937,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    elif cosmo_label == "Planck18_mpg":
        cosmo = camb_cosmo.get_cosmology(
            H0=67.0,
            mnu=0.0,
            omch2=0.119,
            ombh2=0.022,
            omk=0.0,
            As=2.26e-09,
            ns=0.982,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    elif cosmo_label == "Planck18_h74":
        cosmo = camb_cosmo.get_cosmology(
            H0=74,
            mnu=0.0,
            omch2=0.119,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    elif (cosmo_label == "ACCEL2_6144_160") | (cosmo_label == "accel2"):
        # https://arxiv.org/pdf/2407.04473
        # Planck15 ΛCDM Planck TT,TE,EE+lowP (approx...)
        Omegam = 0.31
        Omegab = 0.0487
        h = 0.675
        omch2 = (Omegam - Omegab) * h**2
        ombh2 = Omegab * h**2
        cosmo = camb_cosmo.get_cosmology(
            H0=h * 100,
            mnu=0.0,
            omch2=omch2,
            ombh2=ombh2,
            omk=0.0,
            As=np.exp(3.094) / 1e10,  # Planck15 ΛCDM Planck TT,TE,EE+lowP
            ns=0.96,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    elif (cosmo_label == "Sherwood_2048_40") | (cosmo_label == "sherwood"):
        # https://academic.oup.com/mnras/article/464/1/897/2236089
        # Planck13 ΛCDM Planck+WP+highL+BAO
        Omegam = 0.308
        Omegab = 0.0482
        h = 0.678
        omch2 = (Omegam - Omegab) * h**2
        ombh2 = Omegab * h**2
        cosmo = camb_cosmo.get_cosmology(
            H0=h * 100,
            mnu=0.0,
            omch2=omch2,
            ombh2=ombh2,
            omk=0.0,
            As=np.exp(3.0973) / 1e10,  # Planck13 ΛCDM Planck+WP+highL+BAO
            ns=0.961,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    else:
        raise ValueError(f"cosmo_label {cosmo_label} not implemented")

    if return_all:
        return data_cosmo
    else:
        return cosmo
