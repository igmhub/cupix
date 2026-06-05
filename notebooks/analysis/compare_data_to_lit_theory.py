# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # This notebook compares the Px results with the predictions using best-fit values from several different analyses

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from cupix.likelihood.theory import Theory
from cupix.obsolete.forestflow_emu import FF_emulator
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from lace.cosmo import cosmology
from lace.cosmo.thermal_broadening import thermal_broadening_kms
from cupix.px_data.data_DESI_DR2 import DESI_DR2

# %% [markdown]
# Load the data

# %%
# data = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/binned_out_px-zbins_4-thetabins_10.hdf5", kmax_cut_AA=1)
data_config = {'data_file':"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5", 'kmax_cut_AA':1, 'theta_min_cut_arcmin':3}
data = DESI_DR2(data_config)

print(data.z)

# %% [markdown]
# Load the theory
#

# %%
# use default cosmo
cosmo = cosmology.Cosmology()

theory_config = {'default_lya_model': 'best_fit_igm_from_p1d', 'verbose':True, 'include_hcd':True, 'include_sky':True,
                 'include_continuum':True, 'include_metals':True}
theories = []
for z in data.z:
    theories.append(Theory(z, fid_cosmo=cosmo, config=theory_config))

# %%
# choose a redshift bin to analyse
iz_choice = 0
z = data.z[iz_choice]

# %%
compare_to = "W18"
# compare_to = "CM25"
# compare_to = "AK25"
if compare_to == "W18":
    if z == 2.2:
        T0 = 1.014*1e4 # Kelvin
        gamma = 1.74
        mF = 0.825
        lambdap = 79.4 # [kpc]
    # Walther+ constriaints
    elif z == 2.4:
        T0 = 1.165*1e4
        gamma = 1.63
        mF = 0.799
        lambdap = 81.1 # [kpc]
    else:
        print("Need to input the other redshift values.")

dkms_dMpc_z = theories[iz_choice].fid_cosmo.get_dkms_dMpc(z)
sigma_T_kms = thermal_broadening_kms(T0)
sigT_Mpc = sigma_T_kms / dkms_dMpc_z
kF_Mpc = 1/(lambdap/1000)

# %%
if compare_to == "CM25":
    assert theories[iz_choice].lya_model.default_lya_model == 'best_fit_igm_from_p1d'
    params = {}
else:
    params = {'sigT_Mpc':sigT_Mpc, 'kF_Mpc':kF_Mpc, 'mF':mF, 'gamma':gamma}

# %%
like = Likelihood(iz=iz_choice, data=data, theory=theories[iz_choice], config={'verbose':True})

# %%
like.plot_px(params=params, ylim=[0,0.003], theorylabel=f'best fit from {compare_to}', title=f'z={z:.1f}', include_probability=False, include_chi2=True)


# %%

# %%
