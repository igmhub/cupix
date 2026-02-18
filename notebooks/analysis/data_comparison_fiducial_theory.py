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
#     display_name: forestflow
#     language: python
#     name: forestflow
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from lace.cosmo import camb_cosmo, fit_linP
from lace.cosmo.thermal_broadening import thermal_broadening_kms
import matplotlib.pyplot as plt
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import scipy
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
from astropy.io import fits
import cupix
import os
omnuh2 = 0.0006
mnu = omnuh2 * 93.14
H0 = 67.36
omch2 = 0.12
ombh2 = 0.02237
As = 2.1e-9
ns = 0.9649
nrun = 0.0
w = -1.0
omk = 0
fid_cosmo = {
    'H0': H0,
    'omch2': omch2,
    'ombh2': ombh2,
    'mnu': mnu,
    'omk': omk,
    'As': As,
    'ns': ns,
    'nrun': nrun,
    'w': w
}
theory_AA = None

# %% [markdown]
# Load the data

# %%
# data = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/binned_out_px-zbins_4-thetabins_10.hdf5", kmax_cut_AA=1)
data = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/binned_out_px-zbins_4-thetabins_20_w_res.hdf5", kmax_cut_AA=1, theta_min_cut_arcmin=3)
# choose a redshift bin to analyse
iz_choice = 0
z = np.array([data.z[iz_choice]])
# z = np.asarray([data.z[iz_choice]])
print(z)

# %% [markdown]
# Load the theory
#

# %%
theory_AA = None

# %%
# Load emulator
if theory_AA is None: # only do this once per notebook
    sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
    
    # compute linear power parameters at each z (in Mpc units)
    linP_zs = fit_linP.get_linP_Mpc_zs(
        sim_cosmo, z, 0.7
    )
    dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=z)
    cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)
    ffemu = FF_emulator(z, fid_cosmo, cc)
    ffemu.kp_Mpc = 0.7 # set pivot point

    theory_AA = set_theory(ffemu, k_unit='iAA')
    theory_AA.set_fid_cosmo(z)
    theory_AA.emulator = ffemu


# %%
theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu

# %%
# compare_to = "W18"
# compare_to = "CM25"
compare_to = "AK25"
if compare_to == "W18":
    if z[0] == 2.2:
        T0 = 1.014*1e4 # Kelvin
        gamma = 1.74
        mF = 0.825
        lambdap = 79.4 # [kpc]
    # Walther+ constriaints
    elif z[0] == 2.4:
        T0 = 1.165*1e4
        gamma = 1.63
        mF = 0.799
        lambdap = 81.1 # [kpc]
    else:
        print("Need to input the other redshift values.")
elif compare_to=="CM25":
    # input P1D constraints
    if z[0] == 2.2:
        T0 = 1.537e4
        gamma = 1.88
        mF = 0.8147
        lambdap = 79.4 # [kpc]
    elif z[0] == 2.4:
        T0 = 1.329*1e4
        gamma = 1.682
        mF = 0.7905
        lambdap = 81.1 # [kpc]
    
    else:
        print("Need to input the other redshift values.")
if compare_to == "AK25":
    if z[0] == 2.2:
        T0 = 1.014*1e4 # Kelvin
        gamma = 1.74
        mF = 0.839
        lambdap = 79.4 # [kpc]
    # Walther+ constriaints
    elif z[0] == 2.4:
        T0 = 1.165*1e4
        gamma = 1.63
        mF = 0.799
        lambdap = 81.1 # [kpc]
    else:
        print("Need to input the other redshift values.")
        
sigma_T_kms = thermal_broadening_kms(T0)
sigT_Mpc = sigma_T_kms / dkms_dMpc_zs[0]
kF_Mpc = 1/(lambdap/1000)

# %%
like_params = []
like_params.append(LikelihoodParameter(
    name='sigT_Mpc',
    min_value=-1.0,
    max_value=1.0,
    value=sigT_Mpc,
    ini_value=sigT_Mpc,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='gamma',
    min_value=0.0,
    max_value=3.0,    
    value = gamma,
    ini_value = gamma,
    Gauss_priors_width=1
    ))
like_params.append(LikelihoodParameter(
    name='mF',
    min_value=0.0,
    max_value=1.0,
    ini_value = mF,
    value = mF,
    Gauss_priors_width=0.5
    ))
like_params.append(LikelihoodParameter(
    name='Delta2_p',
    min_value=0.0,
    max_value=1.0,
    ini_value = linP_zs[0]['Delta2_p'],
    value = linP_zs[0]['Delta2_p']
    ))
like_params.append(LikelihoodParameter(
    name='n_p',
    min_value=0.0,
    max_value=1.0,
    ini_value = linP_zs[0]['n_p'],
    value = linP_zs[0]['n_p']
    ))
like_params.append(LikelihoodParameter(
    name='kF_Mpc',
    min_value=0.0,
    max_value=1.0,
    ini_value = kF_Mpc,
    value = kF_Mpc
    ))
like_params.append(LikelihoodParameter(
    name='lambda_P',
    min_value=0.0,
    max_value=1.0,
    ini_value = lambdap,
    value = lambdap
    ))

# %%
# Inputting the Arinyo params directly

# like_params = []

# like_params.append(LikelihoodParameter(
#     name='bias',
#     min_value=-1.0,
#     max_value=1.0,
#     value=-0.1071721,
#     ini_value=-0.3,
#     Gauss_priors_width=.5
#     ))
# like_params.append(LikelihoodParameter(
#     name='beta',
#     min_value=0.0,
#     max_value=3.0,    
#     value = 1.62794518,
#     ini_value = 2.3,
#     Gauss_priors_width=1
#     ))
# like_params.append(LikelihoodParameter(
#     name='q1',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.1112,
#     value = 0.24221474,
#     Gauss_priors_width=.5
#     ))
# like_params.append(LikelihoodParameter(
#     name='kvav',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.51890844,
#     value = 0.51890844
#     ))
# like_params.append(LikelihoodParameter(
#     name='av',
#     min_value=0.0,
#     max_value=2.0,
#     ini_value = 0.2694,
#     value = 0.39530736
#     ))
# like_params.append(LikelihoodParameter(
#     name='bv',
#     min_value=0.0,
#     max_value=2.0,
#     ini_value = 0.0002,
#     value = 1.7016952
#     ))
# like_params.append(LikelihoodParameter(
#     name='kp',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 14.38534737,
#     value = 14.38534737
#     ))

# %%
linP_zs[0]['Delta2_p']

# %%
linP_zs[0]['n_p']

# %%
like = Likelihood(data, theory_AA, free_param_names=["bias"], iz_choice=iz_choice, like_params=like_params)

# %%
like.plot_px(iz_choice, like_params, multiply_by_k=False, ylim2=[-1,1], ylim=[-.005,0.08], every_other_theta=True, show=True, title=f"Redshift {data.z[iz_choice]}", theorylabel='Theory: P18 (cosmo)\n + Chaves-Montero+25 (IGM)', datalabel=f'DR2 measurement', xlim=[0,.7], residual_to_theory=True)


# %%
.05*1.15

# %%
