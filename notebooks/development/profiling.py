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

# %%
import numpy as np
from cupix.likelihood.generate_fake_data import FakeData
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from lace.cosmo import camb_cosmo, fit_linP
import matplotlib.pyplot as plt
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import scipy
import os
from forestflow.archive import GadgetArchive3D
import forestflow
from lace.cosmo.thermal_broadening import thermal_broadening_kms
import h5py as h5
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer, save_analysis_npz
import cupix
from pathlib import Path
import time

# # %load_ext autoreload
# # %autoreload 2
# %load_ext line_profiler

# %% [markdown]
# ### Load some data

# %%
data_file = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/binned_out_px-zbins_4-thetabins_9_w_res.hdf5"

data = DESI_DR2(data_file, kmax_cut_AA=1)

# %%
z = data.z
cosmo = {'H0': 67}
# default_theory options are:
# 'best_fit_arinyo_from_p1d': best fit to the DESI DR1 P1D data from Chaves+2026
# 'best_fit_igm_from_p1d': same but for IGM parameters
# 'best_fit_arinyo_from_colore': best fit to xi from colore mocks. Only works for z=2.2, 2.4, 2.6, 2.8
theory_p1d = set_theory(z, bkgd_cosmo=cosmo, default_theory='best_fit_igm_from_p1d', p3d_label='arinyo', emulator_label='forestflow_emu', k_unit='iAA', verbose=True)

# %%
params = {'mF': 1.}
params_with_rescaling = {'mF': 1., "Delta_p": 0.5}

# %%
z_choice = 2.4
like = Likelihood(data, theory_p1d, z=z_choice, verbose=True)

# %% [markdown]
# The following line shows that the function get_convolved_Px_AA takes the vast majority of the time. 

# %%
# %lprun -f like.get_log_like like.get_log_like()

# %% [markdown]
# The following line shows that get_px_AA takes the majority of time from get_convolved_Px_AA (1s) rather than convolve_window (.0015 s)

# %%
# %lprun -f like.get_convolved_Px_AA like.get_convolved_Px_AA(np.arange(len(data.theta_max_A_arcmin)), like_params=params) # compare to .897s total, 4 calls per get_Px

# %% [markdown]
# The following line shows that lyap3d.model_Px takes the most time (~1s)

# %%
# %lprun -f like.theory.get_px_AA like.theory.get_px_AA(z[iz_choice], forecast.k_m[iz_choice], [5], like_params)


# %%
forecast_file = "../../data/px_measurements/forecast/forecast_ffcentral_cosmo_igm_binned_out_px-zbins_2-thetabins_18_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kmax_cut_AA=1)
z_choice_temp = 2.25 # I just want to get some parameter values so I'll pretend this is the same as my z_choice from above
iz_choice = np.where(forecast.z == z_choice_temp)[0][0]

# read the likelihood params from forecast file
like_params_dict = {}
like_params = []
with h5.File(forecast_file, "r") as f:
    params = f['like_params']
    attrs = params.attrs
    for key in attrs:
        val = attrs[key][:]
        like_params_dict[key] = val[iz_choice] + np.random.uniform(-0.1,0.1)
# check the parameters
for p in like_params_dict:
    print(p, like_params_dict[p])


# %% [markdown]
# Test the emulator only

# %%
# %%time
theory_p1d.emulator.emu.predict_Arinyos(emu_params=like_params_dict, Nrealizations=3000, return_all_realizations=False)

# %%

# %lprun -f like.minus_log_prob like.minus_log_prob([0, 0, 0, 0, 0, 0])

# %%
# %%time
like.get_log_like()

# %% [markdown]
# The emulator is taking .02s, the majority of time (1.02s) still spent in predicting Px

# %%
forecast_file_arinyo = "../../data/px_measurements/forecast/forecast_ffcentral_arinyo_binned_out_px-zbins_2-thetabins_18_noiseless.hdf5"
z_choice_temp = 2.25 # I just want to get some parameter values so I'll pretend this is the same as my z_choice from above
iz_choice = np.where(forecast.z == z_choice_temp)[0][0]

# read the arinyo params from forecast file
arinyo_dict_central = {}

with h5.File(forecast_file_arinyo, "r") as f:
    params = f['like_params']
    attrs = params.attrs
    for key in attrs:
        val = attrs[key][:]
        arinyo_dict_central[key] = np.array([val[iz_choice]])
# check the parameters
for p in arinyo_dict_central:
    print(p, arinyo_dict_central[p])

# %%
theory_p1d.p3d_model

# %%
from cupix.likelihood import lyaP3D
p3d_model = theory_p1d.p3d_model
p3d_fun   = p3d_model.P3D_Mpc_k_mu
k_Mpc = np.linspace(0.01,0.1,len(forecast.k_m[0]))
lyap3d = lyaP3D.LyaP3D([z_choice], P3D_model=p3d_model, P3D_fun=p3d_fun, P3D_coeffs=arinyo_dict_central)


# %%
# %lprun -f lyap3d.model_Px lyap3d.model_Px(np.array([k_Mpc]), np.array([np.linspace(0,100, len(forecast.theta_max_A_arcmin))]))

# %%
from forestflow import pcross

# %%
pcross.Px_Mpc_detailed(k

# %%
# %lprun -f pcross.Px_Mpc_detailed pcross.Px_Mpc_detailed(z=z_choice,kpar_iMpc=np.array([k_Mpc]),rperp_Mpc=np.array([np.linspace(0,100, len(forecast.theta_max_A_arcmin))]), p3d_fun_Mpc=p3d_model.P3D_Mpc_k_mu, p3d_params=arinyo_dict_central)

# %%
