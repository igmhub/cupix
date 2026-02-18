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
# ### Load forecast data

# %%
forecast_file = "/pscratch/sd/m/mlokken/desi-lya/px/data/px_measurements/forecast/forecast_ffcentral_binned_out_px-zbins_2-thetabins_18_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kmax_cut_AA=1)

# %%
iz_choice = np.array([0])
# Load emulator
z = forecast.z[iz_choice]
print(z)
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
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)

ffemu = FF_emulator(z, fid_cosmo, cc, Nrealizations=3000)
ffemu.kp_Mpc = 1 # set pivot point

theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu
# dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array(z), camb_results=cc)

# %%
param_names = ["bias","beta","q1","kvav","av","bv","kp","q2"]
arinyo_dict_central = {}
# read the likelihood params from forecast file
like_params = []
with h5.File(forecast_file, "r") as f:
    params = f['like_params']
    attrs = params.attrs
    for key in param_names: # important to get the sorting right
        if key in attrs:
            val = attrs[key][:]
            if len(val)>0:
                like_params.append(LikelihoodParameter(
                    name=key,
                    value=val[iz_choice],
                    ini_value= val[iz_choice], # not used for now
                    min_value=-1000,
                    max_value=1000, # not used for now
                ))
                arinyo_dict_central[key] = val[iz_choice]
# check the parameters
for p in like_params:
    print(p.name, p.value)

# %%
arinyo_dict_central

# %%
like = Likelihood(forecast, theory_AA, free_param_names=[], iz_choice=iz_choice, like_params=like_params, verbose=True)

# %%
like.get_log_like([lp.value for lp in like_params])

# %% [markdown]
# The following line shows that the function get_convolved_Px_AA takes the vast majority of the time. 

# %%
# %lprun -f like.get_log_like like.get_log_like([lp.value for lp in like_params])

# %% [markdown]
# The following line shows that get_px_AA takes the majority of time from get_convolved_Px_AA (1s) rather than convolve_window (.0015 s)

# %%
# %lprun -f like.get_convolved_Px_AA like.get_convolved_Px_AA(iz_choice, np.arange(len(forecast.theta_max_A_arcmin)), like_params) # compare to .897s total, 4 calls per get_Px

# %% [markdown]
# The following line shows that lyap3d.model_Px takes the most time (~1s)

# %%
# %lprun -f like.theory.get_px_AA like.theory.get_px_AA(z[iz_choice], forecast.k_m[iz_choice], [5], like_params)


# %% [markdown]
# ## Now let's test this with the emulator for cosmo+IGM parameters

# %%
forecast_file = "/global/common/software/desi/users/mlokken/cupix/data/px_measurements/forecast//forecast_ffcentral_cosmo_igm_binned_out_px-zbins_2-thetabins_18_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kmax_cut_AA=1)

# %%
iz_choice = np.array([0])
param_names = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
z = forecast.z[iz_choice]

# %%
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
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)

ffemu = FF_emulator(z, fid_cosmo, cc, Nrealizations=3000)
ffemu.kp_Mpc = 1 # set pivot point

theory_AA = set_theory(ffemu, k_unit='iAA', verbose=True)
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu
# dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array(z), camb_results=cc)

# %%
# read the likelihood params from forecast file
like_params_dict = {}
like_params = []
with h5.File(forecast_file, "r") as f:
    params = f['like_params']
    attrs = params.attrs
    for key in param_names: # important to get the sorting right
        if key in attrs:
            val = attrs[key][:]
            like_params_dict[key] = val[iz_choice][0] + np.random.uniform(-0.1,0.1)
            like_params.append(LikelihoodParameter(
                name=key,
                value=val[iz_choice] + np.random.uniform(-0.1,0.1), # add some noise to the initial value
                ini_value= val[iz_choice], # not used for now
                min_value=-1000,
                max_value=1000, # not used for now
            ))
# check the parameters
for p in like_params_dict:
    print(p, like_params_dict[p])


# %% [markdown]
# Test the emulator only

# %%
# %%time
ffemu.emu.predict_Arinyos(emu_params=like_params_dict, Nrealizations=3000, return_all_realizations=False)

# %%
like = Likelihood(forecast, theory_AA, free_param_names=[], iz_choice=iz_choice, like_params=like_params, verbose=True)

# %%

# %lprun -f like.minus_log_prob like.minus_log_prob([0, 0, 0, 0, 0, 0])

# %%

# %%
# %%time
like.get_log_like([lp.value+.5 for lp in like_params])

# %%
# %lprun -f like.get_log_like like.get_log_like([lp.value+.5 for lp in like_params])

# %% [markdown]
# The following cell shows that now it takes 1.3s for get_Px_AA

# %%
# %lprun -f like.get_convolved_Px_AA like.get_convolved_Px_AA(iz_choice, np.arange(len(forecast.theta_max_A_arcmin)), like_params) # compare to .897s total, 4 calls per get_Px

# %%
# %lprun -f like.theory.get_px_AA like.theory.get_px_AA(z[iz_choice], forecast.k_m[iz_choice], [5], like_params)

# %% [markdown]
# The emulator is taking .02s, the majority of time (1.02s) still spent in predicting Px

# %%
from cupix.likelihood import lyaP3D
p3d_model = ffemu.arinyo.P3D_Mpc
k_Mpc = np.linspace(0.01,0.1,len(forecast.k_m[0]))
lyap3d = lyaP3D.LyaP3D(z, p3d_model, arinyo_dict_central, Arinyo=ffemu.arinyo)


# %%
# %lprun -f lyap3d.model_Px lyap3d.model_Px(np.array([k_Mpc]), np.array([np.linspace(0,100, len(forecast.theta_max_A_arcmin))]))

# %%
from forestflow import pcross

# %%
# %lprun -f pcross.Px_Mpc_detailed pcross.Px_Mpc_detailed(z[iz_choice],np.array([k_Mpc]), np.array([np.linspace(0,100, len(forecast.theta_max_A_arcmin))]), p3d_model, P3D_params=arinyo_dict_central, fast_transition=True)

# %%
print("Top use of time goes to P3D_eval", 145862634.0*1e-9, "s, hankel", 333954697.0*1e-9, "s, CubicSpline", 158602017.0*1e-9, "s, another CubicSpline", 183005814.0*1e-9, "s")

# %%
forecast.U_ZaMn.shape

# %%
