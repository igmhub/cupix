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
# # k range tests
# ### Limiting the k range used in Px computations would speed up the code, but would lose the effects from the window matrix at high-k.
# Here we test whether, with a complex window matrix, 

# %%
import numpy as np
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from lace.cosmo import camb_cosmo
import matplotlib.pyplot as plt
from cupix.likelihood.likelihood_parameter import LikelihoodParameter, par_index, dict_from_likeparam
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import h5py as h5
import cupix
import pandas as pd
cupixpath = cupix.__path__[0].rsplit('/', 1)[0]
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### Step 1: Import a noiseless forecast

# %%
# choose the mode
mode = 'cosmo_igm' # if the parameters you want to test are mean-flux, etc
# mode = 'arinyo' # if the parameters you want to test are Arinyo params

# %%
forecast_file = f"{cupixpath}/data/px_measurements/forecast//forecast_ffcentral_igm_real_binned_out_px-zbins_4-thetabins_9_w_res_noiseless.hdf5"
# forecast_file = f"../../data/px_measurements/forecast/forecast_ffcentral_{mode}_real_binned_out_px-zbins_4-thetabins_9_w_res_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kM_max_cut_AA=1)
# get default theory from forecast
with h5.File(forecast_file, 'r') as f:
    default_theory_label = f['metadata'].attrs['true_lya_theory']
print(f"Default theory label: {default_theory_label}")


# %% [markdown]
# ### Step 2: Load the emulator with Nrealizations = 1000

# %%
# Load emulator
z = forecast.z
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
cosmo = {
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

theory = set_theory(z, cosmo_dict=cosmo, default_theory=default_theory_label, p3d_label='arinyo', emulator_label='forestflow_emu', k_unit='iAA', verbose=True)


# %% [markdown]
# ### Now test the accuracy and timing for different k_m (small-k) bin cuts

# %%
iz_choice = 0
z_choice  = z[iz_choice]

# %%
forecast_k1 = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=1)
forecast_k1p2 = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=1.2)
forecast_k2 = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=2)
forecast_k3 = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=3)
forecast_untouched = DESI_DR2(forecast_file)

# %%
z_choice = 2.25

# %%
like_k1 = Likelihood(forecast_k1, theory, z=z_choice, verbose=True)
like_k1p5 = Likelihood(forecast_k1p2, theory, z=z_choice, verbose=True)
like_k2 = Likelihood(forecast_k2, theory, z=z_choice, verbose=True)
like_k3 = Likelihood(forecast_k3, theory, z=z_choice, verbose=True)
like_untouched = Likelihood(forecast_untouched, theory, z=z_choice, verbose=True)

# %%
k_cuts = [3,2,1.2,1]
# make a figure with residuals for the three k-cut cases
fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
theta_A_bin = 2
i = 0
for like in [like_k3, like_k2, like_k1p5, like_k1]:
    chi2 = like.get_chi2() # if you want to know the chi2 per theta bin, use return_all=True
    print("Chi2 is", chi2)
    Px_convolved = like.get_convolved_Px_AA(theta_A=theta_A_bin)
    print(Px_convolved.shape)
    ax[0].plot(like.data.k_M_centers_AA, Px_convolved[0], label=rf"$k_m$ cut at {k_cuts[i]} $\AA^{{-1}}$")
    ax[1].plot(like.data.k_M_centers_AA, (Px_convolved[0]-like.data.Px_ZAM[0][theta_A_bin][:len(Px_convolved[0])])/like.data.Px_ZAM[0][theta_A_bin][:len(Px_convolved[0])], label=r"$k_m$ cut at 1$\AA^{-1}$")
    i +=1
    # ax[1].plot(like_k1.data.k_M_centers_AA, like.Px_AA[theta_A_bin] - Px_convolved, label=rf"{like.data.km_max_cut_AA} $k_m$ max cut")
    # ax[2].plot(like_k1.data.k_M_centers_AA, chi2[theta_A_bin], label=rf"{like.data.km_max_cut_AA} $k_m$ max cut")
ax[0].legend()
ax[0].set_ylabel(r"$P_\times$ convolved [$\AA$]")
ax[1].set_ylabel(r"Residuals to truth")
ax[1].set_xlabel(r"$k_M$ [$\AA^{-1}$]")
ax[1].set_ylim([-0.05, 0.05])
ax[1].axhline(.01, color='k', linestyle='--')
ax[1].axhline(-.01, color='k', linestyle='--')

# %% [markdown]
# We should limit the small-k values to 2A-1.

# %% [markdown]
#
