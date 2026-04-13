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
# Here we test different k ranges with a complex window matrix

# %%
import numpy as np
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.theory import Theory
import matplotlib.pyplot as plt
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import cupix
import h5py as h5
from lace.cosmo import cosmology
# %load_ext autoreload
# %autoreload 2
cupixpath = cupix.__path__[0].rsplit('/', 1)[0]

# %% [markdown]
# ### Step 1: Import a noiseless forecast

# %%
# choose the mode
mode = 'cosmo_igm' # if the parameters you want to test are mean-flux, etc
# mode = 'arinyo' # if the parameters you want to test are Arinyo params

# %%
forecast_file = f"{cupixpath}/data/px_measurements/forecast/forecast_ffcentral_real_binned_out_px-zbins_4-thetabins_9_w_res_noiseless_z0.hdf5"
# the forecast file contains the window matrix from the real data
forecast = DESI_DR2(forecast_file, kM_max_cut_AA=1)
# get default theory from forecast
with h5.File(forecast_file, 'r') as f:
    default_theory_label = f['metadata'].attrs['true_lya_theory']
print(f"Default theory label: {default_theory_label}")


# %% [markdown]
# ### Step 2: Load the theory

# %%
# Load emulator
zs = forecast.z
print(zs)

theories = []
cosmo = cosmology.Cosmology()
for z in zs:
    theories.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': True, 'default_lya_model':default_theory_label}))
                                                             


# %% [markdown]
# ### Now test the accuracy and timing for different k_m (small-k) bin cuts

# %%
iz_choice = 0
z_choice  = zs[iz_choice]

# %%
forecast_k1 = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=1)
forecast_k1p2 = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=1.2)
forecast_k2 = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=2)
forecast_k3 = DESI_DR2(forecast_file, kM_max_cut_AA=1, km_max_cut_AA=3)
forecast_untouched = DESI_DR2(forecast_file, kM_max_cut_AA=1)

# %%
like_k1 = Likelihood(data=forecast_k1, theory=theories[iz_choice], iz=iz_choice, verbose=True)
like_k1p5 = Likelihood(data=forecast_k1p2, theory=theories[iz_choice], iz=iz_choice, verbose=True)
like_k2 = Likelihood(data=forecast_k2, theory=theories[iz_choice], iz=iz_choice, verbose=True)
like_k3 = Likelihood(data=forecast_k3, theory=theories[iz_choice], iz=iz_choice, verbose=True)
like_untouched = Likelihood(data=forecast_untouched, theory=theories[iz_choice], iz=iz_choice, verbose=True)

# %%
k_cuts = [3,2,1.2,1]
# make a figure with residuals for the three k-cut cases
fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
theta_A_bin = 2
baseline = like_untouched.get_convolved_px()

for i, like in enumerate([like_k3, like_k2, like_k1p5, like_k1]):
    chi2 = like.get_chi2() # if you want to know the chi2 per theta bin, use return_all=True
    print("Chi2 is", chi2)
    Px_convolved = like.get_convolved_px()

    ax[0].plot(like.data.k_M_centers_AA, Px_convolved[theta_A_bin], label=rf"$k_m$ cut at {k_cuts[i]} $\AA^{{-1}}$")
    ax[1].plot(like.data.k_M_centers_AA, (Px_convolved[theta_A_bin]-baseline[theta_A_bin])/baseline[theta_A_bin], label=r"$k_m$ cut at 1$\AA^{-1}$")
ax[0].legend()
ax[0].set_ylabel(r"$P_\times$ convolved [$\AA$]")
ax[1].set_ylabel(r"Residuals to truth")
ax[1].set_xlabel(r"$k_M$ [$\AA^{-1}$]")
ax[1].set_ylim([-0.05, 0.05])
ax[1].axhline(.01, color='k', linestyle='--')
ax[1].axhline(-.01, color='k', linestyle='--')

# %% [markdown]
# We can limit the small-k values to 1.2A-1.

# %% [markdown]
#
