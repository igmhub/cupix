# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: cupix
# ---

# %% [markdown]
# # Debug the Px measurement on Gaussian mocks (generated on-the-fly)

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.likelihood.test_theory import TestTheory
from cupix.likelihood import likelihood
from cupix.px_data.data_DESI_DR2 import DESI_DR2


# %% [markdown]
# ### Setup data

# %%
fname = '/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/gaussian/analysis-0/contaminated/bf3_binned_out_px-zbins_4-thetabins_20_w_res_wp1d.hdf5'
kM_max_cut_AA=0.2
km_max_cut_AA=1.1*kM_max_cut_AA
data = DESI_DR2(fname, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA)

# %%
# get the central value of each redshift bin, of length Nz
zs = data.z
# get a 1D array of central values of the measured k bins, of length Nk_M
k_M = data.k_M_centers_AA
# get two 1D arrays with the edges of each theta bin, of length Nt_A each
theta_A_min = data.theta_min_A_arcmin
theta_A_max = data.theta_max_A_arcmin

# %%
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = data.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_m={Nk_m}")
# rebinned values
Nz, Nt_A, Nk_M = data.Px_ZAM.shape
print(f"rebinned values: Nz={Nz}, Nt_A={Nt_A}, Nk_M={Nk_M}")


# %%
def plot_theta_bin(iz, it_M):
    label = r"${:.2f}' < \theta < {:.2f}'$".format(theta_A_min[it_M], theta_A_max[it_M])
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it_M]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_M]))
#    print(len(k_M), len(Px), len(sig_Px))
    plt.errorbar(k_M, Px, sig_Px, label=label)


# %%
def plot_z_bin(iz, its_M):
    for it_M in its_M:
        plot_theta_bin(iz=iz, it_M=it_M)
    plt.title('Test Px at z={:.1f}'.format(zs[iz]))
    plt.legend()
    plt.xlabel(r'$k_\parallel$ [1/A]')
    plt.ylabel(r'$P_\times(\theta, k_\parallel)$ [A]')


# %%
plot_z_bin(iz=2, its_M=range(5))

# %%
plot_z_bin(iz=0, its_M=range(15,20))

# %% [markdown]
# ### Set up theory

# %%
iz=0
z=data.z[iz]
theory = TestTheory(z=z)

# %%
# No need to average over theta, they all have the same Px 
like = likelihood.Likelihood(data=data, theory=theory, iz=iz, config={'verbose':True, 'N_theta_average':1})

# %%
model_px=like.get_convolved_px(params={})

# %%
like.plot_px(multiply_by_k=False, every_other_theta=True)

# %%
