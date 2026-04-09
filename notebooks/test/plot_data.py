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
# # Use iminuit to fit Px from DESI DR2

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
#from cupix.likelihood.likelihood_parameter import LikelihoodParameter, like_parameter_by_name
from cupix.likelihood.new_likelihood import Likelihood
from cupix.likelihood.theory import Theory
from cupix.likelihood.new_minimizer import IminuitMinimizer

# %% [markdown]
# ## Step 1: Read the data from DESI DR2 and plot it

# %%
fname = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5"
data = DESI_DR2(fname, kM_max_cut_AA=1, km_max_cut_AA=1.2)

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
    plt.title('DESI DR2 at z={:.1f}'.format(zs[iz]))
    plt.legend()
    plt.xlabel(r'$k_\parallel$ [1/A]')
    plt.ylabel(r'$P_\times(\theta, k_\parallel)$ [A]')


# %%
plot_z_bin(iz=0, its_M=range(5))

# %%
plot_z_bin(iz=0, its_M=range(5,10))


# %%
def plot_z_bin_two_panels(iz):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plot_z_bin(iz, its_M=range(5))
    plt.subplot(1, 2, 2)
    plot_z_bin(iz, its_M=range(5,10))
    plt.tight_layout(w_pad=3)


# %%
for iz in range(4):
    plot_z_bin_two_panels(iz=iz)

# %% [markdown]
# ## Step 2: setup theory objects, with and without contaminants (one per z)

# %%
# define fiducial cosmo
cosmo = cosmology.Cosmology()

# %%
theories_lya = []
theories_cont = []
for z in zs:
    theories_lya.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False}))
    theories_cont.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False, 
                                                            'include_hcd': True, 'include_metal': True,
                                                            'include_sky': True, 'include_continuum': True} ))

# %% [markdown]
# ## Step 3: create Likelihoods and compare data vs theory (no fits)

# %%
likes_lya = []
likes_cont = []
for iz, z in enumerate(zs):
    likes_lya.append(Likelihood(data=data, theory=theories_lya[iz], iz=iz, verbose=False))
    likes_cont.append(Likelihood(data=data, theory=theories_cont[iz], iz=iz, verbose=False))

# %%
models_lya = []
models_cont = []
for iz, z in enumerate(zs):
    models_lya.append(likes_lya[iz].get_convolved_px(params={}))
    models_cont.append(likes_cont[iz].get_convolved_px(params={}))


# %%
def compare_theta_bin(iz, it_M):
    plt.title(r"DESI DR2,   z = {:.1f},   ${:.2f}' < \theta < {:.2f}'$".format(
                                                zs[iz], theta_A_min[it_M], theta_A_max[it_M]))
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it_M]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_M]))
    plt.errorbar(k_M, Px, sig_Px, label='data')    
    plt.plot(k_M, models_lya[iz][it_M], label='Lya only')
    plt.plot(k_M, models_cont[iz][it_M], label='Lya + cont')
    plt.legend()
    plt.xlabel(r'$k_\parallel$ [1/A]')
    plt.ylabel(r'$P_\times(\theta, k_\parallel)$ [A]')
    plt.axhline(y=0, ls=':', color='gray')


# %%
# one z, multiple theta
for it_M in [2, 4, 6, 8]:
    plt.figure()
    compare_theta_bin(iz=2, it_M=it_M)

# %%
# one theta, multiple z
for iz, z in enumerate(zs):
    plt.figure()
    compare_theta_bin(iz=iz, it_M=0)

# %%
