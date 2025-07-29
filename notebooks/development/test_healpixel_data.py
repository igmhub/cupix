# %% [markdown]
# # PX measurements from healpixels
#
# ### Read archive, compute mean and covariance and make plots

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
from cupix.px_data import data_healpix, px_window, px_ztk

# %%
# hardcoded for now
basedir = '/Users/afont/Codes/cupix/data/px_measurements/Lyacolore/'
#fname = basedir + '/px-nhp_41-zbins_3-thetabins_7.hdf5'
fname = basedir + '/px-nhp_41-zbins_4-thetabins_40.hdf5'
print(fname) 

# %%
# load PX measurement from different healpixels
archive = data_healpix.HealpixPxArchive(fname)

# %%
# number of Fourier modes (should be equal to number of FFT points, at least for now)
Nk = len(archive.k_bins)
print(f'N_k = {Nk}')
N_fft = Nk
# pixel width, in Angstroms
pw_A = 0.8
# length of FFT grid, in Angstroms
L_A = N_fft * pw_A
print(f'L = {L_A} A')
# number of healpixels
N_hp = len(archive.list_hp)
print(f'Got {N_hp} healpixels')

# %%
# test code to compute mean PX (summing F_m, W_m from healpixels)
mean_px_v1 = archive.get_mean_px()

# %%
# test code to compute mean PX and covariance (using P_m, V_m from healpixels)
mean_px_v2 = archive.get_mean_and_cov()


# %%
def plot_px(iz, it):
    px_zt_v1 = mean_px_v1.list_px_z[iz].list_px_zt[it]
    px_zt_v2 = mean_px_v2.list_px_z[iz].list_px_zt[it]
    k = [k_bin.k for k_bin in archive.k_bins]
    for px in archive.list_px:
        px_zt = px.list_px_z[iz].list_px_zt[it]
        plt.plot(k, px_zt.P_m, alpha=0.1)
    plt.plot(k, px_zt_v1.P_m, label='mean v1')
    plt.plot(k, px_zt_v2.P_m, label='mean v2')
    plt.xlim(0,0.5)
    plt.legend()


# %%
plot_px(1,2)


# %%
def plot_weights(iz, it):
    px_zt_v1 = mean_px_v1.list_px_z[iz].list_px_zt[it]
    px_zt_v2 = mean_px_v2.list_px_z[iz].list_px_zt[it]
    k = [k_bin.k for k_bin in archive.k_bins]
    for px in archive.list_px:
        px_zt = px.list_px_z[iz].list_px_zt[it]
        plt.semilogy(k, px_zt.V_m, alpha=0.1)
    plt.semilogy(k, px_zt_v1.V_m, label='total weights v1')
    plt.semilogy(k, px_zt_v2.V_m, label='total weights v2')
    plt.xlim(0,0.5)
    plt.legend()


# %%
plot_weights(1,5)

# %%

# %%
