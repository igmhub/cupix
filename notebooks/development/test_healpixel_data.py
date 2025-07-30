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
#fname = basedir + '/px-nhp_41-zbins_4-thetabins_40.hdf5'
fname = basedir + '/px-nhp_150-zbins_4-thetabins_40.hdf5'
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
def compare_means(iz, it):
    px_zt_v1 = mean_px_v1.list_px_z[iz].list_px_zt[it]
    px_zt_v2 = mean_px_v2.list_px_z[iz].list_px_zt[it]
    k = [k_bin.k for k_bin in archive.k_bins]
    plt.plot(k, px_zt_v1.P_m, label='mean v1')
    plt.plot(k, px_zt_v2.P_m, label='mean v2')
    plt.xlim(0,0.5)
    plt.legend()
#    plt.ylim(-0.1, 0.2)


# %%
compare_means(1,8)


# %%
def plot_weights(iz, it):
    px_zt_v1 = mean_px_v1.list_px_z[iz].list_px_zt[it]
    px_zt_v2 = mean_px_v2.list_px_z[iz].list_px_zt[it]
    k = [k_bin.k for k_bin in archive.k_bins]
    for px in archive.list_px:
        px_zt = px.list_px_z[iz].list_px_zt[it]
#        plt.semilogy(k, px_zt.V_m, alpha=0.1)
    plt.semilogy(k, px_zt_v1.V_m, label='total weights v1')
    plt.semilogy(k, px_zt_v2.V_m, label='total weights v2')
    plt.xlim(0,0.5)
    plt.legend()


# %%
plot_weights(1,5)

# %% [markdown]
# ## Rebin the mean measurement

# %%
rebin_8_4 = mean_px_v1.rebin(rebin_t_factor=8, rebin_k_factor=4, include_k_0=True)
rebin_4_4 = mean_px_v1.rebin(rebin_t_factor=4, rebin_k_factor=4, include_k_0=True)
rebin_2_4 = mean_px_v1.rebin(rebin_t_factor=2, rebin_k_factor=4, include_k_0=True)
rebin_1_4 = mean_px_v1.rebin(rebin_t_factor=1, rebin_k_factor=4, include_k_0=True)
# k bin 
k_m = [ k_bin.mean() for k_bin in rebin_4_4.k_bins ]
print(len(k_m))


# %%
def compare_theta(iz=0, theta_min=1.0, theta_max=2.0):
    for px in [rebin_8_4, rebin_4_4, rebin_2_4, rebin_1_4]:
        for px_zt in px.list_px_z[iz].list_px_zt:
            min_t = px_zt.t_bin.min_t
            max_t = px_zt.t_bin.max_t
            mean_t = px_zt.t_bin.mean()
            if mean_t < theta_max and mean_t > theta_min:
                label=f'{px_zt.t_bin.min_t} < $\\theta$ < {px_zt.t_bin.max_t}'
                plt.plot(k_m, px_zt.P_m, label=label)
    plt.legend()
    plt.xlim(0, 0.5)


# %%
compare_theta(iz=0, theta_min=1.0, theta_max=2.0)

# %%
compare_theta(iz=1, theta_min=5.0, theta_max=10.0)

# %% [markdown]
# ## Rebin the entire archive, then compute the mean

# %%
archive_4_4 = archive.rebin(rebin_t_factor=4, rebin_k_factor=4, include_k_0=True)

# %%
mean_4_4 = archive_4_4.get_mean_and_cov()


# %%
def compare_means(iz, it):
    k_m = [ k_bin.mean() for k_bin in rebin_4_4.k_bins ]
    px_zt_1 = rebin_4_4.list_px_z[iz].list_px_zt[it]
    plt.plot(k_m, px_zt_1.P_m, label='first average, then rebin')
    px_zt_2 = mean_4_4.list_px_z[iz].list_px_zt[it]
    plt.plot(k_m, px_zt_2.P_m, label='first rebin, then average')
    plt.xlim(0,0.8)
    z_bin = rebin_4_4.z_bins[iz]
    t_bin = rebin_4_4.t_bins[it]
    plt.title(f"{z_bin.min_z} < z < {z_bin.max_z}    ,    {t_bin.min_t}' < theta < {t_bin.max_t}' ")
    plt.legend()


# %%
compare_means(iz=2,it=6)

# %%
compare_means(iz=1,it=0)


# %% [markdown]
# ### Plot mean PX and errorbars

# %%
def plot_px(mean_px, iz, it):
    k_m = [ k_bin.mean() for k_bin in mean_px.k_bins ]
    px_zt = mean_px.list_px_z[iz].list_px_zt[it]
    yerr = np.sqrt(np.diagonal(px_zt.C_mn))
    plt.errorbar(k_m, px_zt.P_m, yerr=yerr)
    plt.xlim(0,0.8)
    z_bin = mean_px.z_bins[iz]
    zlabel = f"{z_bin.min_z:.2f} < z < {z_bin.max_z:.2f}"
    t_bin = mean_px.t_bins[it]
    tlabel = f"{t_bin.min_t:.3f}' < theta < {t_bin.max_t:.3f}'"
    plt.title(zlabel + '    ,    ' + tlabel)
    plt.axhline(y=0, color='gray', ls=':')


# %%
plot_px(mean_4_4, iz=0, it=2)

# %%
plot_px(mean_4_4, iz=1, it=5)

# %%
