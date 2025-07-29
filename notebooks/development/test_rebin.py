# %% [markdown]
# # Rebin a PX measurement

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
from cupix.px_data import data_healpix, px_window, px_ztk, px_binning

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
# define new, coarser k bins
new_k_bins = px_binning.get_coarser_k_bins(archive.k_bins, rebin_factor=4, include_k_0=True)


# %% [markdown]
# ### Rebin in k

# %%
def sum_weights(pix, iz, it):
    test_px_zt = archive.list_px[pix].list_px_z[iz].list_px_zt[it]
    return np.sum(test_px_zt.V_m)


# %%
# find a good PX measurement to work with
iz=1
it=4
for pix in archive.list_hp:
    if sum_weights(pix, iz, it)>0:
        print(pix)
        ipix = pix
        break

# %%
# rebin only a redshift bin and theta bin
test_px_zt = archive.list_px[ipix].list_px_z[iz].list_px_zt[it]
new_px_zt = test_px_zt.rebin_k(rebin_factor=2, include_k_0=True)


# %%
def compare(px_zt_1, px_zt_2):
    k1 = [k_bin.k for k_bin in px_zt_1.k_bins]
    p1 = px_zt_1.P_m
    k2 = [k_bin.mean() for k_bin in px_zt_2.k_bins]
    p2 = px_zt_2.P_m    
    plt.plot(k1, p1, 'o')
    plt.plot(k2, p2, 'x')
    plt.xlim(0,0.1)


# %%
compare(test_px_zt, new_px_zt)

# %% [markdown]
# ## Rebin in theta

# %%
new_t_bins = px_binning.get_coarser_t_bins(archive.t_bins, rebin_factor=4)

# %%
test_px_z = archive.list_px[ipix].list_px_z[iz]

# %%
new_px_z = test_px_z.rebin_t(rebin_factor=4)


# %%
def compare(test_px_z, new_px_z, new_it, plot_V_m=False):
    rebin_factor=int(len(test_px_z.t_bins)/len(new_px_z.t_bins))
    print(rebin_factor)
    k_m = [k_bin.k for k_bin in test_px_zt.k_bins]
    for px_zt in test_px_z.list_px_zt[new_it*rebin_factor:(new_it+1)*rebin_factor]:
        if plot_V_m:
            plt.plot(k_m, px_zt.V_m, alpha=0.2)
        else:
            plt.plot(k_m, px_zt.P_m, alpha=0.2)
    if plot_V_m:
        plt.plot(k_m, new_px_z.list_px_zt[new_it].V_m)
    else:
        plt.plot(k_m, new_px_z.list_px_zt[new_it].P_m)
    plt.xlim([0,0.5])
    #plt.ylim([-0.01, 0.02])


# %%
compare(test_px_z, new_px_z, 4)

# %%
compare(test_px_z, new_px_z, 4, plot_V_m=True)

# %% [markdown]
# ### Rebin the entire PX measurement

# %%
# rebin the entire Px object
test_px = archive.list_px[ipix]
test_px.rebin_k(rebin_factor=4)
test_px.rebin_t(rebin_factor=4)
new_px = test_px.rebin(rebin_t_factor=4, rebin_k_factor=4, include_k_0=True)

# %%
# rebin only a redshift bin
test_px_z = archive.list_px[ipix].list_px_z[iz]
test_px_z.rebin_k(rebin_factor=4)
test_px_z.rebin_t(rebin_factor=4)
new_px_z = test_px_z.rebin(rebin_t_factor=4, rebin_k_factor=4, include_k_0=True)

# %%
