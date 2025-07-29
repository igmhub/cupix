# %% [markdown]
# # Compute covariance of rebinned PX measurement

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
# load PX measurement from different healpixels (before rebinning)
archive = data_healpix.HealpixPxArchive(fname)

# %%
# combine measurement from all healpixels, and covariance (before rebinning)
mean_px = archive.get_mean_and_cov()

# %%
# rebin all measurements in the original archive, create a new one
rebin_t_factor=4
rebin_k_factor=4
rebinned_archive = archive.rebin(
    rebin_t_factor=rebin_t_factor, 
    rebin_k_factor=rebin_k_factor
)

# %%
# get a new measurement combining all healpixels, after rebinning
mean_rebinned_px = rebinned_archive.get_mean_and_cov()


# %%
def compare_px(iz, rebinned_it):
    '''Plot the mean Px, before and after rebinning'''    
    
    # plot mean px, before rebinning
    k_m = [ k_bin.k for k_bin in mean_px.k_bins ]
    # range of original theta bins to plot 
    it_min = rebinned_it * rebin_t_factor
    it_max = (rebinned_it+1) * rebin_t_factor
    for px_zt in mean_px.list_px_z[iz].list_px_zt[it_min:it_max]:
        min_t = px_zt.t_bin.min_t
        max_t = px_zt.t_bin.max_t
        label=f'{px_zt.t_bin.min_t} < $\\theta$ < {px_zt.t_bin.max_t}'
        plt.plot(k_m, px_zt.P_m, alpha=0.5, label=label)

    # plot mean px, after rebinning
    k_m = [ k_bin.mean() for k_bin in mean_rebinned_px.k_bins ]
    px_zt = mean_rebinned_px.list_px_z[iz].list_px_zt[rebinned_it]
    min_t = px_zt.t_bin.min_t
    max_t = px_zt.t_bin.max_t
    label=f'{px_zt.t_bin.min_t} < $\\theta$ < {px_zt.t_bin.max_t}'
    yerr = np.sqrt(np.diagonal(px_zt.C_mn))
    plt.errorbar(k_m, px_zt.P_m, yerr=yerr, label=label)
    plt.legend()
    plt.xlim(0, 0.5)


# %%
compare_px(iz=0, rebinned_it=4)

# %%
compare_px(iz=1, rebinned_it=7)

# %%
