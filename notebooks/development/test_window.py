# %% [markdown]
# # Compute window matrix
#
# ### Consider the impact of rebinning, and of averaging over healpixels 

# %%
import time
import numpy as np
import matplotlib.pyplot as plt

# %%
from cupix.px_data import data_healpix, px_window, px_ztk

# %%
# hardcoded for now
basedir = '/Users/afont/Codes/cupix/data/px_measurements/Lyacolore/'
#fname = basedir + '/px-nhp_41-zbins_3-thetabins_7.hdf5'
fname = basedir + '/px-nhp_41-zbins_4-thetabins_40.hdf5'
#fname = basedir + '/px-nhp_150-zbins_4-thetabins_40.hdf5'
print(fname) 

# %% [markdown]
# ### Computing the window matrices takes some time, and use a lot of memory

# %%
# load PX measurement from different healpixels
start = time.time()
archive = data_healpix.HealpixPxArchive(fname)
#archive = data_healpix.HealpixPxArchive(fname, compute_window=True)
end = time.time()
print('time spent =', end - start)


# %% [markdown]
# ### Play with the window of a particular (z,t) bin, of a healpixel

# %%
# find a good PX measurement to work with its window matrix
def sum_weights(pix, iz, it):
    test_px_zt = archive.list_px[pix].list_px_z[iz].list_px_zt[it]
    return np.sum(test_px_zt.V_m)
iz=1
it=4
for pix in archive.list_hp:
    if sum_weights(pix, iz, it)>0:
        print(pix)
        ipix = pix
        break

# %%
test_px_zt = archive.list_px[pix].list_px_z[iz].list_px_zt[it]
F_m = test_px_zt.F_m
W_m = test_px_zt.W_m
z_bin = test_px_zt.z_bin
t_bin = test_px_zt.t_bin
k_bins = test_px_zt.k_bins
L_A = archive.L_A
sig_A = archive.sig_A
start = time.time()

# %%
R2_m = px_window.compute_R2_m(k_bins, sig_A)

# %%
# %timeit px_window.compute_V_m(W_m, R2_m, L_A)

# %%
# %timeit px_window.compute_U_mn(W_m, R2_m, L_A)

# %%
new_px_zt = px_window.Px_zt_w.from_unnormalized(z_bin, t_bin, k_bins, F_m, W_m, L_A, sig_A, True)

# %%
new_px_zt.U_mn

# %%
plt.imshow(new_px_zt.U_mn, vmax=0.01)
plt.colorbar()

# %%
zoom=50
plt.imshow(new_px_zt.U_mn, vmax=0.01)
plt.xlim(0,zoom)
plt.ylim(0,zoom)
plt.colorbar()

# %% [markdown]
# ### Compute the average window, over healpixels

# %%
V_m, P_m, U_mn = archive.get_mean_window_zt(iz=0, it=4)

# %%
