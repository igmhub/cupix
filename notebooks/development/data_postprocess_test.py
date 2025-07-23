# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
from collections import defaultdict

# %%
sys.path.append("/pscratch/sd/s/sindhu_s/Pcross_lya/LaCE/LaCE/ForestFlow/cupix/")

# %%
from cupix.rebin_cov.rebin_stats import compute_binned_stats_px

# %%
import yaml
import pprint

# %%
config_path = "/pscratch/sd/s/sindhu_s/Pcross_lya/LaCE/LaCE/ForestFlow/cupix/notebooks/development/config.yaml"

# %%
with open(config_path,'r') as f:
    params = yaml.safe_load(f)

# %%
print('Parmeters used:')
pprint.pprint(params)

# %%
px_dr2 = compute_binned_stats_px(**params)


# %% [markdown]
# plot the unbinned px for a few theta values

# %%
# extract keys 
zbin_to_keys = defaultdict(list)
for keys in px_dr2.px_avg.keys():
    zbin = keys[0]  # Extract the z_bin from the key tuple
    zbin_to_keys[zbin].append(keys)

# %%
zbin = 2.2
N_fft = px_dr2.N_fft
plt.errorbar(px_dr2.k_arr[:N_fft//2],px_dr2.px_avg[zbin_to_keys[zbin][35]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance[zbin_to_keys[zbin][35]][:N_fft//2])))
plt.errorbar(px_dr2.k_arr[:N_fft//2],px_dr2.px_avg[zbin_to_keys[zbin][15]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance[zbin_to_keys[zbin][15]][:N_fft//2])))
plt.errorbar(px_dr2.k_arr[:N_fft//2],px_dr2.px_avg[zbin_to_keys[zbin][0]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance[zbin_to_keys[zbin][0]][:N_fft//2])))
# if only binned in k and not theta, you can uncomment and directly compare: 
#plt.errorbar(px_dr2.k_bins[zbin_to_keys[zbin][35]][:N_fft//2],px_dr2.px_avg_bin[zbin_to_keys[zbin][35]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance_bin[zbin_to_keys[zbin][35]][:N_fft//2])),fmt='o')
#plt.errorbar(px_dr2.k_bins[zbin_to_keys[zbin][15]][:N_fft//2],px_dr2.px_avg_bin[zbin_to_keys[zbin][15]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance_bin[zbin_to_keys[zbin][15]][:N_fft//2])),fmt='o')
#plt.errorbar(px_dr2.k_bins[zbin_to_keys[zbin][0]][:N_fft//2],px_dr2.px_avg_bin[zbin_to_keys[zbin][0]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance_bin[zbin_to_keys[zbin][0]][:N_fft//2])),fmt='o')
#plt.xlim(0,1.0)

plt.ylabel(r'$P_{\times AM}^{z}$ [A]')
plt.xlabel(r"$k_\parallel$ [A$^{-1}$]")

# %%
# extract new keys 
zbin_to_rebinkeys = defaultdict(list)
for keys in px_dr2.px_avg_bin.keys():
    zbin = keys[0]  # Extract the z_bin from the key tuple
    zbin_to_rebinkeys[zbin].append(keys)
#print(px_dr2.k_bins.keys())

# %% [markdown]
# plot rebinned px for a few theta values

# %%
zbin = 2.2

N_fft = px_dr2.N_fft
plt.errorbar(px_dr2.k_bins[zbin_to_keys[zbin][9]][:N_fft//2],px_dr2.px_avg_bin[zbin_to_rebinkeys[zbin][9]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance_bin[zbin_to_rebinkeys[zbin][9]][:N_fft//2])),fmt='o')
plt.errorbar(px_dr2.k_bins[zbin_to_keys[zbin][5]][:N_fft//2],px_dr2.px_avg_bin[zbin_to_rebinkeys[zbin][5]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance_bin[zbin_to_rebinkeys[zbin][5]][:N_fft//2])),fmt='o')
plt.errorbar(px_dr2.k_bins[zbin_to_keys[zbin][0]][:N_fft//2],px_dr2.px_avg_bin[zbin_to_rebinkeys[zbin][0]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance_bin[zbin_to_rebinkeys[zbin][0]][:N_fft//2])),fmt='o')
plt.ylabel(r'$P_{\times AM}^{z}$ [A]')
plt.xlabel(r"$k_\parallel$ [A$^{-1}$]")
#plt.savefig('px_theta_binned_z2.2.png',dpi=400,bbox_inches='tight')

# %%
