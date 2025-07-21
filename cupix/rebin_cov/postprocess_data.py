import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from cupix.rebin_cov.rebin_stats import compute_binned_stats_px

import yaml
import argparse
import pprint

parser = argparse.ArgumentParser(description='Computing mean and covariance of rebinned Px using config')
parser.add_argument('config',type=str,help='Path to YAML config file')
args = parser.parse_args()

with open(args.config,'r') as f:
    params = yaml.safe_load(f)

print(''Parmeters used:')
pprint.pprint(params)

px_dr2 = compute_binned_stats_px(**params)


# extract new keys 
zbin_to_rebinkeys = defaultdict(list)
for keys in px_dr2.px_avg_bin.keys():
    zbin = keys[0]  # Extract the z_bin from the key tuple
    zbin_to_rebinkeys[zbin].append(keys)
#print(px_dr2.k_bins.keys())

# plot rebinned px for a few theta values

zbin = 2.2
N_fft = px_dr2.N_fft
plt.errorbar(px_dr2.k_bins[:N_fft//2],px_dr2.px_avg_bin[zbin_to_rebinkeys[zbin][9]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance_bin[zbin_to_rebinkeys[zbin][9]][:N_fft//2])),fmt='o')
plt.errorbar(px_dr2.k_bins[:N_fft//2],px_dr2.px_avg_bin[zbin_to_rebinkeys[zbin][5]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance_bin[zbin_to_rebinkeys[zbin][5]][:N_fft//2])),fmt='o')
plt.errorbar(px_dr2.k_bins[:N_fft//2],px_dr2.px_avg_bin[zbin_to_rebinkeys[zbin][0]][:N_fft//2],yerr=np.sqrt(np.diag(px_dr2.covariance_bin[zbin_to_rebinkeys[zbin][0]][:N_fft//2])),fmt='o')

plt.ylabel(r'$P_{\times AM}^{z}$ [A]')
plt.xlabel(r"$k_\parallel$ [A$^{-1}$]")
plt.savefig('px_theta_binned_z2.2.png',dpi=400,bbox_inches='tight')
    



