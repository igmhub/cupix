import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from cupix.rebin_cov.rebin_stats import compute_binned_stats_px
from cupix.

import yaml
import argparse
import pprint

parser = argparse.ArgumentParser(description='Computing mean and covariance of rebinned Px using config')
parser.add_argument('config',type=str,help='Path to YAML config file')
args = parser.parse_args()

with open(args.config,'r') as f:
    params = yaml.safe_load(f)

print('Parmeters used:')
pprint.pprint(params)

px_dr2 = compute_binned_stats_px(**params)

# Save the results 
#save_results_to_hdf5(px_dr2,out_path,out_filename)





