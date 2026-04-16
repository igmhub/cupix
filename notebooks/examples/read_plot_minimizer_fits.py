# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial: Plot saved results from the minimizer

# %%
import numpy as np
from cupix.likelihood.new_minimizer import plot_ellipses
from cupix.likelihood.theory import Theory
import matplotlib.pyplot as plt


# %% [markdown]
# Set the zs and choice of z

# %%
zs = [2.2, 2.4, 2.6, 2.8]
iz = 0
z_choice = zs[iz]

# %%
include_xi_fits = True

# %%
if include_xi_fits:
    # optional: set up theory if you want the colore best-fit for comparison
    from lace.cosmo import cosmology
    theories_xi = []
    cosmo = cosmology.Cosmology()
    for z in zs:
        theories_xi.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False, 'default_lya_model':'best_fit_arinyo_from_colore'}))
                                                                

# %%

# Load results file
outfile = np.load(f"/global/common/software/desi/users/mlokken/cupix/data/fitter_results/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50_iminuit_{z_choice}.npz")

for key in outfile.keys():
    # print(key)
    # print(outfile[key])
    bias = outfile['bias']
    bias_err = outfile['bias_err']
    beta = outfile['beta']
    beta_err = outfile['beta_err']
    cov = outfile[f'cov']

if include_xi_fits:
    true_vals = {'bias':theories_xi[iz].lya_model.default_lya_params['bias'], 'beta':theories_xi[iz].lya_model.default_lya_params['beta']}
    true_val_label = "Laura fit"
else:
    true_vals = None
    true_val_label = None
plot_ellipses(bias, beta, 'bias', 'beta', bias_err, beta_err, cov, nsig=3, true_vals=true_vals, true_val_label=true_val_label) # you can also input xrange and yrange
plt.title(f"z = {z_choice}")


# %%
