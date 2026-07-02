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
from cupix.inference.minimizer_funcs import plot_ellipse, plot_corner, load_mini_results
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
    
    cosmo = cosmology.Cosmology()
    theory = Theory(z=z_choice, fid_cosmo=cosmo, config={'verbose': False, 'default_lya_model':'best_fit_arinyo_from_colore'})
    true_vals = {'bias':theory.lya_model.default_lya_params['bias'], 'beta':theory.lya_model.default_lya_params['beta']}
    true_val_label = rf"$\xi_{{3D}}$ fit"
else:
    true_vals = None
    true_val_label = None


# %%
# load mini results
# enter the path to directory
minires_fname = f"/global/common/software/desi/users/mlokken/cupix/data/fitter_results/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50_iminuit_{z_choice}.npz"

outfile = np.load(minires_fname)
# turn outfile into a dictionary
results_dict = {key: outfile[key] for key in outfile.files}


# %%
plot_ellipse(results_dict, 'bias','beta', true_vals=true_vals, true_val_label=true_val_label)
plt.title(f"z = {z_choice}")

# %%
