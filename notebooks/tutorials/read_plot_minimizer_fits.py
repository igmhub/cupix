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
#     display_name: cupix
#     language: python
#     name: cupix
# ---

# %% [markdown]
# # Tutorial: Plot saved results from the minimizer

# %%
import numpy as np
from cupix.inference.minimizer_funcs import plot_ellipse, plot_corner, load_mini_results
from cupix.likelihood.theory import Theory
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2


# %% [markdown]
# Set the zs and choice of z

# %%
zs = [2.2, 2.4, 2.6, 2.8]
z_choice = zs[iz]

# %%
include_xi_fits = True

# %%
results = []
freepars = []
theories = []
likes = []
for iz in [0,1,2,3]:
    results_directory = "/pscratch/sd/m/mlokken/desi-lya/px/mocks/minimizer_fits/20260914_trucont_bias_beta_kp/"
    results_dict, free_params, data, cosmo, theory, like, setup_config, inf_config = load_mini_results(results_directory, filename=f"iminuit_results_{iz}.npz", setup_config_fname=f"setup_config_mini_z{iz}.yaml")
    results.append(results_dict)
    freepars.append(free_params)
    theories.append(theory)
    likes.append(like)

# %%
plot_corner(results[0], freepars[0], show_truth=True)

# %%
for iz in [0,1,2,3]:
    if include_xi_fits:
        true_vals = {'bias':theories[iz].get_param('bias'), 'beta':theories[iz].get_param('beta'), 'kp':theories[iz].get_param('kp_Mpc')}
        print(true_vals)
        true_val_label = rf"$\xi_{{3D}}$ fit"
    else:
        true_vals = None
        true_val_label = None
    plot_ellipse(results[iz], 'bias','beta', color="green", label=f"z={zs[iz]}", true_vals=true_vals, true_val_label=true_val_label, title=f"z={zs[iz]}")
    print(results[iz]['prob'], results[iz]['chi2'])
    

# %%
