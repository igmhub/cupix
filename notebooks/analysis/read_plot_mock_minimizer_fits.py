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

# %%
# errors from 'single mock' without /sqrt(50) rescaling on the mocks
results_single_mock = []
freepars_single_mock = []
theories_single_mock = []
likes_single_mock = []
for iz in [0,1,2,3]:
    results_directory = "/pscratch/sd/m/mlokken/desi-lya/px/mocks/minimizer_fits/20260914_trucont_bias_beta_kp/"
    results_dict, free_params, data, cosmo, theory, like, setup_config, inf_config = load_mini_results(results_directory, filename=f"iminuit_results_{iz}.npz", setup_config_fname=f"setup_config_mini_z{iz}.yaml", inf_config_fname=f"inference_config_mini.yaml")
    results_single_mock.append(results_dict)
    freepars_single_mock.append(free_params)
    theories_single_mock.append(theory)
    likes_single_mock.append(like)

# %%
results_uncont = []
freepars_uncont = []
theories_uncont = []
likes_uncont = []

results_trucont = []
freepars_trucont = []
theories_trucont = []
likes_trucont = []

for iz in [0,1,2,3]:
    # # input the results from minimizer_pipeline.py here.
    # results_directory = "/pscratch/sd/m/mlokken/desi-lya/px/mocks/minimizer_fits/20260915_uncont_fix_cont/"
    # results_dict, free_params, data, cosmo, theory, like, setup_config, inf_config = load_mini_results(results_directory, filename=f"iminuit_results_{iz}.npz", setup_config_fname=f"setup_config_mini_z{iz}.yaml")
    # results_uncont.append(results_dict)
    # freepars_uncont.append(free_params)
    # theories_uncont.append(theory)
    # likes_uncont.append(like)

    results_directory = "/pscratch/sd/m/mlokken/desi-lya/px/mocks/minimizer_fits/20260918_trucont_bias_beta_kp/"
    results_dict, free_params, data, cosmo, theory, like, setup_config, inf_config = load_mini_results(results_directory, filename=f"iminuit_results_{iz}.npz", setup_config_fname=f"setup_config_mini_z{iz}.yaml", inf_config_fname=f"inference_config_mini_z{iz}.yaml")
    results_trucont.append(results_dict)
    freepars_trucont.append(free_params)
    theories_trucont.append(theory)
    likes_trucont.append(like)

# %%
plot_corner(results_trucont[1], freepars_trucont[1], show_truth=False)

# %%
xranges = []
for iz in [0,1,2,3]:
    true_vals = {'bias':theories_trucont[iz].get_param('bias'), 'beta':theories_trucont[iz].get_param('beta'), 'kp':theories_trucont[iz].get_param('kp_Mpc')}
    true_val_label = r"$\xi_{{3D}}$ fit"
    ax1 = plot_ellipse(results_trucont[iz], 'bias','beta', true_vals=true_vals, true_val_label=true_val_label, color="green", label="True-continuum", title=f"z={zs[iz]}", ylabel=r'$\beta$', xlabel='$b$', xlim=[-.13,-.09], ylim=[1.5,1.9])
    # plot_ellipse(results_uncont[iz], 'bias','beta', ax=ax1, color="orange", label="Fitted-continuum", latex_label_y=r'\beta', latex_label_x='b')
    # plot a contour of the single mock results
    plot_ellipse(results_single_mock[iz], 'bias','beta', ax=ax1, color="blue", ylabel=r'$\beta$', xlabel='$b$', linestyle='dashed', fill=False, nsig=1, include_point=False, label=r'DR2-like 1$\sigma$')


# %%
import matplotlib as mpl
mpl.rcParams.update({'font.size': 15})

# %%
# plot that as a four-panel figure
fig, ax = plt.subplots(2, 2, figsize=(11, 9), sharex=False)
# set font size

for iz in range(4):
    true_vals = {'bias':theories_trucont[iz].get_param('bias'), 'beta':theories_trucont[iz].get_param('beta'), 'kp':theories_trucont[iz].get_param('kp_Mpc')}
    
    ax_i = ax[iz//2, iz%2]
    label1 = None
    label2 = None
    true_label = None
    latex_label_x = ""
    latex_label_y = ""
    if iz ==0:
        label1 = "True-\ncontinuum"
        label2 = "Fitted-\ncontinuum"
        true_label = r"$\xi_{{3D}}$ fit"
        latex_label_y = r'$\beta$'
    elif iz==3 or iz==4:
        latex_label_x = "b"
    elif iz==2:
        latex_label_y = r'$\beta$'
        latex_label_x = "b"
    plot_ellipse(results_trucont[iz], 'bias','beta',ax=ax_i, color="green", label=label1, true_vals=true_vals, true_val_label=true_label, title=f"z={zs[iz]}", latex_label_y=latex_label_y, latex_label_x=latex_label_x)
    plot_ellipse(results_uncont[iz], 'bias','beta', ax=ax_i, color="orange", label=label2, latex_label_y=latex_label_y, latex_label_x=latex_label_x)
plt.tight_layout()


# %%
likes[0].plot_px()

# %%
