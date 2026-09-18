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
# Plot saved results from the minimizer

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
results = []
freepars = []
theories = []
likes = []
for iz in [0,1,2,3]:
    results_directory = "/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/20260705_july_baseline/"
    results_dict, free_params, data, cosmo, theory, like, setup_config, inf_config = load_mini_results(results_directory, filename=f"iminuit_results_{iz}.npz", setup_config_fname=f"setup_config_mini_{iz}.yaml")
    results.append(results_dict)
    freepars.append(free_params)
    theories.append(theory)
    likes.append(like)

# %%
results_wq1 = []
freepars_wq1 = []
theories_wq1 = []
likes_wq1 = []
for iz in [0]:
    results_directory = "/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/20260713_july_baseline_plusq1/"
    results_dict, free_params, data, cosmo, theory, like, setup_config, inf_config = load_mini_results(results_directory, filename=f"iminuit_results_{iz}.npz", setup_config_fname="setup_config_mini.yaml")
    results_wq1.append(results_dict)
    freepars_wq1.append(free_params)
    theories_wq1.append(theory)
    likes_wq1.append(like)

# %%
plot_corner(results_wq1[0], freepars_wq1[0], show_truth=False)

# %%

likes_wq1[0].plot_px(params={'bias': results_wq1[0]['bias'], 'beta': results_wq1[0]['beta'], 'b_H': results_wq1[0]['b_H'], 'b_X': results_wq1[0]['b_X']}, multiply_by_k=False, include_probability=True,  title=f"z={zs[0]}", datalabel='DESI DR2', theorylabel="Best-fit theory")

# %%
for iz in [0, 1, 2,3]:
    plot_corner(results[iz-1], freepars[iz-1], show_truth=False)

# %%
ax = plot_ellipse(results[0], 'bias','beta', label=f"z={zs[0]}", color="C0")
for iz in [1, 2,3]:
    results_dict = results[iz]
    plot_ellipse(results_dict, 'bias','beta', ax=ax, color=f"C{iz}", label=f"z={zs[iz]}")


# %%
z_hiram = [2.13, 2.4, 2.81]
hiram_bias = [-0.0703, -0.1428, -0.2286]
hiram_bias_err = [.01, .01, .01]
hiram_b_hcd = [-0.061, -0.024, -0.0102]
hiram_bhcd_err = [.01, .01, .01]
total_bias_hiram = [-0.1317,-0.1670, -0.2388]
total_bias_err_hiram = [0.0035, 0.0045, 0.0045]
hiram_beta = [2.25, 1.469, 1.208]
hiram_beta_err = [0.3, .1, .55]

# %%
# get balpha index
balpha_i = [i for i in range(len(free_params)) if free_params[i].name == "bias"]
beta_i = [i for i in range(len(free_params)) if free_params[i].name == "beta"]

# %%
results

# %%
beta_alpha = np.array([res['beta'] for res in results])
beta_alpha_err = np.array([res['beta_err'] for res in results])
bH   = np.array([res['b_H'] for res in results])
bH_err = np.array([res['b_H_err'] for res in results])
balpha = np.array([res['bias'] for res in results])
balpha_err = np.array([res['bias_err'] for res in results])
balpha_i = [i for i in range(len(free_params)) if free_params[i].name == 'bias'][0]
beta_H = [theories[i].get_param('beta_H') for i in range(len(theories))]
cov_balpha_bH = np.array([res['cov'][0,3] for res in results])
bias_prime = balpha + bH
bias_prime_err = np.array(np.sqrt(balpha_err**2 + bH_err**2 + 2*cov_balpha_bH))
beta_alpha_prime = (balpha*beta_alpha + bH*beta_H) / bias_prime
beta_alpha_prime_relerr = bias_prime_err/bias_prime + beta_alpha * balpha_err/balpha + beta_H * bH_err/bH
beta_prime_hiram = np.array([1.423, 1.322, 1.176])
beta_prime_err_hiram = np.array([0.048, 0.045, 0.046])


# %%
plt.errorbar(zs, beta_alpha, yerr=beta_alpha_err, fmt='o', color='black',label=r'$P_\times$')
plt.errorbar(z_hiram, hiram_beta, yerr=hiram_beta_err, color='red', fmt='o', label='Herrera-Alcantar+26')
plt.ylabel(r"$\beta_\alpha$")
plt.xlabel(r"$z$")
plt.legend()

# %%
plt.errorbar(zs, balpha, yerr=balpha_err, fmt='o', color='black',label=r'$P_\times$')
plt.errorbar(z_hiram, hiram_bias, yerr=hiram_bias_err, color='red', fmt='o', label='Herrera-Alcantar+26')
plt.ylabel(r"$b_\alpha$")
plt.xlabel(r"$z$")
plt.legend()

# %%
plt.errorbar(zs, bH, yerr=bH_err, fmt='o', color='black',label=r'$P_\times$')
plt.errorbar(z_hiram, hiram_b_hcd, yerr=hiram_bhcd_err, color='red', fmt='o', label='Herrera-Alcantar+26')
plt.ylabel(r"$b_H$")
plt.xlabel(r"$z$")
plt.legend()

# %%
import matplotlib
font = {'size'   : 18}

matplotlib.rc('font', **font)

plt.errorbar(zs, bias_prime, yerr=bias_prime_err, fmt='o', color='black', label=r'$P_\times$')
plt.errorbar(z_hiram, total_bias_hiram, yerr=total_bias_err_hiram, color='red', fmt='o', label='BAO multi-$z$')
plt.ylabel(r"$b^{\prime}_\alpha$ = $b_\alpha + b_\mathrm{{HCD}}$")
plt.errorbar(2.33, -0.1558, yerr=0.0027, fmt='*', color='blue', ms=10, label='BAO')
plt.xlabel(r"$z$")
plt.legend(loc='upper right')

# %%

# %%

# %%

# %%

# %%
plt.scatter(zs, beta_alpha_prime, marker='o', color='black', label=r'$P_\times$')
# plt.errorbar(zs, beta_alpha_prime, np.abs(beta_alpha_prime_relerr*beta_alpha_prime), fmt='o', color='black', label=r'$P_\times$')
plt.errorbar(z_hiram, beta_prime_hiram, yerr=beta_prime_err_hiram, color='red', fmt='o', label='Herrera-Alcantar+26')
plt.ylabel(r"$\beta^{\prime}_\alpha$")
plt.scatter(2.33, 1.31, marker='*', color='blue', s=80, label='DESI DR2 BAO')
plt.xlabel(r"$z$")
plt.legend()

# %%
likes

# %%
for l, like in enumerate(likes):
    like.plot_px(params={'bias': results[l]['bias'], 'beta': results[l]['beta'], 'b_H': results[l]['b_H'], 'b_X': results[l]['b_X']}, multiply_by_k=False, include_probability=True,  title=f"z={zs[l]}", datalabel='DESI DR2', theorylabel="Best-fit theory")


# %%
