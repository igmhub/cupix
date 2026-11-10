# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plot chain results from the MCMC

# %%

from cupix.inference.mcmc_results import MCMCResults
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import os
from forestflow import priors
from cupix.likelihood.theory import Theory
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from lace.cosmo import cosmology
from cupix.likelihood.likelihood import Likelihood
import h5py



# %% [markdown]
# Set the zs and choice of z

# %%
mode = 'igm' # 'arinyo'

# %%
# ls /pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/20261030_arinyo_paperbaseline_binfix/  #2_arinyo_paperbaseline


# %%
if mode=='arinyo':
    dir = f"/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/20261030_{mode}_paperbaseline_binfix"
elif mode == 'igm':
    dir = f"/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/20261102_{mode}_paperbaseline_binfix"
    # 20261015_{mode}_paperbaseline"

# %% [markdown]
# # Load the chains

# %%
chains = MCMCResults(dir, iz=[0,1,2,3])

# %% [markdown]
# ## Check the autocorrelation time

# %%
for iz in [0,1,2,3]:
    chains.plot_tau_estimates(iz=iz)

# %% [markdown]
# ## Inspect the chains

# %%
for iz in [0,1,2,3]:
    chains.plot_chains(iz=iz, param_name='mF')
    chains.plot_chains(iz=iz, param_name='sigT_Mpc')

# %% [markdown]
# Decide on nburnin from the previous plots, thin based on number of walkers

# %%
nburnin = 200
thin = 4

# %%
for iz in [0,1,2,3]:
    chains.plot_best_fit(iz=iz, nburnin=nburnin, thin=thin)

# %%
for iz in [0,1,2,3]:
    chains.plot_contours(iz=iz, nburnin=nburnin, thin=thin, params_to_plot=['mF', 'T0', 'b_H','b_SiII','b_SiIII'])

# %%
chains.chain_files

# %%
include_eff=False
include_T0=True
for iz, z in enumerate(zs):
    if len(results[iz])>0:
        print(f"Best-fit parameters for z={z}:")
        if mode=='arinyo':
            derived_params=['beta_eff', 'b_eff']
        elif mode=='igm':
            derived_params=['T0']

        latex_table(results[iz]["chain_data"], nburnin=nburnin, thin=3, derived_pars=derived_params)
        plot_contours(results[iz]["chain_data"], results[iz]["free_params"], nburnin=nburnin, thin=4, theory=results[iz]['theory'],derived_pars=derived_params, show_priors=True, params_to_plot=["bias", "beta", "b_SiII", "b_SiIII", "b_H", "b_eff", "beta_eff"], show_bounds=True, title=f"z={z}")
        

# %%
z_hiram = [2.13, 2.41, 2.81]
hiram_bias = [-0.0703, -0.1428, -0.2286]
hiram_bias_err = [.01, .01, .01]
hiram_b_hcd = [-0.061, -0.024, -0.0102]
hiram_bhcd_err = [.01, .01, .01]
eff_bias_hiram = [-0.1317,-0.1670, -0.2388]
eff_bias_err_hiram = [[0.0035, 0.0035], [0.0041, 0.0045], [0.0045, 0.0045]]
hiram_beta = [2.25, 1.469, 1.208]
hiram_beta_err = [0.3, .1, .55]
beta_prime_hiram = np.array([1.423, 1.322, 1.176])
beta_prime_err_hiram = np.array([0.048, 0.045, 0.046])
bias_bao_singlez = -0.1352
bH_bao_singlez = -0.0206
bias_eff_bao_singlez = (bias_bao_singlez + bH_bao_singlez)
beta_bao_singlez = 1.445
beta_hcd_bao_singlez = 0.508
beta_eff_bao_singlez = (bias_bao_singlez*beta_bao_singlez + bH_bao_singlez*beta_hcd_bao_singlez) / bias_eff_bao_singlez

            # we assume beta_H is 0.5

# %%
# plot 
plt.figure(figsize=(6,5))
plt.errorbar(z_hiram, eff_bias_hiram, yerr=np.asarray(eff_bias_err_hiram).T, color='red', fmt='o', label='BAO multi-$z$')

plt.errorbar(2.33, bias_eff_bao_singlez, yerr=0.0027, fmt='*', color='blue', ms=10, label='BAO single-$z$')

for iz in range(len(zs)):
    if iz==0:
        our_label = r'$P_\times$'
    else:
        our_label = ""
    if len(results[iz])>0:
        bestfit_dict = chain_bestfit_dict(results[iz]["chain_data"], results[iz]["free_params"], nburnin=nburnin, thin=3, derived_pars=['b_eff', 'beta_eff'])
        plt.errorbar(zs[iz], bestfit_dict["b_eff"], yerr=[[abs(bestfit_dict["b_eff"]-bestfit_dict["b_eff_err_low"])], [abs(bestfit_dict["b_eff"]-bestfit_dict["b_eff_err_up"])]], fmt='o', color='black', label=our_label)
                     
plt.xlabel(r"$z$")
plt.legend(loc='upper right', fontsize=13)
plt.ylabel(r"$b_\alpha^\prime$")
plt.savefig("../../plots/beff_vs_z.pdf", bbox_inches='tight')

# %%
beta_bao_singlez = 1.365

# %%
# same for effective beta
# plot 
plt.figure(figsize=(6,5))
plt.errorbar(z_hiram, beta_prime_hiram, yerr=beta_prime_err_hiram, color='red', fmt='o', label='BAO multi-$z$')

for iz in range(len(zs)):
    if iz==0:
        our_label = r'$P_\times$'
    else:
        our_label = ""
    if len(results[iz])>0:
        bestfit_dict = chain_bestfit_dict(results[iz]["chain_data"], results[iz]["free_params"], nburnin=150, thin=3, derived_pars=['b_eff', 'beta_eff'])
        plt.errorbar(zs[iz], bestfit_dict["beta_eff"], yerr=[[abs(bestfit_dict["beta_eff"]-bestfit_dict["beta_eff_err_low"])], [abs(bestfit_dict["beta_eff"]-bestfit_dict["beta_eff_err_up"])]], fmt='o', color='black', label=our_label)

plt.axhline(y=beta_eff_bao_singlez, label=r'$\beta$ used for BAO single-z')
plt.scatter(2.33, beta_eff_bao_singlez, color='blue', marker='*', s=100)
plt.xlabel(r"$z$")
plt.legend(loc='upper right', fontsize=13)
plt.ylabel(r"$\beta_\alpha^\prime$")
plt.savefig("../../plots/beta_eff_vs_z.pdf", bbox_inches='tight')

# %%
z = np.array([2.2, 2.4, 2.6, 2.8])
mF_gaikwad = np.array([.826, .79, .767, .74])
gaikwad_err = np.array([0.0206, 0.0210, 0.0216, 0.0212])
mF_cm = np.array([priors.get_IGM_priors(zi)["mean"]["mF"] for zi in z])
mF_cm_err = np.array([priors.get_IGM_priors(zi)["std"]["mF"] for zi in z])

# %%

plt.fill_between(z, mF_gaikwad-gaikwad_err, mF_gaikwad+gaikwad_err, alpha=.2, label='Gaikwad+21')
plt.fill_between(z, mF_cm - mF_cm_err, mF_cm + mF_cm_err, alpha=.5, label='Chaves-Montero+25', color='orange')
for iz in range(4):
    
    if iz==0:
        label=r"$P_\times$, this work"
    else:
        label=None
    
    bestfits = chains.chain_bestfit_dict(iz=iz,nburnin=nburnin, thin=thin)
    plt.errorbar(chains.theories[str(iz)].z, bestfits['mF'], yerr=bestfits['mF_err'], marker='o', color='green', label=label)


plt.legend(fontsize=15)
plt.ylabel(r"$\overline{F}$", fontsize=15)
plt.xlabel('z', fontsize=15)
# plt.savefig("../../plots/mF_estimates_preliminary.pdf", bbox_inches="tight")

# %%
