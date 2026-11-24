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
# izs = [0]
izs = [0,1,2,3]

# %%
mode = 'igm' # 'arinyo'

# %%
if mode=='arinyo':
    dir = f"/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/20261030_{mode}_paperbaseline_binfix"
elif mode == 'igm':
    dir = f"/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/20261124_{mode}_binfix_morefree"
    # 20261015_{mode}_paperbaseline"

# %% [markdown]
# # Load the chains

# %%
chains = MCMCResults(dir, izs)

# %% [markdown]
# ## Check the autocorrelation time

# %%
for iz in izs:
    chains.plot_tau_estimates(iz=iz)

# %% [markdown]
# ## Inspect the chains

# %%
for iz in izs:# [0,1,2,3]:
    chains.plot_chains(iz=iz, param_name='mF')
    chains.plot_chains(iz=iz, param_name='sigT_Mpc')
    chains.plot_chains(iz=iz, param_name='gamma')

# %% [markdown]
# Decide on nburnin from the previous plots, thin based on number of walkers

# %%
nburnin = 50
thin = 4

# %%
chains.freepars

# %%
for iz in izs:  #[0,1,2,3]:
    chains.plot_best_fit(iz=iz, nburnin=nburnin, thin=thin, include_probability=False, include_chi2=True, every_other_theta=True, ncols_leg=2)

# %%
chains.setup_configs['0'].data_config.keys()

# %%
data.k_M_edges

# %%

# load the original data and plot it underneath
data_config = {'data_file':chains.setup_configs['0'].data_config['data_file'], 'theta_min_cut_arcmin':5} # get all data
data = DESI_DR2(data_config)
for iz in izs:
    chains.plot_best_fit(iz=iz, nburnin=nburnin, thin=thin, include_probability=False, include_chi2=False, every_other_theta=False, extra_data=data,
                         xlim=[0,.65], ncols_leg=2, ylim=[-.0005, 0.009], multiply_by_k=True, xshade=[0, 0.03451457], ylim2=[-3.5, 3.5])


# %%
for iz in izs:
    chains.plot_contours(iz=iz, nburnin=nburnin, thin=thin, params_to_plot=['mF', 'T0', 'gamma', 'kF_Mpc', 'b_H','b_SiII','b_SiIII'])

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
z_turner = np.array([2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75, 2.85])
tau_turner = np.array([0.158, 0.179, 0.200, 0.226, 0.235, 0.268, 0.292, 0.316])
tau_err_turner = np.array([.012, .015, .016, .016, .018, .019, .020, .021])
mF_turner = np.exp(-1*tau_turner)
mF_err_turner = mF_turner * tau_err_turner
# mF_cm = np.array([priors.get_IGM_priors(zi)["mean"]["mF"] for zi in z])
# mF_cm_err = np.array([priors.get_IGM_priors(zi)["std"]["mF"] for zi in z])
mF_cm =  np.array([0.8151, 0.7906,  0.7637, 0.7303])
mF_cm_err = np.array([0.0047,0.0044, 0.0049, .0056])

# %%
from lace.cosmo.thermal_broadening import T0_from_broadening_kms
T0_gaikwad = np.array([11000, 12750, 13500, 14750])/1e4
T0_gaikwad_err = np.array([1028, 1132, 1390, 1341])/1e4
sigT_cm = np.array([0.1611, 0.1467, 0.1311, 0.1205])
sigT_cm_err = np.array([0.0043, 0.0037, 0.0057, 0.0068])
T0_cm = []
T0_cm_err = []
for iz in range(len(z)):
    dkms_dMpc = chains.theories[str(iz)].fid_cosmo.get_dkms_dMpc(z[iz])
    sigT_kms = sigT_cm[iz] * dkms_dMpc
    T0_cm.append(T0_from_broadening_kms(sigT_kms) / 1e4)

    T0_cm_err_iz = 2 * sigT_cm_err[iz] / sigT_cm[iz] * T0_cm[iz]
    T0_cm_err.append(T0_cm_err_iz)
    if iz==2:
        break
T0_cm = np.asarray(T0_cm)
T0_cm_err = np.asarray(T0_cm_err)

# %%
T0_cm_err.shape

# %%
plt.fill_between(z[:3], T0_cm - T0_cm_err, T0_cm + T0_cm_err, alpha=.5, label='Chaves-Montero+25', color='orange')
plt.fill_between(z, T0_gaikwad - T0_gaikwad_err, T0_gaikwad + T0_gaikwad_err, alpha=.5, label='Gaikwad+25', color='red')
for iz in izs:
    
    if iz==0:
        label=r"$P_\times$, this work"
    else:
        label=None
    
    bestfits = chains.chain_bestfit_dict(iz=iz,nburnin=nburnin, thin=thin)
    plt.errorbar(chains.theories[str(iz)].z, bestfits['T0'], yerr=bestfits['T0_err'], marker='o', color='green', label=label)

plt.legend()



# %%

plt.fill_between(z, mF_gaikwad-gaikwad_err, mF_gaikwad+gaikwad_err, alpha=.2, label='Gaikwad+21')
plt.fill_between(z, mF_cm - mF_cm_err, mF_cm + mF_cm_err, alpha=.5, label='Chaves-Montero+25', color='orange')
plt.fill_between(z_turner, mF_turner - mF_err_turner, mF_turner + mF_err_turner, alpha=.5, label='Turner+24', color='red')
for iz in izs:
    
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
