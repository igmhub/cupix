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
#     display_name: cupix
#     language: python
#     name: cupix
# ---

# %%
# Combine measurements

# %%
import numpy as np

# %%

from cupix.inference.sampling_funcs import load_mcmc_results, plot_contours, chain_bestfit_dict
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import os
from forestflow import priors

# %%
maindir = "/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/"

# %%
dirs = ["20260702_drop_bal_dla_igm_fixgammaT_z0", "20260706_july_baseline_igm_z1", "20260707_july_baseline_igm_z2", "20260708_july_baseline_igm_z3"]
fns = ["chain_partial_it750.h5", "chain_partial_it1450.h5", "chain.h5", "chain_partial_it1600.h5"]

# %%
z = np.array([2.2, 2.4, 2.6, 2.8])
mF_gaikwad = np.array([.826, .79, .767, .74])
gaikwad_err = np.array([0.0206, 0.0210, 0.0216, 0.0212])
mF_cm = np.array([priors.get_IGM_priors(zi)["mean"]["mF"] for zi in z])
mF_cm_err = np.array([priors.get_IGM_priors(zi)["std"]["mF"] for zi in z])


# %%
plt.fill_between(z, mF_gaikwad-gaikwad_err, mF_gaikwad+gaikwad_err, alpha=.2, label='Gaikwad+2021')
plt.fill_between(z, mF_cm - mF_cm_err, mF_cm + mF_cm_err, alpha=.5, label='Chaves-Montero+2025', color='orange')
for iz in range(4):
    chain, free_params, data, cosmo, theory, like, setup_config, inf_config = load_mcmc_results(maindir+dirs[iz], fname=fns[iz])
    chain_noburnin = chain[300:, :, :]
    chain_noburnin = chain_noburnin.reshape(-1, chain_noburnin.shape[2])
    bestfits = chain_bestfit_dict(chain_noburnin, free_params)
    # like.plot_px(params=bestfits, include_chi2=True, every_other_theta=False, multiply_by_k=False, xlim=[0,.5], theorylabel="Best fit", title=f"z={theory.z}", datalabel="DESI DR2")
    sigma_mF = np.std(chain_noburnin[:, 0])
    if iz==0:
        label=r"$P_\times$, this work"
    else:
        label=None
    plt.errorbar(theory.z, bestfits['mF'], yerr=sigma_mF, marker='o', color='green', label=label)
    print(theory.get_param('kC_Mpc'))

plt.legend(fontsize=15)
plt.ylabel(r"$\overline{F}$", fontsize=15)
plt.xlabel('z', fontsize=15)
# plt.savefig("../../plots/mF_estimates_preliminary.pdf", bbox_inches="tight")
