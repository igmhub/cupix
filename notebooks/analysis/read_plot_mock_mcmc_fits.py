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
# # Tutorial: Plot saved mock test results from the MCMC

# %%

from cupix.inference.sampling_funcs import load_mcmc_results, plot_contours, chain_bestfit_dict
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import os
from forestflow import priors


# %% [markdown]
# Set the zs and choice of z

# %%
zs = [2.2, 2.4, 2.6, 2.8]

# %%
# ls /pscratch/sd/m/mlokken/desi-lya/px/mocks/mcmc_fits/20260917_cont_freeXH

# %%
dir = "/pscratch/sd/m/mlokken/desi-lya/px/mocks/mcmc_fits/20260917_cont_freeXH"
fn = "chain_partial_it800.h5"

# %%
results_cont = []
freepars_cont = []
theories_cont = []
likes_cont = []

for iz in [0]:
    chain, free_params, data, cosmo, theory, like, setup_config, inf_config = load_mcmc_results(dir, fn, iz=iz)
    chain_noburnin = chain[300:, :, :]
    chain_noburnin = chain_noburnin.reshape(-1, chain_noburnin.shape[2])
    bestfits = chain_bestfit_dict(chain_noburnin, free_params)
    print(bestfits)
    like.plot_px(params={'b_X':0, 'b_H':0}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best-fit from uncont.", title=f"z={theory.z}", datalabel="Contaminated mocks", include_probability=False)

    like.plot_px(params=bestfits, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit", title=f"z={theory.z}", datalabel="Contaminated mocks", include_probability=False)
    # plot without metals
    bestfits_no_metals = bestfits.copy()
    bestfits_no_metals['b_X'] = 0.0

    like.plot_px(params=bestfits_no_metals, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit (no metals)", title=f"z={theory.z}", datalabel="Contaminated mocks", include_probability=False)
    
    bestfits_no_hcds = bestfits.copy()
    bestfits_no_hcds['b_H'] = 0.0
    print(bestfits_no_hcds)
    like.plot_px(params=bestfits_no_hcds, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit (no HCDs)", title=f"z={theory.z}", datalabel="Contaminated mocks", include_probability=False)
    
    bestfits_no_X_H = bestfits.copy()
    bestfits_no_X_H['b_X'] = 0.0
    bestfits_no_X_H['b_H'] = 0.0

    print(bestfits_no_X_H)
    like.plot_px(params=bestfits_no_X_H, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit (no metals or HCDs)", title=f"z={theory.z}", datalabel="Contaminated mocks", include_probability=False)

    # sigma_mF = np.std(chain_noburnin[:, 0])
    # if iz==0:
    #     label=r"$P_\times$, this work"
    # else:
    #     label=None
    # plt.errorbar(theory.z, bestfits['mF'], yerr=sigma_mF, marker='o', color='green', label=label)
    # print(theory.get_param('kC_Mpc'))

    # results_directory = "/pscratch/sd/m/mlokken/desi-lya/px/mocks/minimizer_fits/20260915_uncont_fix_cont/"
    # results_dict, free_params, data, cosmo, theory, like, setup_config, inf_config = load_mini_results(results_directory, filename=f"iminuit_results_{iz}.npz", setup_config_fname=f"setup_config_mini_z{iz}.yaml")
    # results_uncont.append(results_dict)
    # freepars_uncont.append(free_params)
    # theories_uncont.append(theory)
    # likes_uncont.append(like)


# %%
bestfits

# %%
plot_contours(chain_noburnin, free_params)

# %%
free_params

# %%
chain[:,4]

# %%
