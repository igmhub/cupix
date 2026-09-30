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
# # Tutorial: Plot saved mock test results from the MCMC

# %%

from cupix.inference.sampling_funcs import load_mcmc_results, plot_contours, chain_bestfit_dict
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



# %% [markdown]
# Set the zs and choice of z

# %%
zs = [2.2, 2.4, 2.6, 2.8]

# %%
# ls /pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/20260929_arinyo_SiII_III

# %%
dir = "/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/20260929_arinyo_SiII_III_fixtau"
fn = "chain_partial_it450.h5"

# %%
xlim = [0, 0.7]

for iz in [0]:
    
    mcmc_results = load_mcmc_results(dir, fn, iz=iz, inf_fname=f'inference_config_mcmc_arinyo_z{iz}.yaml', setup_fname=f'setup_config_mcmc_z{iz}.yaml')
    bestfits = chain_bestfit_dict(mcmc_results["chain_data"], mcmc_results["free_params"], nburnin=300, thin=5)
    bestfits_only_SiII = bestfits.copy()
    bestfits_only_SiII['b_SiIII']  = 0
    bestfits_only_SiIII = bestfits.copy()
    bestfits_only_SiIII['b_SiII']  = 0
    bestfits_no_metals = bestfits.copy()
    bestfits_no_metals['b_SiII']  = 0
    bestfits_no_metals['b_SiIII']  = 0
    

    mcmc_results["like"].plot_px(params=bestfits_no_metals, include_chi2=True, every_other_theta=True, multiply_by_k=True, xlim=xlim, theorylabel="Best-fit without metals", title=f"z={mcmc_results['theory'].z}", datalabel="DR2", include_probability=False, ylim2=[-5,15])
    mcmc_results["like"].plot_px(params=bestfits_only_SiII, include_chi2=True, every_other_theta=True, multiply_by_k=True, xlim=xlim, theorylabel="Best-fit without SiIII", title=f"z={mcmc_results['theory'].z}", datalabel="DR2", include_probability=False, ylim2=[-5,15])
    mcmc_results["like"].plot_px(params=bestfits_only_SiIII, include_chi2=True, every_other_theta=True, multiply_by_k=True, xlim=xlim, theorylabel="Best-fit without SiII", title=f"z={mcmc_results['theory'].z}", datalabel="DR2", include_probability=False, ylim2=[-5,15])
    mcmc_results["like"].plot_px(params=bestfits, include_chi2=True, every_other_theta=True, multiply_by_k=True, xlim=xlim, theorylabel="Best-fit with metals", title=f"z={mcmc_results['theory'].z}", datalabel="DR2", include_probability=True, ylim2=[-5,15], n_free_p=len(mcmc_results["free_params"]))
    


# %%
plot_contours(mcmc_results["chain_data"], mcmc_results["free_params"], nburnin=200, thin=3)

# %%
-.086 + -0.053

# %%

tau_info = np.loadtxt(os.path.join(dir, "tau_estimates.txt"))
plt.plot(tau_info[:,0], tau_info[:,1])
# plot the necessary autocorrelation
F = 50
plt.plot(tau_info[:,0], tau_info[:,0]/F)

# %%
taus.shape

# %%
