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

# %%
# import cupix.inference.sampling_funcs as sf
from cupix.inference.mcmc_results import MCMCResults
from cupix.utils.utils import get_path_repo
import os
import numpy as np
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2

# %%
cupixpath = get_path_repo('cupix')
chain_dir = os.path.join(cupixpath, "data", "example_chain") 
mcmc_results = MCMCResults(chain_dir, iz = 0)

# %% [markdown]
# Plot some diagnoses for the chain. First, the autocorrelation time

# %%
mcmc_results.plot_tau_estimates(iz=0)

# %% [markdown]
# Now, examine the chains themselves

# %%
mcmc_results.plot_chains(param_name='As')

# %%
mcmc_results.plot_prob()

# %%
mcmc_results.plot_contours(params_to_plot=['As','ns','Delta2_p', 'n_p'], thin=3, show_best_fit=True)

# %%
mcmc_results.chain_bestfit_dict()

# %%
mcmc_results.latex_table(nburnin=500)
