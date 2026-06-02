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

# %%
from cupix.parameter_inference.sampling_funcs import load_mcmc_results, plot_chain_flattened, plot_contours
# %load_ext autoreload
# %autoreload 2

# %%
chain_directory = "/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/20260602_dist_cont/"
chain, free_params, data, cosmo, theory, like, setup_config, inf_config = load_mcmc_results(chain_directory)

# %%
chain.shape

# %%
plot_chain_flattened(chain, free_params, 0, outdir=chain_directory, )

# %%
plot_contours(chain, free_params, title="Unconverged chain on contaminated forecast", save=False, show=True)

# %%

# %%
