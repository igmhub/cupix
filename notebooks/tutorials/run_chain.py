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
# # Running a short MCMC chain with `cupix`
#
# This notebook walks through the same steps as `mcmc_sampler.py`, one piece at a time:
#
# 1. Load the **setup config** (data, cosmology, theory, likelihood) and the **inference config** (free parameters, priors, sampler settings)
# 2. Load the data (mock / forecast) for one redshift bin
# 3. Build the `Cosmology` → `Theory` → `Likelihood` → `Posterior` stack
# 4. Set up the `emcee` walkers and run a **short** chain
# 5. Look at the chain, check the autocorrelation time, and save it
#
# To keep the notebook simple and portable, the chain runs **serially** (no `multiprocessing` pool). For production runs, use `mcmc_sampler.py` on a cluster. See the note at the end.

# %% [markdown]
# ## 0. Imports

# %%
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import emcee

import cupix
from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.config import Config
from cupix.inference.inference_config import InferenceConfig
from cupix.inference.posterior import Posterior
from cupix.utils.utils import get_path_repo
from cupix.inference.sampling_funcs import (
    prepare_free_parameters,
    get_initial_walkers,
    create_output_directory,
    record_mcmc_settings,
    save_chain,
)

# %% [markdown]
# ## 1. Paths and configuration files
#
# `cupix` uses two YAML files:
#
# | File | Contents | Read by |
# |---|---|---|
# | **setup config** | `data_config`, `cosmo_config`, `theory_config`, `like_config` (including the redshift index `iz`) | `Config` |
# | **inference config** | `params_config` (free parameters, ranges, priors), `post_config`, `samp_config` (walkers, steps, burn-in, …), output location | `InferenceConfig` |
#
# We start from the example configs that ship with `cupix`. Copy them and edit the copies if you want to change things.
#
# 👉 **Set `DATA_PATH` to your mock / forecast data file.**

# %%
# Root of the cupix repository (example configs live in <repo>/example_configs)
CUPIX_PATH = get_path_repo('cupix')


SETUP_CONFIG_PATH = os.path.join(CUPIX_PATH, "example_configs", "fcast_best_fit_arinyo_from_p1d_theory_config.yaml")
INF_CONFIG_PATH   = os.path.join(CUPIX_PATH, "example_configs", "inference_config_ex.yaml")

# >>> TODO: point this at your mock / forecast data <<<
DATA_PATH = os.path.join(CUPIX_PATH, "data", "px_measurements", "forecast", "fcast_best_fit_arinyo_from_p1d_realwin_bf3_binned_out_px-zbins_4-thetabins_10_w_res_noiseless.hdf5")

SHORT_TAG = "tutorial"   # used in the output directory name: YYYYMMDD_<tag>

for p in (SETUP_CONFIG_PATH, INF_CONFIG_PATH):
    print(p, "exists:", os.path.exists(p))

# %%
setup_config = Config(SETUP_CONFIG_PATH)
inf_config = InferenceConfig(INF_CONFIG_PATH)

setup_config.print_all()
inf_config.print_all()

# %% [markdown]
# ### Point the data config at your file
#
# The data file location lives in `setup_config.data_config`. Look at the printout above to find the key that holds the file path, then override it here. You can also edit it directly in your copy of the YAML file.

# %%
# >>> TODO: replace "<data_path_key>" with the actual key name in data_config <<<
setup_config.data_config["data_file"] = DATA_PATH

# %%
setup_config.data_config["data_file"]

# %% [markdown]
# ## 2. Load the data and select a redshift bin
#
# `like_config['iz']` chooses the redshift bin that is fitted.

# %%
data = DESI_DR2(setup_config.data_config)

iz = setup_config.like_config["iz"]
z = data.z[iz]
print(f"Available redshifts: {data.z}")
print(f"Fitting redshift bin iz={iz}  (z = {z})")

# %% [markdown]
# ## 3. Build cosmology → theory → likelihood
#
# * `Cosmology` is the fiducial cosmology (CAMB runs inside it).
# * `Theory` is the Lyα model at redshift `z`, with an emulator if the model uses one.
# * `Likelihood` compares the theory with the data in bin `iz`.
#
# In `mcmc_sampler.py`, `delay_load_emu=True` is set so the emulator can be loaded separately in each worker process, which avoids pickling problems. We run serially here, so we load the emulator right away.

# %%
cosmo = cosmology.Cosmology(cosmo_params_dict=setup_config.cosmo_config)

setup_config.theory_config["delay_load_emu"] = False   # serial run: load emulator immediately
theory = Theory(z=z, fid_cosmo=cosmo, config=setup_config.theory_config)

like = Likelihood(data=data, theory=theory, iz=iz, config=setup_config.like_config)

# %% [markdown]
# Quick check: evaluate the χ² once at the default parameters. This also runs CAMB and caches the result, so later calls are faster.

# %%
t0 = time.time()
chi2 = like.get_chi2()
print(f"chi2 at default parameters: {chi2:.2f}   ({time.time() - t0:.1f} s)")

# %% [markdown]
# ## 4. Free parameters and the posterior
#
# The parameters listed in `inf_config.params_config` are sampled. Each one has an initial value, a flat range, and optionally a Gaussian prior and a "true" value (useful for forecasts). Every other parameter stays fixed.

# %%
free_param_names = list(inf_config.params_config.keys())
free_params = prepare_free_parameters(
    free_param_names, theory, setup_config.theory_config,
    params_config=inf_config.params_config,
)

print(f"{'name':>20} {'ini':>10} {'min':>10} {'max':>10} {'true':>10} {'prior mu':>10} {'prior sig':>10}")
for par in free_params:
    print(f"{par.name:>20} {str(par.ini_value):>10} {str(par.min_value):>10} {str(par.max_value):>10} "
          f"{str(par.true_value):>10} {str(par.gauss_prior_mean):>10} {str(par.gauss_prior_width):>10}")

post = Posterior(like, free_params, config=inf_config.post_config)
if post.fixed_params is not None:
    print("\nFixed parameters:", post.fixed_params)

# %% [markdown]
# Check that the log-posterior is finite at the initial point before sampling:

# %%
p_ini = np.array([par.ini_value for par in free_params])
print("log posterior at initial values:", post.get_log_posterior_from_values(p_ini))

# %% [markdown]
# ## 5. Sampler settings for a *short* chain
#
# The production script picks `max_nsteps = 100 + 10·Np³` and `nburnin = 50 + 3·Np³`. Those are long, so here we use small fixed numbers that are enough to see the sampler work. Increase them, or set them in `samp_config`, for real results.
#
# `emcee` needs an **even** number of walkers, and at least `2 × Np`.

# %%
Np = len(free_params)

nwalkers = inf_config.samp_config.get("nwalkers", 4 * Np)
nwalkers = max(nwalkers, 2 * Np)
nwalkers += nwalkers % 2          # must be even

NSTEPS  = 300    # short chain for the tutorial
NBURNIN = 100    # discarded when analysing

assert NBURNIN < NSTEPS
print(f"Np = {Np}, nwalkers = {nwalkers}, nsteps = {NSTEPS}, nburnin = {NBURNIN}")
print(f"Total posterior evaluations ≈ {nwalkers * NSTEPS}")

# %% [markdown]
# Create the output directory and record the settings. The same helper functions as in the script are used, so the outputs look the same.

# %%
runname = "EXAMPLE"
outdir = "EXAMPLE_DIRECTORY" # this needs to be entered
print("Outputs will be saved to", outdir)
os.mkdir(outdir)
record_mcmc_settings(outdir, nwalkers, NSTEPS, NBURNIN, iz)

# %% [markdown]
# ## 6. Run the chain
#
# We use the same move mix as the production script: 80% differential-evolution moves and 20% DE-snooker moves. The walkers start in a small ball around the initial values (`get_initial_walkers`).

# %%
moves = [
    (emcee.moves.DEMove(), 0.8),
    (emcee.moves.DESnookerMove(), 0.2),
]

sampler = emcee.EnsembleSampler(
    nwalkers, Np,
    post.get_log_posterior_from_values,   # log-probability function
    moves=moves,
)

p0 = get_initial_walkers(free_params, nwalkers)

t0 = time.time()
sampler.run_mcmc(p0, NSTEPS, progress=True)
print(f"Sampling took {time.time() - t0:.1f} s")
print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}  (aim for ~0.2–0.5)")

# %% [markdown]
# ## 7. Convergence: autocorrelation time
#
# The production script stops once the τ estimates are stable to 5% **and** the chain is longer than `F × τ`, where `F_tau` defaults to 30. A short chain will usually *not* pass that test, which is expected here. Treat the numbers below as a rough guide.

# %%
tau = sampler.get_autocorr_time(tol=0, quiet=True)
F = inf_config.samp_config.get("F_tau", 30)
for par, t in zip(free_params, tau):
    print(f"{par.name:>20}: tau ≈ {t:6.1f} steps")
print(f"\nChain length {NSTEPS} vs F*max(tau) = {F * np.max(tau):.0f}  ->",
      "converged" if NSTEPS > F * np.max(tau) else "NOT converged (expected for a short chain)")

# %% [markdown]
# ## 8. Look at the chains
#
# Trace plots show each walker's path. The shaded region is the burn-in that we discard. Dashed lines mark the true values, if there are any (for example in a forecast).

# %%
chain = sampler.get_chain()          # shape (nsteps, nwalkers, Np)
logprob = sampler.get_log_prob()     # shape (nsteps, nwalkers)

fig, axes = plt.subplots(Np + 1, 1, figsize=(9, 1.8 * (Np + 1)), sharex=True)
for i, par in enumerate(free_params):
    ax = axes[i]
    ax.plot(chain[:, :, i], color="k", alpha=0.2, lw=0.6)
    if par.true_value is not None:
        ax.axhline(par.true_value, color="C1", ls="--")
    ax.set_ylabel(par.name)
    ax.axvspan(0, NBURNIN, color="0.9", zorder=-1)
axes[-1].plot(logprob, color="C0", alpha=0.2, lw=0.6)
axes[-1].set_ylabel("log post")
axes[-1].set_xlabel("step")
plt.tight_layout()

# %% [markdown]
# ### Posterior summary and corner plot
#
# We discard the burn-in, thin by about τ/2, and flatten. If the `corner` package is installed, it is used for the corner plot.

# %%
thin = max(1, int(np.min(tau) / 2))
flat = sampler.get_chain(discard=NBURNIN, thin=thin, flat=True)
print(f"{flat.shape[0]} posterior samples after burn-in/thinning\n")

for i, par in enumerate(free_params):
    lo, med, hi = np.percentile(flat[:, i], [16, 50, 84])
    truth = f"   (true: {par.true_value})" if par.true_value is not None else ""
    print(f"{par.name:>20} = {med:.4g} +{hi - med:.3g} -{med - lo:.3g}{truth}")

# %%
try:
    import corner
    truths = [par.true_value for par in free_params]
    corner.corner(flat, labels=[p.name for p in free_params],
                  truths=None if all(t is None for t in truths) else truths,
                  show_titles=True, quantiles=[0.16, 0.5, 0.84])
except ImportError:
    print("`pip install corner` for a corner plot; showing 1D histograms instead.")
    fig, axes = plt.subplots(1, Np, figsize=(3 * Np, 2.5))
    for i, par in enumerate(free_params):
        ax = np.atleast_1d(axes)[i]
        ax.hist(flat[:, i], bins=40, histtype="step", color="k")
        if par.true_value is not None:
            ax.axvline(par.true_value, color="C1", ls="--")
        ax.set_xlabel(par.name)
    plt.tight_layout()

# %% [markdown]
# ## 9. Save the chain
#
# `save_chain` writes the full chain (burn-in included, same as the production script) and the log-probabilities to HDF5, in the format the `cupix` analysis tools expect.

# %%
save_chain(outdir, chain, free_params, log_prob=logprob, fname=f"chain_z{iz}.h5")
print("Saved to", os.path.join(outdir, f"chain_z{iz}.h5"))

# %% [markdown]
# ## Next steps: running in parallel
#
# For long chains, use the script, which spreads the walkers over a `multiprocessing` pool:
#
# ```bash
# python mcmc_sampler.py <short_tag> <setup_config.yaml> <inference_config.yaml>
# ```
#
# Things that differ from this notebook:
#
# * **Worker processes.** The script uses the `spawn` start method. Each worker loads the emulator itself (`delay_load_emu=True`) and runs CAMB once in `init_worker`. That is why the worker functions live in a module and not in a notebook: under `spawn`, functions defined in a notebook can't be pickled.
# * **Number of walkers.** By default this equals `SLURM_CPUS_PER_TASK`, unless `nwalkers` is set in `samp_config`.
# * **Automatic stopping.** τ is checked every 100 steps, and the run stops when τ is stable to 5% and the chain is longer than `F_tau × τ`.
# * **Checkpoints.** A partial chain is saved every 100 steps (`chain_z{iz}_partial_it{N}.h5`), together with `tau_estimates_z{iz}.txt`.
