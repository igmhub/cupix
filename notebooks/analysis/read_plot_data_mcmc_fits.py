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
import h5py



# %% [markdown]
# Set the zs and choice of z

# %%
zs = [2.2, 2.4, 2.6, 2.8]

# %%
# ls /pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/20261007_arinyo_paperbaseline #2_arinyo_paperbaseline

# %%
dir = "/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/20261007_arinyo_paperbaseline"

# %% [markdown]
# ## Check the autocorrelation time

# %%
for iz, z in enumerate(zs):
    if os.path.exists(os.path.join(dir, f"tau_estimates_z{iz}.txt")):
        tau_info = np.loadtxt(os.path.join(dir, f"tau_estimates_z{iz}.txt"))
        plt.plot(tau_info[:,0], tau_info[:,1], label=f'z={z}')
        # plot the necessary autocorrelation
        F = 30
        plt.plot(tau_info[:,0], tau_info[:,0]/F)
        plt.ylabel(r"$\langle \tau \rangle$")
        plt.xlabel(r"Iteration")
    plt.legend()

# %% [markdown]
# ## Inspect the chains

# %%
nburnin=300

# %%
results = []
for iz in range(len(zs)):
    fn = f"chain_z{iz}.h5"
    if not os.path.exists(os.path.join(dir, fn)):
        results.append([])
        print(f"Chain was not run for z{iz}, skipping...")
        continue
    
    # fn = f"chain_z{iz}.h5"
    mcmc_results = load_mcmc_results(dir, fn, iz=iz, inf_fname=f'inference_config_mcmc_arinyo_z{iz}.yaml', setup_fname=f'setup_config_mcmc_z{iz}.yaml')
    results.append(mcmc_results)
    chain = mcmc_results["chain_data"]["chain"]
    logprob = mcmc_results["chain_data"]["log_prob"]
    for i in range(len(mcmc_results["free_params"])):
        for w in range(chain.shape[1]):
            plt.plot(np.arange(chain.shape[0]), chain[:, w, i], alpha=0.5)
        plt.axvline(nburnin, color='k', linestyle='--', label='Burn-in limit')
        plt.title(f"Chains for {mcmc_results["free_params"][i].name}")
        plt.ylabel(mcmc_results["free_params"][i].name)
        plt.show()
        plt.clf()
        
    for w in range(logprob.shape[1]):
        plt.plot(np.arange(logprob.shape[0]), logprob[:, w], alpha=0.5)
    
    plt.title(f"Log probability, z={zs[iz]}")
    plt.ylabel("Log probability")
    plt.axvline(nburnin, color='k', linestyle='--', label='Burn-in limit')
    plt.show()
    plt.clf()
    # again, with nburnin removed limit
    for w in range(logprob.shape[1]):
        plt.plot(np.arange(logprob.shape[0])[nburnin:], logprob[nburnin:, w], alpha=0.5)
    
    plt.title(f"Log probability, z={zs[iz]}")
    plt.ylabel("Log probability")
    plt.show()
    plt.clf()

# %%
xlim = [0, 0.7]

for iz in range(len(zs)):
    if len(results[iz])>0:
        bestfits = chain_bestfit_dict(results[iz]["chain_data"], results[iz]["free_params"], nburnin=nburnin, thin=3)
        bestfits_only_SiII = bestfits.copy()
        bestfits_only_SiII['b_SiIII']  = 0
        bestfits_only_SiIII = bestfits.copy()
        bestfits_only_SiIII['b_SiII']  = 0
        bestfits_no_metals = bestfits.copy()
        bestfits_no_metals['b_SiII']  = 0
        bestfits_no_metals['b_SiIII']  = 0
        params = {par.name: bestfits[par.name] for par in results[iz]["free_params"]}

        
        # mcmc_results["like"].plot_px(params=bestfits_no_metals, include_chi2=True, every_other_theta=True, multiply_by_k=True, xlim=xlim, theorylabel="Best-fit without metals", title=f"z={mcmc_results['theory'].z}", datalabel="DR2", include_probability=False, ylim2=[-5,15])
        # mcmc_results["like"].plot_px(params=bestfits_only_SiII, include_chi2=True, every_other_theta=True, multiply_by_k=True, xlim=xlim, theorylabel="Best-fit without SiIII", title=f"z={mcmc_results['theory'].z}", datalabel="DR2", include_probability=False, ylim2=[-5,15])
        # mcmc_results["like"].plot_px(params=bestfits_only_SiIII, include_chi2=True, every_other_theta=True, multiply_by_k=True, xlim=xlim, theorylabel="Best-fit without SiII", title=f"z={mcmc_results['theory'].z}", datalabel="DR2", include_probability=False, ylim2=[-5,15])
        results[iz]["like"].plot_px(params=params, include_chi2=False, every_other_theta=False, multiply_by_k=True, xlim=xlim, theorylabel="Best-fit theory", title=f"z={results[iz]['theory'].z}", datalabel="DR2 measurements", include_probability=False, ylim2=[-5,5], n_free_p=len(results[iz]["free_params"]))
        
        


# %%
def results_to_latex(res, names=None, fmt="{:.3f}", caption=None, label=None):
    """
    res   : dict with 'par', 'par_err_up', 'par_err_low'
    names : optional dict mapping par -> LaTeX string (without $), e.g. {'H0': r'H_0'}
    """
    names = names or {}
    pars = [k for k in res if not k.endswith(('_err_up', '_err_low', '_err', '_mean'))]
    print(pars)
    lines = [r"\begin{table}", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines += [r"\begin{tabular}{lc}", r"\hline", r"Parameter & Value \\", r"\hline"]

    for p in pars:
        name = names.get(p, p.replace('_', r'\_') if False else p)
        val = fmt.format(res[p])
        up = fmt.format(abs(res[f"{p}_err_up"]))
        lo = fmt.format(abs(res[f"{p}_err_low"]))
        lines.append(rf"${name}$ & ${val}^{{+{up}}}_{{-{lo}}}$ \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)




# %%
for iz, z in enumerate(zs):
    if len(results[iz])>0:
        # plot_contours(results[iz]["chain_data"], results[iz]["free_params"], nburnin=nburnin, thin=4, title=f"z={z}", include_eff=True, show_priors=True)
        print(f"Best-fit parameters for z={z}:")
        res = results_to_latex(chain_bestfit_dict(results[iz]["chain_data"], results[iz]["free_params"], nburnin=nburnin, thin=3, include_eff=True),  caption="Best-fit parameters", label="tab:results")
        print(res)

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
        bestfit_dict = chain_bestfit_dict(results[iz]["chain_data"], results[iz]["free_params"], nburnin=nburnin, thin=3, include_eff=True)
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
        bestfit_dict = chain_bestfit_dict(results[iz]["chain_data"], results[iz]["free_params"], nburnin=150, thin=3, include_eff=True)
        plt.errorbar(zs[iz], bestfit_dict["beta_eff"], yerr=[[abs(bestfit_dict["beta_eff"]-bestfit_dict["beta_eff_err_low"])], [abs(bestfit_dict["beta_eff"]-bestfit_dict["beta_eff_err_up"])]], fmt='o', color='black', label=our_label)

plt.axhline(y=beta_eff_bao_singlez, label=r'$\beta$ used for BAO single-z')
plt.scatter(2.33, beta_eff_bao_singlez, color='blue', marker='*', s=100)
plt.xlabel(r"$z$")
plt.legend(loc='upper right', fontsize=13)
plt.ylabel(r"$\beta_\alpha^\prime$")
plt.savefig("../../plots/beta_eff_vs_z.pdf", bbox_inches='tight')

# %%
