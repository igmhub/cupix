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
# ls /global/cfs/cdirs/desicollab/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-stack/jura-124/redshift_bins/fits/output_fitter/

# %%
# check multi-z results
from astropy.io import fits
hiram = fits.open("/global/cfs/cdirs/desicollab/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-stack/jura-124/redshift_bins/fits/output_fitter/lyaxlya-0.0_2.25_metalmatrix.fits")
for cols in hiram[1].header:
    print(cols)
hiram[1].header['bias_SiIII(1207)']


# %%
# check multi-z results
from astropy.io import fits
hiram = fits.open("/global/cfs/cdirs/desicollab/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-stack/jura-124/redshift_bins/fits/output_fitter/lyaxlya-2.25_2.6_metalmatrix.fits")
for cols in hiram[1].header:
    print(cols)
hiram[1].header['bias_SiIII(1207)']


# %%
# check multi-z results
from astropy.io import fits
hiram = fits.open("/global/cfs/cdirs/desicollab/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-stack/jura-124/redshift_bins/fits/output_fitter/lyaxlya-2.25_2.6_metalmatrix.fits")
for cols in hiram[1].header:
    print(cols)
hiram[1].header['bias_SiIII(1207)']


# %%
# check multi-z results
from astropy.io import fits
hiram = fits.open("/global/cfs/cdirs/desicollab/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-stack/jura-124/redshift_bins/fits/output_fitter/lyaxlya-2.6_10.0_metalmatrix.fits")
for cols in hiram[1].header:
    print(cols)
hiram[1].header['bias_SiIII(1207)']


# %%
zs = [2.2, 2.4, 2.6, 2.8]

# %%
# ls /pscratch/sd/m/mlokken/desi-lya/px/mocks/mcmc_fits/20260924*

# %%
dir = "/pscratch/sd/m/mlokken/desi-lya/px/mocks/mcmc_fits/20260924_cont_free5/"
fn = "chain_partial_it1650.h5"

# %%
results_cont = []
freepars_cont = []
theories_cont = []
likes_cont = []
bestfits_cont = []
for iz in [2]:
    print('h')
    mcmc_results = load_mcmc_results(dir, fn, iz=iz)
    bestfits = chain_bestfit_dict(mcmc_results["chain_data"], mcmc_results["free_params"], nburnin=300, thin=5)
    print(bestfits)
    bestfits_cont.append(bestfits)
    mcmc_results["like"].plot_px(params={'b_X':0, 'b_H':-0}, include_chi2=False, every_other_theta=True, multiply_by_k2=True, xlim=[0,.3], theorylabel="Best-fit without metals + HCDs.", title=f"z={mcmc_results['theory'].z}", datalabel="Contaminated mocks", include_probability=False, ylim2=[-5,15])

    mcmc_results["like"].plot_px(params=bestfits, include_chi2=True, every_other_theta=False, multiply_by_k2=True, xlim=[0,.3], theorylabel="Best fit", title=f"z={mcmc_results['theory'].z}", datalabel="Contaminated mocks", include_probability=False, ylim2=[-5,15])

    
    # # plot without metals
    bestfits_no_metals = bestfits.copy()
    bestfits_no_metals['b_X'] = 0.0

    mcmc_results["like"].plot_px(params=bestfits_no_metals, include_chi2=True, every_other_theta=False, multiply_by_k2=True, xlim=[0,.3], theorylabel="Best fit (no metals)", title=f"z={mcmc_results['theory'].z}", datalabel="Contaminated mocks", include_probability=False, ylim2=[-5,15])
    
    # bestfits_no_hcds = bestfits.copy()
    # bestfits_no_hcds['b_H'] = 0.0
    # print(bestfits_no_hcds)
    # return_dict["like"].plot_px(params=bestfits_no_hcds, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit (no HCDs)", title=f"z={return_dict["theory"].z.z}", datalabel="Contaminated mocks", include_probability=False)
    
    # bestfits_no_X_H = bestfits.copy()
    # bestfits_no_X_H['b_X'] = 0.0
    # bestfits_no_X_H['b_H'] = 0.0

    # print(bestfits_no_X_H)
    # return_dict["like"].plot_px(params=bestfits_no_X_H, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit (no metals or HCDs)", title=f"z={return_dict["theory"].z.z}", datalabel="Contaminated mocks", include_probability=False)

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
plot_contours(mcmc_results["chain_data"], mcmc_results["free_params"], nburnin=500, thin=1)

# %% [markdown]
# ## Following part is done very by-hand to get good plots for the paper

# %%
input_bias_beta = []
for iz, z in enumerate(zs):
    if iz==2:
        input_bias_beta.append({'bias':-0.125, 'beta':2.4})
    else:
        input_bias_beta.append({'bias':-0.125, 'beta':1.5}) # fix this later if needed, I don't remember

# %%
# ls /global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-1/contaminated/baseline/unbinned_out_px-zbins_4-thetabins_20_w_res.hdf5 -lrth


# %%
unbinned_mock = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-1/contaminated/baseline/unbinned_out_px-zbins_4-thetabins_20_w_res.hdf5"
mockdata = DESI_DR2(config={'data_file':unbinned_mock, 'kM_min_cut_AA':0.0, 'kM_max_cut_AA':0.7, 'km_max_cut_AA':0.77, 'theta_min_cut_arcmin':2.0, 'theta_max_cut_arcmin':25.0})
theories = []
cosmo = cosmology.Cosmology()
config={'verbose': True, 'include_hcd': True, 'include_metal': True,
        'include_sky': False, 'include_continuum': True, 'default_lya_model':'best_fit_arinyo_from_colore'}
likes = []
for iz in [2]:
    z = zs[iz]
    theory = Theory(z=z, fid_cosmo=cosmo, config=config)
    like = Likelihood(data=mockdata, theory=theory, iz=iz, config={'verbose':True})
    likes.append(like)
    
    like.plot_px(params={'b_X':0, 'b_H':-0} | input_bias_beta[iz], include_chi2=False, every_other_theta=True, multiply_by_k=True, xlim=[0,.5], theorylabel="Best-fit from uncont.", title=f"z={mcmc_results['theory'].z}", datalabel="Contaminated mocks", include_probability=False, connect_residuals=True)
    like.plot_px(params=bestfits_cont[0]| input_bias_beta[iz], include_chi2=False, every_other_theta=True, multiply_by_k=True, xlim=[0,.5], theorylabel="Best-fit from cont.", title=f"z={mcmc_results['theory'].z}", datalabel="Contaminated mocks", include_probability=False, connect_residuals=True)
    print(bestfits_cont[iz]| input_bias_beta[iz])



# %%
bestfits

# %%
