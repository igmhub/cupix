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
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.inference.free_parameter import FreeParameter
from cupix.inference.posterior import Posterior
from cupix.inference.minimize_posterior import Minimizer
from cupix.inference.minimizer_funcs import plot_ellipse, plot_corner, load_mini_results

# %%
# ls /global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/contaminated/

# %%
fname = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/contaminated/contaminated_baseline_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50_ncov.hdf5"

data = DESI_DR2(config = {'data_file':fname, 'kM_min_cut_AA': 0.03, 'kM_max_cut_AA':0.7, 'km_max_cut_AA':0.77, 'theta_min_cut_arcmin':30})
zs = data.z

# %%
cosmo = cosmology.Cosmology()
config={'verbose': True, 'include_hcd': True, 'include_metal': True,
        'include_sky': False, 'include_continuum': True, 'default_lya_model': 'best_fit_arinyo_from_colore'}
theories = []
for iz,z in enumerate(zs):
    theories.append(Theory(z=z, fid_cosmo=cosmo, config=config))

# %%
likes = []
likes_uncont = []
largescale_results = []
for iz, z in enumerate(zs):
    likes.append(Likelihood(data=data, theory=theories[iz], iz=iz, config={'verbose':True}))
    # set the large-scale bias and beta as bestfits
    results_directory = "/pscratch/sd/m/mlokken/desi-lya/px/mocks/minimizer_fits/20260918_uncont_fix_cont/"
    results_dict, _, _, _, _, like_uncont, _, _ = load_mini_results(results_directory, filename=f"iminuit_results_{iz}.npz", setup_config_fname=f"setup_config_mini_z{iz}.yaml", inf_config_fname=f"inference_config_mini_z{iz}.yaml")
    likes_uncont.append(like_uncont)
    largescale_results.append(results_dict)


# %%
largescale_results[2]['bias'], largescale_results[2]['beta']

# %%
ylim = [-0.0005, 0.025]
for iz in [2]:
    # plot Px with large-scale bias and beta, then apply b_X and b_H from mcmc run
    likes_uncont[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':0, 'b_X':0}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit without contaminants", title=f"z={theories[iz].z}", datalabel="Uncontaminated mocks", include_probability=False, ylim2=[-5,15], connect_residuals=True, ylim=ylim)
    likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':0, 'b_X':0}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit from uncontaminated", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False, ylim2=[-5,15], connect_residuals=True, ylim=ylim)

    
    likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':-0.02, 'beta_H':0.7, 'b_X':0, 'beta_X':0}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Incl. HCD model", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False, ylim2=[-5,15], connect_residuals=True, ylim=ylim)
    # likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':-0.02, 'beta_H':0.5, 'b_X':-.005, 'beta_X':1.5}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.5], theorylabel="Metal + HCD", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False, ylim2=[-10,10], connect_residuals=True)
    likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':-0.02, 'beta_H':0.5, 'b_X':-.006, 'beta_X':1.}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Incl. HCD model + metals", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False, ylim2=[-5,15], connect_residuals=True, ylim=ylim)




    # likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':-0.0186, 'b_X':0}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Only HCDs", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False)

# %% [markdown]
# ## Now try for small oscillations

# %%
# get the smaller thetas

fname = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-1/contaminated/baseline/unbinned_out_px-zbins_4-thetabins_20_w_res.hdf5"

data = DESI_DR2(config = {'data_file':fname, 'kM_min_cut_AA': 0.03, 'kM_max_cut_AA':0.7, 'km_max_cut_AA':0.77, 'theta_min_cut_arcmin':2})
zs = data.z

# %%
cosmo = cosmology.Cosmology()
config={'verbose': True, 'include_hcd': True, 'include_metal': True,
        'include_sky': False, 'include_continuum': True, 'default_lya_model': 'best_fit_arinyo_from_colore'}
theories = []
likes = []
for iz,z in enumerate(zs):
    theory = Theory(z=z, fid_cosmo=cosmo, config=config)
    # change the lr_metal
    print(theory.cont_model.lr_metal)
#     theory.cont_model.lr_metal = 1260 # reset
#     print(theory.cont_model.lr_metal)
    theories.append(theory)
    likes.append(Likelihood(data=data, theory=theory, iz=iz, config={'verbose':True}))


# %%
ylim = [-0.0005, 0.3]
for iz in [2]:
    # plot Px with large-scale bias and beta, then apply b_X and b_H from mcmc run
    # likes_uncont[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':0, 'b_X':0}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit without contaminants", title=f"z={theories[iz].z}", datalabel="Uncontaminated mocks", include_probability=False, ylim2=[-5,15], connect_residuals=True, ylim=ylim)
    # likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':0, 'b_X':0}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit from uncontaminated", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False, ylim2=[-5,15], connect_residuals=True, ylim=ylim)

    
    # likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':-0.02, 'beta_H':0.5, 'b_X':0, 'beta_X':0}, include_chi2=False, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Incl. HCD model", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False, ylim2=[-5,15], connect_residuals=True, ylim=ylim)
    # likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':-0.02, 'beta_H':0.5, 'b_X':-.005, 'beta_X':1.5}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.5], theorylabel="Metal + HCD", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False, ylim2=[-10,10], connect_residuals=True)
    likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':-0.02, 'beta_H':0.5, 'b_X':-.01, 'beta_X':1.}, include_chi2=False, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Incl. HCD model + metals", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False, ylim2=[-5,15], connect_residuals=True, ylim=ylim, extra_params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':-0.02, 'beta_H':0.5, 'b_X':0}, extra_label='no metals')


# %%
