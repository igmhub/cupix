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
fname = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/contaminated/contaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
data = DESI_DR2(config = {'data_file':fname, 'kM_max_cut_AA':0.7, 'km_max_cut_AA':0.77, 'theta_min_cut_arcmin':15})
data.cov_ZAM/=np.sqrt(50) # rescale the covariance appropriately for the stack
zs = data.z

# %%
cosmo = cosmology.Cosmology()
config={'verbose': False, 'include_hcd': True, 'include_metal': True,
        'include_sky': False, 'include_continuum': True, 'default_lya_model': 'best_fit_arinyo_from_colore'}
theories = []
for iz,z in enumerate(zs):
    theories.append(Theory(z=z, fid_cosmo=cosmo, config=config))

# %%
theories[0].lya_model.default_lya_model

# %%
likes = []
largescale_results = []
for iz, z in enumerate(zs):
    likes.append(Likelihood(data=data, theory=theories[iz], iz=iz, config={'verbose':False}))
    # set the large-scale bias and beta as bestfits
    results_directory = "/pscratch/sd/m/mlokken/desi-lya/px/mocks/minimizer_fits/20260914_trucont_bias_beta_kp/"
    results_dict, _, _, _, _, _, _, _ = load_mini_results(results_directory, filename=f"iminuit_results_{iz}.npz", setup_config_fname=f"setup_config_mini_z{iz}.yaml")
    largescale_results.append(results_dict)


# %%
for iz in [0]:
    # plot Px with large-scale bias and beta, then apply b_X and b_H from mcmc run
    likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':0, 'b_X':0}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit large-scale", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False)
    # likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':-0.0186, 'b_X':-0.004051}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Best fit with metals and HCDs", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False)
    likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':0, 'b_X':-0.004051}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Only metals", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False)
    likes[iz].plot_px(params={'bias':largescale_results[iz]['bias'], 'beta':largescale_results[iz]['beta'], 'b_H':-0.0186, 'b_X':0}, include_chi2=True, every_other_theta=False, multiply_by_k=True, xlim=[0,.3], theorylabel="Only HCDs", title=f"z={theories[iz].z}", datalabel="Contaminated mocks", include_probability=False)

# %%
