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
import numpy as np
from cupix.likelihood.new_likelihood import Likelihood
from cupix.likelihood.theory import Theory
import matplotlib.pyplot as plt
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.new_minimizer import IminuitMinimizer
import cupix
from lace.cosmo import cosmology
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Load the mock data

# %%
# mock_type = 'raw'
# mock_type = 'uncontaminated'
mock_type = 'tru_cont'
# mock_type = 'obs'
analysis_type = 'stack'
# analysis_type = 'single'

bin_type = 'unbinned'
# bin_type = 'medium_binned'
# bin_type = 'coarse_binned'
if bin_type == 'unbinned':
    bin_label='binned'
    ntheta = 20
elif bin_type=='coarse_binned':
    bin_label='binned'
    ntheta = 10
elif bin_type=='medium_binned':
    bin_label='binned'
    ntheta=18

# %%
# ls /global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/tru_cont/tru_cont_binned_*

# %%
if analysis_type == 'stack':
    fname = f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/{mock_type}/{mock_type}_{bin_label}_out_bf3_px-zbins_4-thetabins_{ntheta}_w_res_avg50.hdf5"
    mockdata = DESI_DR2(fname, theta_min_cut_arcmin=30, kM_max_cut_AA=1, km_max_cut_AA=1.2)
    print(fname)
    # MockData.cov_ZAM /= np.sqrt(50) # use this line if you want to reduce to the stack-on-mock errors
elif analysis_type == 'single':
    mockdata = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-0/{mock_type}/{bin_type}_out_px-zbins_2-thetabins_{ntheta}_w_res.hdf5", theta_min_cut_arcmin=14, kmax_cut_AA=1)
zs = np.array(mockdata.z)


# %%
mockdata.theta_centers_arcmin

# %%
zs

# %% [markdown]
# Set up theory

# %%
theories_xi = []
theories_new = []
cosmo = cosmology.Cosmology()
for z in zs:
    theories_xi.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False, 'default_lya_model':'best_fit_arinyo_from_colore'}))
    theories_new.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False, 'default_lya_model':'best_fit_arinyo_from_colore', 'q1':0, 'q2':0, 'kp':10000})) # linear theory
                                                             

# %%
likes_lya_xi = []
likes_lya_new = []

for iz, z in enumerate(zs):
    likes_lya_xi.append(Likelihood(data=mockdata, theory=theories_xi[iz], iz=iz, verbose=False))
    likes_lya_new.append(Likelihood(data=mockdata, theory=theories_new[iz], iz=iz, verbose=False))

# %% [markdown]
# First, plot the theory model on top of the stack

# %%
iz = 0
likes_lya_xi[0].plot_px(multiply_by_k=False, every_other_theta=False, xlim=[-.01, .4], datalabel="Tru-cont mock stack", theorylabel=r"$\xi$ best-fit", show=True, residual_to_theory=False, title=f"z={zs[iz]}")

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
like_params = []
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-.5,
    max_value=-.05,
    ini_value=-.15,
    value =-.15
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.5,
    max_value=2.5,
    ini_value=1.5,
    value =1.5
    ))


# %%
mini = IminuitMinimizer(likes_lya_new[iz], free_params=like_params, verbose=True)


# %%
mini.minimize()

# %%
mini.get_best_fit_chi2()

# %%
mini.get_best_fit_chi2(return_info=True)

# %%
mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, xlim=[-.01, .4], datalabel="True-cont mock", theorylabel='Best fit', show=True)

# %%
mini.get_best_fit_probability()

# %%
mini.get_best_fit_value(f'bias', return_hesse=True), mini.get_best_fit_value(f'beta', return_hesse=True)

# %%
mini.plot_ellipses(f"bias", f"beta", nsig=3, xrange=[-.139, -.103], yrange=[1.12,1.7], true_vals={'bias':theories_xi[iz].lya_model.default_lya_params['bias'], 'beta':theories_xi[iz].lya_model.default_lya_params['beta']}, true_val_label='Laura fit', show_ini_vals=True) 
# plt.title(f"z={z_choice}")
# mini.plot_ellipses("bias_1", "beta_1", nsig=3, cube_values=False, true_vals={'bias_1':like_parameter_by_name(like_params, 'bias_1').value, 'beta_1':like_parameter_by_name(like_params, 'beta_1').value}, true_val_label='Laura fit')

# %%
import os

mini.save_results(outfile=os.path.splitext(os.path.basename(fname))[0]+f"_iminuit_{zs[iz]}")

# %%
# check results
outfile = np.load(f"/global/common/software/desi/users/mlokken/cupix/data/fitter_results/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50_iminuit_{zs[iz]}.npz")
for key in outfile.keys():
    print(key)
    print(outfile[key])

# %%
from cupix.likelihood.new_minimizer import plot_ellipses
# plot results without minimizer object
zs = [2.2, 2.4, 2.6, 2.8]
iz = 0
z_choice = zs[iz]
# check results
outfile = np.load(f"/global/common/software/desi/users/mlokken/cupix/data/fitter_results/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50_iminuit_{z_choice}.npz")

for key in outfile.keys():
    # print(key)
    # print(outfile[key])
    bias = outfile['bias']
    bias_err = outfile['bias_err']
    beta = outfile['beta']
    beta_err = outfile['beta_err']
    cov = outfile[f'cov']

plot_ellipses(bias, beta, 'bias', 'beta', bias_err, beta_err, cov, nsig=3, true_vals={'bias':theories_xi[iz].lya_model.default_lya_params['bias'], 'beta':theories_xi[iz].lya_model.default_lya_params['beta']}, true_val_label='Laura fit', xrange=[-.14,-.1], yrange=[1.1, 1.7])
plt.title(f"z = {z_choice}")


# %%
