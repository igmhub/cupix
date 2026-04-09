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
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.lya_theory import Theory
from cupix.likelihood.forestflow_emu import FF_emulator
from lace.cosmo import camb_cosmo
import matplotlib.pyplot as plt
from cupix.likelihood.likelihood_parameter import LikelihoodParameter, like_parameter_by_name
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
import cupix
cupixpath = cupix.__path__[0].rsplit('/', 1)[0]
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
    # /global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/tru_cont/tru_cont_binned_out_px-zbins_4-thetabins_10_w_res_avg50.hdf5
    data = f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/{mock_type}/{mock_type}_{bin_label}_out_bf3_px-zbins_4-thetabins_{ntheta}_w_res_avg50.hdf5"
    MockData = DESI_DR2(data, theta_min_cut_arcmin=30, kM_max_cut_AA=1, km_max_cut_AA=1.2)
    print(data)
    # MockData.cov_ZAM /= np.sqrt(50)
    # MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/{mock_type}/{mock_type}_{bin_label}_out_px-zbins_4-thetabins_{ntheta}_w_res_avg50.hdf5", theta_min_cut_arcmin=30, kM_min_cut_AA=0, kM_max_cut_AA=1, km_max_cut_AA=1.2)
elif analysis_type == 'single':
    MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-0/{mock_type}/{bin_type}_out_px-zbins_2-thetabins_{ntheta}_w_res.hdf5", theta_min_cut_arcmin=14, kmax_cut_AA=1)
zs = np.array(MockData.z)


# %%
MockData.theta_centers_arcmin

# %%
zs

# %% [markdown]
# Set up theory

# %%
theory = Theory(zs, default_lya_theory='best_fit_arinyo_from_colore', emulator_label="forestflow_emu", verbose=True)

# %%
iz_choice = 3
z_choice = zs[iz_choice]
print(z_choice)
like = Likelihood(MockData, theory, z=z_choice, verbose=False)

# %% [markdown]
# First, plot the CF best-fit theory model on top of the stack

# %%
like.plot_px(multiply_by_k=False, every_other_theta=False, xlim=[-.01, .4], datalabel="Tru-cont mock stack", theorylabel=r"$\xi$ best-fit", show=True, residual_to_theory=False, title=f"z={z_choice}")

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
arinyo_par_names = theory.arinyo_par_names

like_params = []
like_params.append(LikelihoodParameter(
    name=f'bias_{like.theory_iz}',
    min_value=-.5,
    max_value=0.0,
    ini_value=-.25, #theory.default_param_dict[f'bias_{like.theory_iz}'],
    value = theory.default_param_dict[f'bias_{like.theory_iz}']
    ))

like_params.append(LikelihoodParameter(
    name=f'beta_{like.theory_iz}',
    min_value=0.0,
    max_value=3.0,    
    ini_value = 1.5, # theory.default_param_dict[f'beta_{like.theory_iz}'],
    value=theory.default_param_dict[f'beta_{like.theory_iz}']
    # Gauss_priors_width=.5
    ))

linear = True
if linear:
    like_params.append(LikelihoodParameter(
        name=f'q1_{like.theory_iz}',
        value =0
    ))

    like_params.append(LikelihoodParameter(
        name=f'q2_{like.theory_iz}',
        value =0
    ))

    like_params.append(LikelihoodParameter(
        name=f'kp_{like.theory_iz}',
        value =10000
    ))
for like_param in like_params:
    print(like_param.name, like_param.value)

# %%
mini = IminuitMinimizer(like, like_params, [f'bias_{iz_choice}', f'beta_{iz_choice}'], verbose=True)

# %%
# iz = 0
# for i in range(3):
#     plt.imshow(MockData.cov_ZAM[iz,i])
#     plt.colorbar()
#     plt.show()
#     plt.imshow(np.linalg.inv(MockData.cov_ZAM[iz,i]))
#     plt.colorbar()
#     plt.show()
    

# %%
mini.minimize()

# %%
mini.chi2()

# %%
mini.chi2(return_all=True)

# %%
mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, xlim=[-.01, .4], datalabel="True-cont mock", theorylabel='Best fit', show=True)

# %%
mini.fit_probability()

# %%
mini.best_fit_value(f'bias_{iz_choice}', return_hesse=True), mini.best_fit_value(f'beta_{iz_choice}', return_hesse=True)

# %%
mini.plot_ellipses(f"bias_{like.data_iz}", f"beta_{like.data_iz}", nsig=3, cube_values=False, true_vals={f'bias_{like.data_iz}':like_parameter_by_name(like_params, f'bias_{like.data_iz}').value, f'beta_{like.data_iz}':like_parameter_by_name(like_params, f'beta_{like.data_iz}').value}, true_val_label='Laura fit')
plt.title(f"z={z_choice}")
# mini.plot_ellipses("bias_1", "beta_1", nsig=3, cube_values=False, true_vals={'bias_1':like_parameter_by_name(like_params, 'bias_1').value, 'beta_1':like_parameter_by_name(like_params, 'beta_1').value}, true_val_label='Laura fit')

# %%
import os

mini.save_results(outfile=os.path.splitext(os.path.basename(data))[0]+f"_iminuit_{z_choice}")

# %%
# check results
outfile = np.load(f"/global/common/software/desi/users/mlokken/cupix/data/fitter_results/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50_iminuit_{z_choice}.npz")
for key in outfile.keys():
    print(key)
    print(outfile[key])

# %%
