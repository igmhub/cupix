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

# bin_type = 'unbinned'
bin_type = 'medium_binned'
# bin_type = 'coarse_binned'
if bin_type == 'unbinned':
    bin_label='unbinned'
    ntheta = 20
elif bin_type=='coarse_binned':
    bin_label='binned'
    ntheta = 5
elif bin_type=='medium_binned':
    bin_label='binned'
    ntheta=18

# %%
# ls /global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/tru_cont/tru_cont_binned_*

# %%
if analysis_type == 'stack':
    # MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/{mock_type}/{mock_type}_{bin_label}_out_px-zbins_2-thetabins_{ntheta}_w_res_avg50.hdf5", theta_min_cut_arcmin=10, kM_min_cut_AA=.03, kM_max_cut_AA=1, km_max_cut_AA=1.2)
    MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/{mock_type}/{mock_type}_bf2_{bin_label}_out_px-zbins_4-thetabins_{ntheta}_w_res_avg50_ncov.hdf5", theta_min_cut_arcmin=0, kM_min_cut_AA=.03, kM_max_cut_AA=1, km_max_cut_AA=1.2)
    MockData.cov_ZAM *= 50
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
like = Likelihood(MockData, theory, z=2.2, verbose=False)

# %% [markdown]
# First, plot the CF best-fit theory model on top of the stack

# %%
like.plot_px(multiply_by_k=True, every_other_theta=False, xlim=[-.01, .4], datalabel="Mock Data", theorylabel=r"$\xi$ best-fit", show=True)

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values
arinyo_par_names = theory.arinyo_par_names

like_params = []
like_params.append(LikelihoodParameter(
    name=f'bias_{like.theory_iz}',
    min_value=-.13,
    max_value=-.1,
    ini_value=-0.11,
    value =theory.default_param_dict[f'bias_{like.theory_iz}']
    ))

like_params.append(LikelihoodParameter(
    name=f'beta_{like.theory_iz}',
    min_value=1.3,
    max_value=2.0,    
    ini_value = 1.5,
    value=theory.default_param_dict[f'beta_{like.theory_iz}']
    # Gauss_priors_width=.5
    ))

linear = False
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

# %%
mini = IminuitMinimizer(like, like_params, ['bias_0', 'beta_0'], verbose=False)

# %%
mini.minimize()

# %%
mini.plot_best_fit(multiply_by_k=True, every_other_theta=False, xlim=[-.01, .4], datalabel="Mock Data", theorylabel='Best fit', show=True)

# %%
mini.plot_ellipses("bias_0", "beta_0", nsig=3, cube_values=False, true_vals={'bias_0':like_parameter_by_name(like_params, 'bias_0').value, 'beta_0':like_parameter_by_name(like_params, 'beta_0').value}, true_val_label='Laura fit')
# mini.plot_ellipses("bias_1", "beta_1", nsig=3, cube_values=False, true_vals={'bias_1':like_parameter_by_name(like_params, 'bias_1').value, 'beta_1':like_parameter_by_name(like_params, 'beta_1').value}, true_val_label='Laura fit')

# %%
for p in like_params:
    if p.name in like.free_param_names:
        print(mini.minimizer.limits[p.name])
        print(p.value_from_cube(np.asarray(mini.minimizer.limits[p.name])))


# %%
results_dict = {}
for parname in like.free_param_names:
    bestfit, err = mini.best_fit_value(parname, return_hesse=True)
    results_dict[parname] = bestfit
    results_dict[parname+'_err'] = err
covariance = mini.minimizer.covariance
results_dict['cov'] = covariance

## save the following for the sake of plotting ellipses:
ix = like.index_by_name("beta")
iy = like.index_by_name("bias")
print(ix,iy)

# find out best-fit values, errors and covariance for parameters
val_x = mini.minimizer.values[ix]
val_y = mini.minimizer.values[iy]
sig_x = mini.minimizer.errors[ix]
sig_y = mini.minimizer.errors[iy]
r = mini.minimizer.covariance[ix, iy] / sig_x / sig_y
results_dict['r']=r
prob = like.fit_probability(mini.minimizer.values)
results_dict['prob'] = prob
chi2 = like.get_chi2(mini.minimizer.values)
results_dict['chi2'] = chi2
repo = os.path.dirname(cupix.__path__[0])
savefile = f"iminuit_{analysis_type}_{mock_type}_{bin_type}_mocks_bias_beta_z{z[iz_choice]}.npz"
savedir = os.path.join(repo, "data", "fitter_results")
savepath = os.path.join(savedir,savefile)
print(savepath)
save_analysis_npz(results_dict, savepath)

# %%
# ls -lrth /global/common/software/desi/users/mlokken/cupix/data/fitter_results/iminuit_stack_tru_cont_binned_mocks_bias_beta_z2.2.npz


# %%
like.ndeg(0)

# %%
mini.minimizer.covariance

# %%
data = np.load("/global/common/software/desi/users/mlokken/cupix/data/fitter_results/iminuit_stack_tru_cont_binned_mocks_bias_beta_z2.2.npz")


# %%
data['chi2']

# %%
mini.plot_best_fit(multiply_by_k=True, ylim2=[-1,1], every_other_theta=True, show=True,  title=f"Redshift {MockData.z[iz_choice]}, {mock_type}, {analysis_type}", theorylabel=rf'Model from best-fit $\xi$', datalabel='Mock 0', xlim=[0,0.4], residual_to_theory=True)


# %%
mini.best_fit_value("bias"),mini.best_fit_value("beta")

# %%
print(mini.best_fit_value('bias', return_hesse=True))

# %%

# %%
mini.plot_ellipses("bias", "beta", nsig=2, cube_values=False, true_vals={"bias":cf_bias, "beta":cf_beta}, true_val_label=r"$\xi$ best-fit")

# %% [markdown]
# # Detail the rebinning

# %%
mock_type = 'tru_cont'
analysis_type = 'stack'
bin_type = 'binned'
if bin_type == 'unbinned':
    ntheta = 20
else:
    ntheta = 5

if analysis_type == 'stack':
    MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/{mock_type}/{mock_type}_{bin_type}_out_px-zbins_2-thetabins_{ntheta}_w_res_avg50.hdf5", theta_min_cut_arcmin=14, kmax_cut_AA=1)
    # rescale the cov matrix to be from 1 mock
    # MockData.cov_ZAM *= np.sqrt(50)
elif analysis_type == 'single':
    MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-0/{mock_type}/{bin_type}_out_px-zbins_2-thetabins_{ntheta}_w_res.hdf5", theta_min_cut_arcmin=14, kmax_cut_AA=1)
# choose a redshift bin to analyse
iz_choice = 0
z = MockData.z

like_binned = Likelihood(MockData, theory_AA, free_param_names=["bias","beta","q1"], iz_choice=iz_choice, like_params=like_params)

# %%
bin_type = 'unbinned'
if bin_type == 'unbinned':
    ntheta = 20
else:
    ntheta = 5

if analysis_type == 'stack':
    MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/{mock_type}/{mock_type}_{bin_type}_out_px-zbins_2-thetabins_{ntheta}_w_res_avg50.hdf5", theta_min_cut_arcmin=14, kmax_cut_AA=1)
    # rescale the cov matrix to be from 1 mock
    # MockData.cov_ZAM *= np.sqrt(50)
elif analysis_type == 'single':
    MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-0/{mock_type}/{bin_type}_out_px-zbins_2-thetabins_{ntheta}_w_res.hdf5", theta_min_cut_arcmin=14, kmax_cut_AA=1)
# choose a redshift bin to analyse

like_unbinned = Likelihood(MockData, theory_AA, free_param_names=["bias","beta","q1"], iz_choice=iz_choice, like_params=like_params)


# %%
def plot_one_bin(binnedlike, unbinnedlike, iz, itheta, like_params, multiply_by_k=True, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None, ylim=None, ylim2=None, xlim=None, title=None, residual_to_theory=False):
    """Plot the Px data and theory for one bin, including the unbinned thetas.
    """
    
    plt.rcParams.update({'font.size': 20})
    # plot all theta on one, easily distinguishable colors
    colors = plt.cm.tab10(np.linspace(0,1,len(binnedlike.data.theta_min_A_arcmin)))
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    k = (binnedlike.data.k_M_edges[:-1]+binnedlike.data.k_M_edges[1:])/2.
    k_unbinned = (unbinnedlike.data.k_M_edges[:-1]+unbinnedlike.data.k_M_edges[1:])/2.
    print(len(k_unbinned))
    skip = 1
    if every_other_theta:
        skip = 2
    if multiply_by_k:
        factor = k
        factor_unbinned = k_unbinned
        ylabel = r'$k P_\times$'
    else:
        factor = 1.0
        ylabel = r'$P_\times$ [$\AA$]'
    
    errors = np.diag(binnedlike.data.cov_ZAM[iz, itheta, :, :])**0.5
    div = errors
    divname = 'errors'
    
    theory_iA_iz = binnedlike.get_convolved_Px_AA(iz,itheta,like_params)
    if residual_to_theory:
        div = theory_iA_iz
        divname = 'theory'
    

    mintheta,maxtheta = binnedlike.data.theta_min_A_arcmin[itheta], binnedlike.data.theta_max_A_arcmin[itheta]
    itheta_small_flag = binnedlike.data.B_A_a[itheta].astype(bool)
    itheta_small_toplot = np.arange(len(unbinnedlike.data.theta_centers_arcmin))[itheta_small_flag]
    theta_small_toplot_vals = unbinnedlike.data.theta_min_A_arcmin[itheta_small_toplot]
    print(itheta_small_toplot, theta_small_toplot_vals)
    theory_iA_iz_unb  = unbinnedlike.get_convolved_Px_AA(iz,itheta_small_toplot,like_params)
    
    for i, itheta_small in enumerate(itheta_small_toplot):
        errors_unbinned = np.diag(unbinnedlike.data.cov_ZAM[iz, itheta_small, :, :])**0.5
        ax[0].errorbar(k_unbinned, unbinnedlike.data.Px_ZAM[iz, itheta_small, :]*factor_unbinned, errors_unbinned*factor_unbinned, color=colors[i], linestyle='none', marker='o')
        ax[0].plot(k_unbinned, theory_iA_iz_unb[i]*factor_unbinned, color=colors[i], linestyle='solid')


    for itheta_small, px in enumerate(theory_iA_iz_unb):
        ax[0].plot(k_unbinned, px*factor_unbinned, color=colors[itheta_small], label=rf'$\theta={theta_small_toplot_vals[itheta_small]:.2f}^\prime$')

    ax[0].errorbar(k, binnedlike.data.Px_ZAM[iz, itheta, :]*factor, errors*factor, color='magenta', linestyle='none', marker='o')
    ax[0].plot(k, theory_iA_iz*factor, color='magenta', linestyle='solid', label=r'$\theta_A={:.2f}^\prime$'.format(binnedlike.data.theta_centers_arcmin[itheta]))

    ax[1].plot(k, (binnedlike.data.Px_ZAM[iz, itheta, :]-theory_iA_iz)/div, color='magenta', marker='o', linestyle='none')

    ax[1].axhline(0, color='black', linestyle='dashed', linewidth=1)
    ax[1].set_xlabel(r'$k [\AA^{-1}]$')
    ax[0].set_ylabel(ylabel)
    ax[0].legend()
    ax[1].set_ylabel(f'(Binned Data-Theory)/{divname}')
    # ax[1].legend()
    if ylim2 is None:
        ax[1].set_ylim([-3,3])
    else:
        ax[1].set_ylim(ylim2)
    if ylim is not None:
        ax[0].set_ylim(ylim)
    if xlim is not None:
        ax[1].set_xlim(xlim)
    
    handles, labels = ax[0].get_legend_handles_labels()
    if theorylabel is None:
        theorylabel = 'Theory prediction, windowed'
    if datalabel is None:
        datalabel='Data'
    if title is not None:
        plt.suptitle(title)
    handles.append(plt.Line2D([], [], color='black', linestyle='solid', label=theorylabel))
    handles.append(plt.Line2D([], [], color='black', marker='o', linestyle='none', label=datalabel))
    ax[0].legend(handles=handles, loc='upper right', fontsize='small')
    if plot_fname is not None:
        plt.savefig(plot_fname + ".pdf")
        plt.savefig(plot_fname + ".png")
    else:
        if show:
            plt.show()
    return

# %%
plot_one_bin(like_binned, like_unbinned, iz_choice, 0, like_params, multiply_by_k=True, ylim2=[-1,1], every_other_theta=True, show=True,  title=f"Redshift {MockData.z[iz_choice]}, {mock_type}, {analysis_type}", theorylabel=rf'Model from best-fit $\xi$', datalabel='Stacked measurement', xlim=[0,1], residual_to_theory=True)


# %% [markdown]
# ## True-continuum mocks

# %%
MockData_trucont = DESI_DR2("/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/tru_cont/tru_cont_binned_out_px-zbins_2-thetabins_5_w_res_avg50_ncov.hdf5", theta_min_cut_arcmin=14, kmax_cut_AA=1)

# %%
like = Likelihood(MockData_trucont, theory_AA, free_param_names=["bias","beta"], iz_choice=iz_choice, like_params=like_params)

# %% [markdown]
# First, plot the CF best-fit theory model on top of the stack

# %%
like.plot_px(iz_choice, like_params, multiply_by_k=False, ylim=[-0.00025,0.03], ylim2=[-10,10], every_other_theta=False, show=True,  title=f"Redshift {MockData_raw.z[iz_choice]}", theorylabel=f'Model from best-fit $\chi$', datalabel='Stack on true-continuum mocks')

# %%
mini = IminuitMinimizer(like, verbose=True)
mini.minimize()

# %%
final_values = np.asarray([mini.best_fit_value(pname) for pname in like.free_param_names])
ini_values   = np.asarray([p.ini_value for p in like_params if p.name in like.free_param_names])
priors       = np.asarray([p.Gauss_priors_width for p in like_params if p.name in like.free_param_names])
hesse_errs   = np.asarray([mini.best_fit_value(pname, return_hesse=True)[1] for pname in like.free_param_names])
true_values  = np.asarray([p.value for p in like_params if p.name in like.free_param_names])
plt.plot(like.free_param_names, ini_values, 'o', label='Initial values')
plt.errorbar(like.free_param_names, final_values, yerr= hesse_errs, fmt='x', label='Final values with Hesse errors')
plt.plot(like.free_param_names, true_values, '*', color='green', label=r"$\xi$ best-fit", markersize=10)
# plot the priors as a shaded vertical band per parameter
for i, param in enumerate(like.free_param_names):
    plt.fill_between([i-0.2, i+0.2], final_values[i]-priors[i], final_values[i]+priors[i], color='gray', alpha=0.3, label='Priors' if i==0 else "")
plt.ylabel("Parameter values")
plt.xlabel("Parameters")
plt.legend(fontsize=15)


# %%
prob = like.fit_probability(mini.minimizer.values)
prob

# %%
mini.plot_best_fit(multiply_by_k=True, xlim=[-.01, 1], ylim=[-0.00025,0.00175], ylim2=[-10,10], every_other_theta=False, show=True,  title=f"Redshift {MockData_raw.z[iz_choice]}", theorylabel=rf'Model from best-fit $P_\times$', datalabel='Stack on raw mocks')

# %%
mini.plot_ellipses("bias", "beta", nsig=2, cube_values=False, true_vals={"bias":-0.115, "beta":1.55}, true_val_label=r"$\xi$ best-fit")

# %%
mini.best_fit_value("bias")

# %%
