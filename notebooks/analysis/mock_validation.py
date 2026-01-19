# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: forestflow
#     language: python
#     name: forestflow
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from lace.cosmo import camb_cosmo, fit_linP
import matplotlib.pyplot as plt
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import scipy
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
from astropy.io import fits
import cupix
import os
omnuh2 = 0.0006
mnu = omnuh2 * 93.14
H0 = 67.36
omch2 = 0.12
ombh2 = 0.02237
As = 2.1e-9
ns = 0.9649
nrun = 0.0
w = -1.0
omk = 0
fid_cosmo = {
    'H0': H0,
    'omch2': omch2,
    'ombh2': ombh2,
    'mnu': mnu,
    'omk': omk,
    'As': As,
    'ns': ns,
    'nrun': nrun,
    'w': w
}
theory_AA = None

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
bin_type = 'binned'
if bin_type == 'unbinned':
    ntheta = 20
else:
    ntheta = 5

# %%
if analysis_type == 'stack':
    MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/{mock_type}/{mock_type}_{bin_type}_out_px-zbins_2-thetabins_{ntheta}_w_res_avg50.hdf5", theta_min_cut_arcmin=14, kmax_cut_AA=1)
    # rescale the cov matrix to be from 1 mock
    print(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/{mock_type}/{mock_type}_{bin_type}_out_px-zbins_2-thetabins_{ntheta}_w_res_avg50.hdf5")
    # MockData.cov_ZAM *= np.sqrt(50)
elif analysis_type == 'single':
    MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-0/{mock_type}/{bin_type}_out_px-zbins_2-thetabins_{ntheta}_w_res.hdf5", theta_min_cut_arcmin=14, kmax_cut_AA=1)
# choose a redshift bin to analyse
iz_choice = 0
z = np.array([MockData.z[iz_choice]])


# %%
theory_AA=None

# %%
# Load emulator
if theory_AA is None: # only do this once per notebook
    sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
    cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)
    ffemu = FF_emulator(z, fid_cosmo, cc)
    ffemu.kp_Mpc = 1 # set pivot point

    theory_AA = set_theory(ffemu, k_unit='iAA')
    theory_AA.set_fid_cosmo(z)
    theory_AA.emulator = ffemu

# %%
# Load Laura's CF fits
with fits.open(f"/global/cfs/cdirs/desicollab/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-0/jura-124/raw_bao_unblinding/fits/output_fitter-z-bins/bin_{z[0]}/lyaxlya.fits") as zbin_cf_file:
    zbin_cf_fit = zbin_cf_file[1].header
    cf_bias = zbin_cf_fit['bias_LYA']
    cf_beta = zbin_cf_fit['beta_LYA']
    cf_q1   = zbin_cf_fit['dnl_arinyo_q1']
    cf_kv   = zbin_cf_fit['dnl_arinyo_kv']
    cf_av   = zbin_cf_fit['dnl_arinyo_av']
    cf_bv   = zbin_cf_fit['dnl_arinyo_bv']
    cf_kp   = zbin_cf_fit['dnl_arinyo_kp']

# %%
# # original Laura fits
# like_params = []
# like_params.append(LikelihoodParameter(
#     name='bias',
#     min_value=-1.0,
#     max_value=1.0,
#     value=-0.115,
#     ini_value=-0.095,
#     Gauss_priors_width=.5
#     ))
# like_params.append(LikelihoodParameter(
#     name='beta',
#     min_value=0.0,
#     max_value=3.0,    
#     value = 1.55,
#     ini_value = 1.35,
#     Gauss_priors_width=1
#     ))
# like_params.append(LikelihoodParameter(
#     name='q1',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.1112,
#     value = 0.1112,
#     Gauss_priors_width=0.5
#     ))
# like_params.append(LikelihoodParameter(
#     name='kvav',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.0001**0.2694,
#     value = 0.0001**0.2694
#     ))
# like_params.append(LikelihoodParameter(
#     name='av',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.2694,
#     value = 0.2694
#     ))
# like_params.append(LikelihoodParameter(
#     name='bv',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.0002,
#     value = 0.0002
#     ))
# like_params.append(LikelihoodParameter(
#     name='kp',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.5740,
#     value = 0.5740
#     ))

# %%
# new Laura fit parameters

like_params = []

like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=1.0,
    value=cf_bias,
    ini_value=-0.3,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.0,
    max_value=3.0,    
    value = cf_beta,
    ini_value = 2.3,
    Gauss_priors_width=1
    ))
like_params.append(LikelihoodParameter(
    name='q1',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.1112,
    value = cf_q1,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='kvav',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.0001**0.2694,
    value = cf_kv**cf_av
    ))
like_params.append(LikelihoodParameter(
    name='av',
    min_value=0.0,
    max_value=2.0,
    ini_value = 0.2694,
    value = cf_av
    ))
like_params.append(LikelihoodParameter(
    name='bv',
    min_value=0.0,
    max_value=2.0,
    ini_value = 0.0002,
    value = cf_bv
    ))
like_params.append(LikelihoodParameter(
    name='kp',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.5740,
    value = cf_kp
    ))

# %%
like = Likelihood(MockData, theory_AA, free_param_names=["bias", "beta"], iz_choice=iz_choice, like_params=like_params)

# %% [markdown]
# First, plot the CF best-fit theory model on top of the stack

# %%
MockData.theta_centers_arcmin

# %%
like.plot_px(iz_choice, like_params, multiply_by_k=True, ylim2=[-1,6], every_other_theta=False, show=True,  title=f"Redshift {MockData.z[iz_choice]}, {mock_type}, {analysis_type}", theorylabel=rf'Model from IFAE-QL best-fit $\xi$', datalabel=f'{analysis_type} measurement', xlim=[0,.5], residual_to_theory=False)


# %%
mini = IminuitMinimizer(like, verbose=True)
mini.minimize()

# %%
for p in like_params:
    if p.name in like.free_param_names:
        print(mini.minimizer.limits[p.name])
        print(p.value_from_cube(np.asarray(mini.minimizer.limits[p.name])))
    

# %%
final_values = np.asarray([mini.best_fit_value(pname) for pname in like.free_param_names])
ini_values   = np.asarray([p.ini_value for p in like_params if p.name in like.free_param_names])
# priors       = np.asarray([p.Gauss_priors_width for p in like_params if p.name in like.free_param_names])
priors       = np.asarray([p.value_from_cube(np.asarray(mini.minimizer.limits[p.name])) for p in like_params if p.name in like.free_param_names])
hesse_errs   = np.asarray([mini.best_fit_value(pname, return_hesse=True)[1] for pname in like.free_param_names])
true_values  = np.asarray([p.value for p in like_params if p.name in like.free_param_names])
plt.plot(like.free_param_names, ini_values, 'o', label='Initial values')
plt.errorbar(like.free_param_names, final_values, yerr= hesse_errs, fmt='x', label='Final values with Hesse errors')
plt.plot(like.free_param_names, true_values, '*', color='green', label=r"$\xi$ best-fit", markersize=10, alpha=.5)
# plot the priors as a shaded vertical band per parameter
for i, param in enumerate(like.free_param_names):
    plt.fill_between([i-0.2, i+0.2], priors[i][0], priors[i][1], color='gray', alpha=0.3, label='Limits' if i==0 else "")
plt.ylabel("Parameter values")
plt.xlabel("Parameters")
plt.legend(fontsize=15, loc='upper left', ncol=2)
plt.ylim([-2,5])


# %%
def save_analysis_npz(results, filename="analysis_results.npz"):
    """
    results: list or dict of per-analysis dictionaries
    """
    out = {}

    if isinstance(results, list):
        for i, r in enumerate(results):
            out[f'analysis-{i}'] = r
    else:  # dict
        for k, r in results.items():
            out[str(k)] = r

    # Save each dict as an object
    np.savez(filename, **out, allow_pickle=True)


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
