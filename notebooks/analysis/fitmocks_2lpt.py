# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
import matplotlib.pyplot as plt
import time
import copy
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
from cupix.inference.sampling_funcs import prepare_free_parameters

# %%
# In this notebook we will work with a single z bin
iz=0

# setup cosmology (should check this is the right cosmology in the mocks)
cosmodict = {'ombh2':0.02237, 'omch2':0.12, 'omk':0, 'h':0.6736, 'As':2.0830e-9, 'ns':0.9649, 'w0':-1, 'wa':0}

cosmo = cosmology.Cosmology(cosmodict)
# starting point for Lya bias parameters in mocks
# default_lya_model = 'pressure_only_arinyo_from_colore'
default_lya_model = 'best_fit_arinyo_from_colore'

# path to mocks
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/colore/"


# %% [markdown]
# # Load all data and compare windows
#

# %%
# ls /global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/colore/analysis-200

# %%
def plot_onetheta_bin(data_list, iz, it_M,  data_labels=None, colors=None):
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    c = 0
    if data_labels is None:
        data_labels = ['data {}'.format(i) for i in range(len(data_list))]
    for data in data_list:
        theta_A_min = data.theta_min_A_arcmin
        theta_A_max = data.theta_max_A_arcmin
        
        # 1D array with measured Px, length Nk_M
        Px = data.Px_ZAM[iz][it_M]
        k_M = data.k_M_centers_AA
        # get also errorbars
        sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_M]))
        
        plt.errorbar(k_M, Px, sig_Px, label=data_labels[c])
        c += 1
    plt.legend()
    title = '{} < theta < {}'.format(theta_A_min[it_M], theta_A_max[it_M])
    plt.title(title)

def plot_px_and_window_residuals(measurement_files, measurement_labels, iz, itheta):
    fnames = [mockdir + file for file in measurement_files]
    colors = plt.cm.tab10(np.linspace(0, 1, len(measurement_files)))
    linestyles = ['solid', 'dashed', 'dotted']
    data_list = []
    kM_max_cut_AA = 1.0 # 0.7
    km_max_cut_AA = 1.1*kM_max_cut_AA
    theta_min_cut_arcmin = 0 # 25.0
    for m in fnames:
        print(m)
        data_config = {'data_file':m, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
        data = DESI_DR2(data_config)
        data_list.append(data)
    plot_onetheta_bin(data_list, iz, itheta,  data_labels=measurement_labels, colors=colors)
    plt.show()
    plt.clf()
    z = data.z[iz]
    theories = []
    likes = []
    for data in data_list:
        theory_config = {'verbose': True, 'default_lya_model': default_lya_model, 'include_continuum':False}
        theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
        like = Likelihood(theory=theory, data=data, iz=iz)
        theories.append(theory)
        likes.append(like)
    
    model_px_list = []
    
    for l, like in enumerate(likes):
        model_px = like.get_convolved_px()
        theory_iA = model_px[itheta]
        model_px_list.append(theory_iA)
        plt.plot(data_list[l].k_M_centers_AA, theory_iA, color=colors[l], linewidth=1, label=measurement_labels[l])
        
    plt.legend()
    plt.show()
    plt.clf()

    # plot residuals to first convolved theory in the list
    plt.axhspan(-.03,.03, label='3%', color='grey', alpha=.1)
    for i in range(len(model_px_list)-1):
        plt.plot(data_list[i].k_M_centers_AA, (model_px_list[i+1] - model_px_list[0])/model_px_list[0], color=colors[i+1], label=measurement_labels[i+1], ls=linestyles[i])
    
    plt.xlim([0, 1])
    plt.title(rf"$z={z}, \theta={data_list[0].theta_centers_arcmin[itheta]}$")

    plt.ylabel(f'residual w.r.t. {measurement_labels[0]}', fontsize=15)
    plt.xlabel('k [1/AA]')
    plt.legend(fontsize=12, ncol=2)
    # plt.ylim([-1,1])
    


# %%
files = ["analysis-200/uncontaminated/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5",
         "analysis-200/contaminated/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5",
         "analysis-200/contaminated/drop_BALs/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5",
         "analysis-200/contaminated/drop_DLAs_and_BALs/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"]
labels = ["uncont", "fully_cont", "drop_BAL", "drop_DLA+BAL"]
iz = 2
itheta = 17
plot_px_and_window_residuals(files, labels, iz, itheta)

# %%
# speed-up code by only looking at low kpar
kM_max_cut_AA = 1.0 # 0.7
km_max_cut_AA = 1.1*kM_max_cut_AA
theta_min_cut_arcmin = 0 # 25.0
fname = mockdir + "analysis-200/tru_cont/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
trucont_data = DESI_DR2(data_config)
fname = mockdir + "analysis-200/uncontaminated/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
uncont_data = DESI_DR2(data_config)
fname = mockdir + f"partially_contaminated/analysis-200/onlydlas_nomask/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
DLA_contam_data = DESI_DR2(data_config)
fname = mockdir + f"partially_contaminated/analysis-200/onlydlas_mask/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
DLA_contam_data_mask = DESI_DR2(data_config)
fname = mockdir + f"partially_contaminated/analysis-200/onlymetals/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
metal_contam = DESI_DR2(data_config)
fname = mockdir + f"analysis-200/contaminated/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
fully_contam = DESI_DR2(data_config)
fname = mockdir + f"analysis-200/contaminated/drop_BALs/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
fully_contam_drop_BAL = DESI_DR2(data_config)
fname = mockdir + f"analysis-200/contaminated/drop_DLAs_and_BALs/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"
data_config = {'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':theta_min_cut_arcmin}
fully_contam_drop_BALDLA = DESI_DR2(data_config)

# %%
analysis_type_labels = ['trucont', 'uncont', 'DLA contam no mask', 'DLA contam mask', 'metal contam', 'fully contam', 'drop BAL', 'drop BAL+DLA']

plot_onetheta_bin([trucont_data, uncont_data, DLA_contam_data, DLA_contam_data_mask, metal_contam, fully_contam, fully_contam_drop_BAL, fully_contam_drop_BALDLA], iz=iz, it_M=10, data_labels=analysis_type_labels)

# %%
# generate one example theory and apply each window function to it
iz = 2
z = trucont_data.z[iz]
theory_config = {'verbose': True, 'default_lya_model': default_lya_model, 'include_continuum':False}
theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
like_trucont = Likelihood(theory=theory, data=trucont_data, iz=iz)
like_uncont = Likelihood(theory=theory, data=uncont_data, iz=iz)
like_DLA_contam = Likelihood(theory=theory, data=DLA_contam_data, iz=iz)
like_DLA_contam_mask = Likelihood(theory=theory, data=DLA_contam_data_mask, iz=iz)
like_metal_contam = Likelihood(theory=theory, data=metal_contam, iz=iz)
like_fully_contam = Likelihood(theory=theory, data=fully_contam, iz=iz)
like_drop_bal = Likelihood(theory=theory, data=fully_contam_drop_BAL, iz=iz)
like_drop_baldla = Likelihood(theory=theory, data=fully_contam_drop_BALDLA, iz=iz)
# likes = [like_trucont, like_uncont, like_DLA_contam, like_DLA_contam_mask, like_metal_contam, like_fully_contam, like_drop_bal, like_drop_baldla]
likes = [like_trucont, like_uncont, like_fully_contam, like_drop_bal, like_drop_baldla]

# %%
analysis_type_labels = ['trucont', 'uncont', 'fully contam', 'drop BAL', 'drop BAL+DLA']
colors = plt.cm.tab10(np.linspace(0, 1, len(likes)))
it_A = 19
l = 0
model_px_list = []
for like in likes:
    model_px = like.get_convolved_px()
    theory_iA = model_px[it_A]
    model_px_list.append(theory_iA)
    plt.plot(DLA_contam_data.k_M_centers_AA, theory_iA, color=colors[l], linewidth=2, label=analysis_type_labels[l])
    
    # like.plot_px_windowed_theory()
    l += 1
plt.legend()

# %%
# compare_to = 'true_cont'
compare_to = 'uncont'
if compare_to == 'uncont':
    ic = 2
else:
    ic = 1
# plot residuals
plt.axhspan(-.03,.03, label='3%', color='grey', alpha=.1)
for i in range(len(model_px_list)-ic):
    plt.plot(DLA_contam_data.k_M_centers_AA, (model_px_list[i+ic] - model_px_list[ic-1])/model_px_list[ic-1], color=colors[i+ic], label=analysis_type_labels[i+ic])
plt.xlim([0, 1])
plt.title(rf"$\theta={uncont_data.theta_centers_arcmin[it_A]}$")

plt.ylabel(f'residual w.r.t. {compare_to}', fontsize=15)
plt.xlabel('k [1/AA]')
plt.legend(fontsize=12, ncol=2)
plt.ylim([-.1,.1])

# %% [markdown]
# # Fit true-continuum mocks

# %%
fname = mockdir + "analysis-200/tru_cont/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"
# speed-up code by only looking at low kpar
z = trucont_data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))


# %%
theory_config = {'verbose': True, 'default_lya_model': default_lya_model, 'include_continuum':False}
theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)

# %%
like = Likelihood(data=trucont_data, theory=theory, iz=iz, config={'verbose':False})

# %%
like.plot_px(params={'bias':-.13, 'beta':1.4, 'kp_Mpc':1.2, 'av':.1, 'bv':.1}, every_other_theta=True, multiply_by_k=False)

# %%
# # set broader priors than the defaults, since IFAE-QL mocks are different

params_config = {
    'bias': {'gauss_prior_width': None, 'gauss_prior_mean':None},
    'beta': {'gauss_prior_width': None, 'gauss_prior_mean':None},
    # 'q1': {'gauss_prior_width': None, 'gauss_prior_mean':None},
}

#     'kp_Mpc': {'gauss_prior_width': None, 'gauss_prior_mean':None},
#     'q1': {'gauss_prior_width': None, 'gauss_prior_mean':None},
#     'av': {'gauss_prior_width': None, 'gauss_prior_mean':None},
#     'bv': {'gauss_prior_width': None, 'gauss_prior_mean':None},
#

free_params = prepare_free_parameters(['bias','beta'], theory, theory_config, params_config=params_config) #'q1'
# free_params = prepare_free_parameters(['bias','beta','kp_Mpc', 'q1', 'av', 'bv'], theory, theory_config, params_config=params_config)
for p in free_params:
    # print all attributes
    print(p.__dict__)
    # set initial value for bias / beta based on best-fit values from Laura
# ini_bias = theory.lya_model.default_lya_params['bias']
# ini_beta = theory.lya_model.default_lya_params['beta']
# par_bias = FreeParameter(
#     name='bias',
#     min_value=-0.5,
#     max_value=-0.01,
#     ini_value=ini_bias,
#     delta=0.01,   
# )
# par_beta = FreeParameter(
#     name='beta',
#     min_value=0.1,
#     max_value=5.0,
#     ini_value=ini_beta,
#     delta=0.1,
# )

# # add other free parameters (without prior values for now)
# ini_q1 = theory.lya_model.default_lya_params['q1']
# par_q1 = FreeParameter(
#     name='q1',
#     min_value=0.0,
#     max_value=5.0,
#     ini_value=ini_q1,
#     delta=0.01
# )
# ini_av = theory.lya_model.default_lya_params['av']
# par_av = FreeParameter(
#     name='av',
#     min_value=0.0,
#     max_value=5.0,
#     ini_value=ini_av,
#     delta=0.01
# )
# ini_bv = theory.lya_model.default_lya_params['bv']
# par_bv = FreeParameter(
#     name='bv',
#     min_value=0.0,
#     max_value=5.0,
#     ini_value=ini_bv,
#     delta=0.01
# )
# ini_kp = theory.lya_model.default_lya_params['kp_Mpc']
# par_kp = FreeParameter(
#     name='kp_Mpc',
#     min_value=0.0,
#     max_value=5.0,
#     ini_value=ini_kp,
#     delta=0.01
# )
# ini_kv = theory.lya_model.default_lya_params['kv_Mpc']
# par_kv = FreeParameter(
#     name='kv_Mpc',
#     min_value=0.0,
#     max_value=5.0,
#     ini_value=ini_kv,
#     delta=0.01
# )

# %%
# free_params = [par_bias, par_beta, par_q1, par_av, par_bv, par_kp, par_kv]

# %%
post = Posterior(like, free_params, config={'verbose':False})
true_cont_mini = Minimizer(post, config={'verbose':False})
true_cont_mini.silence()
start = time.time()
true_cont_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
true_cont_mini.print_results()

# %%
true_cont_mini.plot_best_fit(multiply_by_k=True, every_other_theta=False, residual_to_theory=True, include_chi2=True, xlim=[0,0.4])

# %%
true_cont_mini.plot_ellipse('bias','beta', label='true cont')
true_cont_mini.plot_corner()

# %% [markdown]
# # Fit uncontaminated mocks

# %%
fname = mockdir + "analysis-200/uncontaminated/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"
z = uncont_data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))


# %%
theory_config = {'verbose': False, 'default_lya_model': default_lya_model, 'include_continuum':True}
uncont_theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
print(uncont_theory.lya_model.default_lya_params)

# %%
like = Likelihood(data=uncont_data, theory=uncont_theory, iz=iz, config={'verbose':False})

# %% [markdown]
# Let's see how Laura's best-fit to IFAE-QL compares just for curiosity sake

# %%
like.plot_px(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
params_config = {'bias': {'min_value': -0.3, 'max_value': -0.01}, 'beta':{'min_value':0.1, 'max_value':3.0}, 'kp_Mpc':{'min_value':0.0, 'max_value':5.0}}
free_params = prepare_free_parameters(['bias','beta','kp_Mpc'], uncont_theory, theory_config, params_config=params_config) 


# %%
ini_pC = uncont_theory.cont_model.default_continuum_params['pC']

par_pC = FreeParameter(
    name='pC',
    min_value=0.01,
    max_value=2.0,
    ini_value=ini_pC,
    delta=0.01
)

free_params.append(par_pC)

# %%
for p in free_params:
    print(p.name, p.true_value, p.ini_value, p.delta, p.min_value, p.max_value)

# %%
post = Posterior(like, free_params, config={'verbose':False})
uncont_mini = Minimizer(post, config={'verbose':False})
uncont_mini.silence()
start = time.time()
uncont_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
uncont_mini.print_results()

# %%
uncont_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
uncont_mini.plot_ellipses('bias', 'beta')

# %%
uncont_mini.plot_ellipses('bias', 'kp_Mpc')

# %% [markdown]
# # Now use bias and beta and try to fit b_HCD from partially-contaminated mocks

# %%
mask_dlas = True
if mask_dlas:
    maskstr = 'nomask'
else:
    maskstr = 'mask'

# %%
fname = mockdir + f"partially_contaminated/analysis-200/onlydlas_{maskstr}/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"

z = DLA_contam_data_mask.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))


# %%
theory_config = {'verbose': True, 'default_lya_model': default_lya_model,
                 'include_continuum': True, 'include_hcd': True,
                 'include_metal': False, 'include_sky': False, 'bias':uncont_mini.get_best_fit_params()['bias'], 'beta':uncont_mini.get_best_fit_params()['beta'], 'kp_Mpc':uncont_mini.get_best_fit_params()['kp_Mpc'], 'pC':0.6474}
theory_parcon_wmask = Theory(z=z, fid_cosmo=cosmo, config=theory_config)


# %%
params_config = {'b_H': {'ini_value':-.04, 'min_value': -0.2, 'max_value': 0, 'delta':.002}, 'L_H_Mpc':{'min_value':0, 'max_value':20.0, 'ini_value':5, 'delta':.5}}
free_params = prepare_free_parameters(['b_H','L_H_Mpc'], theory_parcon_wmask, theory_config, params_config=params_config)

for p in free_params:
    print(p.name, p.true_value, p.ini_value, p.delta, p.min_value, p.max_value)

# %%
like_parcon_wmask = Likelihood(data=DLA_contam_data_mask, theory=theory_parcon_wmask, iz=iz, config={'verbose':True})
post_parcon_wmask = Posterior(like_parcon_wmask, free_params, config={'verbose':False})
parcon_wmask_mini = Minimizer(post_parcon_wmask, config={'verbose':False})
parcon_wmask_mini.silence()
start = time.time()
parcon_wmask_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
parcon_wmask_mini.print_results()

# %%
parcon_wmask_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
parcon_wmask_mini.plot_ellipses('b_H', 'L_H_Mpc')

# %% [markdown]
# # Now free bias, beta, kp and try to fit HCD params from partially-contaminated mocks

# %%
theory_config = {'verbose': True, 'default_lya_model': default_lya_model,
                 'include_continuum': True, 'include_hcd': True,
                 'include_metal': False, 'include_sky': False, 'kp_Mpc':uncont_mini.get_best_fit_params()['kp_Mpc'], 'pC':0.6474}
theory_parcon_wmask = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
params_config = {'b_H': {'ini_value':-.04, 'min_value': -0.2, 'max_value': 0, 'delta':.002}, 'L_H_Mpc':{'min_value':0, 'max_value':40.0, 'ini_value':15, 'delta':2},
                 'bias': {'min_value': -0.3, 'max_value': -.001}, 'beta':{'min_value':0.1, 'max_value':3.0}, 'kp_Mpc':{'min_value':0.0, 'max_value':5.0}}
free_params_dlacont = prepare_free_parameters(['bias','beta','b_H','L_H_Mpc'], theory_parcon_wmask, theory_config, params_config=params_config)

for p in free_params:
    print(p.name, p.true_value, p.ini_value, p.delta, p.min_value, p.max_value)

# %%
like_parcon_wmask = Likelihood(data=DLA_contam_data_mask, theory=theory_parcon_wmask, iz=iz, config={'verbose':True})
post_parcon_wmask_free = Posterior(like_parcon_wmask, free_params, config={'verbose':False})
parcon_wmask_free_mini = Minimizer(post_parcon_wmask_free, config={'verbose':False})
parcon_wmask_free_mini.silence()
start = time.time()
parcon_wmask_free_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
parcon_wmask_free_mini.print_results()

# %%
parcon_wmask_free_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
parcon_wmask_free_mini.plot_ellipses('b_H', 'L_H_Mpc')

# %%
theory_parcon_wmask.cont_model.default_hcd_params

# %% [markdown]
# # The same but with the addition of beta_H

# %%
params_config = {'beta_H': {'ini_value':.4, 'min_value': 0, 'max_value': 5, 'delta':.5},
                 'b_H': {'ini_value':-.04, 'min_value': -0.2, 'max_value': 0, 'delta':.002},
                   'L_H_Mpc':{'min_value':0, 'max_value':40.0, 'ini_value':5, 'delta':2},
                 'bias': {'min_value': -0.3, 'max_value': -0.01}, 'beta':{'min_value':0.1, 'max_value':3.0},
                   'kp_Mpc':{'min_value':0.0, 'max_value':5.0}}
free_params_dlacont_wbeta = prepare_free_parameters(['bias','beta','beta_H','b_H','L_H_Mpc'], theory_parcon_wmask, theory_config, params_config=params_config) # 'kp_Mpc',

for p in free_params:
    print(p.name, p.true_value, p.ini_value, p.delta, p.min_value, p.max_value)

# %%
post_parcon_wmask_freebeta = Posterior(like_parcon_wmask, free_params, config={'verbose':False})
parcon_wmask_freebeta_mini = Minimizer(post_parcon_wmask_freebeta, config={'verbose':False})
parcon_wmask_freebeta_mini.silence()
start = time.time()
parcon_wmask_freebeta_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
parcon_wmask_freebeta_mini.print_results()

# %%
parcon_wmask_freebeta_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
parcon_wmask_freebeta_mini.plot_ellipses('L_H_Mpc', 'beta_H')

# %% [markdown]
# # And the last step as well for unmasked DLAs
#

# %%
mask_dlas = False
if mask_dlas:
    maskstr = 'nomask'
else:
    maskstr = 'mask'

fname = mockdir + f"partially_contaminated/analysis-200/onlydlas_{maskstr}/bf3_binned_out_px-zbins_4-thetabins_20_w_res_w_p1d.hdf5"


z = DLA_contam_data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))

like_parcon_nomask = Likelihood(data=DLA_contam_data, theory=theory_parcon_wmask, iz=iz, config={'verbose':True})

# %%
params_config = {'b_H': {'ini_value':-.04, 'min_value': -0.1, 'max_value': 0, 'delta':.002}, 'L_H_Mpc':{'min_value':0, 'max_value':40., 'ini_value':15, 'delta':2},
                 'bias': {'min_value': -0.1, 'max_value': -0.01}, 'beta':{'min_value':0.1, 'max_value':3.0}, 'kp_Mpc':{'min_value':0.0, 'max_value':5.0}}
free_params = prepare_free_parameters(['bias','beta','kp_Mpc','b_H','L_H_Mpc'], theory_parcon_wmask, theory_config, params_config=params_config)

for p in free_params:
    print(p.name, p.true_value, p.ini_value, p.delta, p.min_value, p.max_value)

# %%
post_parcon_nomask = Posterior(like_parcon_nomask, free_params, config={'verbose':False})
parcon_nomask_mini = Minimizer(post_parcon_nomask, config={'verbose':False})
parcon_nomask_mini.silence()
start = time.time()
parcon_nomask_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
parcon_nomask_mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, residual_to_theory=True, include_chi2=True)

# %%
parcon_nomask_mini.print_results()

# %%
parcon_nomask_mini.plot_ellipses('bias', 'beta', color='red', label='DLA cont, no mask')


# %%
xrange = [-0.18, -0.1]
yrange = [.8, 2.5]
ax = parcon_nomask_mini.plot_ellipses('bias', 'beta', xrange=xrange, yrange=yrange, label='DLA cont, no mask')
ax = true_cont_mini.plot_ellipses('bias', 'beta', color='blue', xrange=xrange, yrange=yrange, label='true cont')
uncont_mini.plot_ellipses('bias', 'beta', ax=ax, color='green', xrange=xrange, yrange=yrange, label='uncont')
parcon_wmask_free_mini.plot_ellipses('bias', 'beta', ax=ax, color='orange', xrange=xrange, yrange=yrange, label='DLA cont, with mask')
parcon_nomask_mini.plot_ellipses('bias', 'beta', ax=ax, color='red', xrange=xrange, yrange=yrange, label='DLA cont, no mask')
parcon_wmask_freebeta_mini.plot_ellipses('bias', 'beta', ax=ax, color='purple', xrange=xrange, yrange=yrange, label=r'DLA cont, with mask, free $\beta_H$')

# %%
# xrange = [-.1,-.15]
# xrange = [-.045,0]
ax = parcon_nomask_mini.plot_ellipses('b_H', 'L_H_Mpc',  label='DLA cont, no mask', xrange=xrange)
parcon_wmask_free_mini.plot_ellipses('b_H', 'L_H_Mpc', xrange=xrange, ax=ax, color='orange', label='DLA cont, with mask')

# %% [markdown]
# # And finally, for fully-contaminated

# %%
theory_config_full = {'verbose': True, 'default_lya_model': default_lya_model,
                 'include_continuum': True, 'include_hcd': True,
                 'include_metal': True, 'include_sky': False, 'kp_Mpc':uncont_mini.get_best_fit_params()['kp_Mpc'], 'pC':0.6474}
theory_fullcon = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
like_fullcon = Likelihood(data=fully_contam, theory=theory_fullcon, iz=iz, config={'verbose':True})

# %%
params_config = {'b_H': {'ini_value':-.04, 'min_value': -0.1, 'max_value': 0, 'delta':.002}, 'L_H_Mpc':{'min_value':0, 'max_value':40., 'ini_value':15, 'delta':2},
                 'bias': {'min_value': -0.1, 'max_value': -0.01}, 'beta':{'min_value':0.1, 'max_value':3.0}, 'kp_Mpc':{'min_value':0.0, 'max_value':5.0}, 'b_X': {'ini_value':-0.005, 'min_value': -0.015, 'max_value': 0, 'delta':.002}}
free_params_wmetal = prepare_free_parameters(['bias','beta','kp_Mpc','b_H','L_H_Mpc', 'b_X'], theory_parcon_wmask, theory_config, params_config=params_config)

for p in free_params:
    print(p.name, p.true_value, p.ini_value, p.delta, p.min_value, p.max_value)
post_fullcon = Posterior(like_fullcon, free_params_wmetal, config={'verbose':False})
fullcon_mini = Minimizer(post_fullcon, config={'verbose':False})
fullcon_mini.silence()
start = time.time()
fullcon_mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# %%
# save minimizer results

outdir = "/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/partially_cont_2lpt_mocks"
true_cont_mini.save_results(outdir=outdir, outfile="true_cont_mini")
uncont_mini.save_results(outdir=outdir, outfile="uncont_mini")
parcon_wmask_free_mini.save_results(outdir=outdir, outfile="DLA_contam_mask_mini")
parcon_nomask_mini.save_results(outdir=outdir, outfile="DLA_contam_nomask_mini")
parcon_wmask_freebeta_mini.save_results(outdir=outdir, outfile="DLA_contam_mask_freebeta_mini")
fullcon_mini.save_results(outdir=outdir, outfile="fully_contam_mini")

# %%
results_dict_parcon_nomask = parcon_nomask_mini.get_results_dict()
results_dict_parcon_wmask = parcon_wmask_free_mini.get_results_dict()
results_dict_parcon_wmask_freebeta = parcon_wmask_freebeta_mini.get_results_dict()


# %%

# %%
from cupix.inference.minimize_posterior import plot_corner
fig, ax = plot_corner(results_dict_parcon_wmask,
    free_params_dlacont,
    nsig=2,
    show_truth=False,
    color="C0",
    label="DLA cont, with mask"
)
plot_corner(results_dict_parcon_nomask,
    free_params_dlacont,
    nsig=2,
    axes=ax,
    fig=fig,
    show_truth=False,
    color="C1",
    label="DLA cont, no mask",
)
plot_corner(results_dict_parcon_wmask_freebeta,
    free_params_dlacont,
    nsig=2,
    axes=ax,
    fig=fig,
    show_truth=False,
    color="C2",
    label=r"DLA cont, with mask, free $\beta_H$",
)

# %%
# plot bias and beta together from uncontaminated and partially contaminated fits
# let's focus on the no-DLA-masking and DLA-masking cases


# %%
