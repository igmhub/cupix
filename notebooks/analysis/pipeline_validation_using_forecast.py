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

# %% [markdown]
# ## Version 1: Read the outputs from the script validate_pipeline.py

# %%
import numpy as np
from cupix.likelihood.generate_fake_data import FakeData
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
import os
from forestflow.archive import GadgetArchive3D

import forestflow
from lace.cosmo.thermal_broadening import thermal_broadening_kms
import h5py as h5
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
# %load_ext autoreload
# %autoreload 2

# %%
def get_pars(data, parnamelist):
    # bias, beta, bias_err, beta_err = data['bias'], data['beta'], data['bias_err'], data['beta_err']
    print(data.keys())
    parlist = []
    parerrlist = []
    for par in parnamelist:
        parlist.append(data[par])
        parerrlist.append(data[par+'_err'])
    return(parlist, parerrlist)

    
def get_chi2_prob(data):
    chi2, prob = data['chi2'], data['prob']
    return(chi2,prob)
def plot_ellipses(pname_x, pname_y, val_x, val_y, sig_x, sig_y, cov, r, nsig=2, true_vals=None, true_val_label="true value"):
        """Plot Gaussian contours for parameters (pname_x,pname_y)
        - nsig: number of sigma contours to plot
        - cube_values: if True, will use unit cube values."""

        from matplotlib.patches import Ellipse
        from numpy import linalg as LA
        print(r)

        # shape of ellipse from eigenvalue decomposition of covariance
        w, v = LA.eig(
            np.array(
                [
                    [sig_x**2, sig_x * sig_y * r],
                    [sig_x * sig_y * r, sig_y**2],
                ]
            )
        )

        # semi-major and semi-minor axis of ellipse
        a = np.sqrt(w[0])
        b = np.sqrt(w[1])
        print(a,b)
        # figure out inclination angle of ellipse
        alpha = np.arccos(v[0, 0])
        if v[1, 0] < 0:
            alpha = -alpha
        # compute angle in degrees (expected by matplotlib)
        alpha_deg = alpha * 180 / np.pi

        # make plot
        fig = plt.subplot(111)
        for isig in range(1, nsig + 1):
            ell = Ellipse(
                (val_x, val_y), 2 * isig * a, 2 * isig * b, angle=alpha_deg
            )
            ell.set_alpha(0.6 / isig)
            fig.add_artist(ell)
        if true_vals is not None:
            plt.axvline(true_vals[pname_x], color='grey', linestyle='--', label=true_val_label)
            plt.axhline(true_vals[pname_y], color='grey', linestyle='--')
                        
            plt.legend()
        plt.xlabel(pname_x)
        plt.ylabel(pname_y)
        if true_vals is None:
            plt.xlim(val_x - (nsig + 1) * sig_x, val_x + (nsig + 1) * sig_x)
            plt.ylim(val_y - (nsig + 1) * sig_y, val_y + (nsig + 1) * sig_y)
        else:
            minx = min(val_x - (nsig + 1) * sig_x, true_vals[pname_x]-.1*abs(true_vals[pname_x]))
            maxx = max(val_x + (nsig + 1) * sig_x, true_vals[pname_x]+.1*abs(true_vals[pname_x]))
            miny = min(val_y - (nsig + 1) * sig_y, true_vals[pname_y]-.1*abs(true_vals[pname_y]))
            maxy = max(val_y + (nsig + 1) * sig_y, true_vals[pname_y]+.1*abs(true_vals[pname_y]))
            plt.ylim([miny,maxy])
            plt.xlim([minx,maxx])


# %%
fit_results = np.load("/pscratch/sd/m/mlokken/desi-lya/px/data/fitter_results/validation_forecast_results_central_noiseless_z225.npz")
pars_varied = ['mF', 'gamma']
[par1,par2],[par1_err,par2_err] = get_pars(fit_results, pars_varied)
chi2,prob = get_chi2_prob(fit_results)
cov = fit_results['cov']
r   = fit_results['r']

# %%
iz_choice = 0
# read the forecast data to get the truth
forecast_file = "/pscratch/sd/m/mlokken/desi-lya/px/data/px_measurements/forecast/forecast_ffcentral_cosmo_igm_binned_out_px-zbins_2-thetabins_18_noiseless.hdf5"
# forecast_file = "/pscratch/sd/m/mlokken/desi-lya/px/data/px_measurements/forecast/forecast_ffrandom_982_binned_out_px-zbins_2-thetabins_18_noiseless.hdf5"
f = h5.File(forecast_file)
# true_vals = {
#     'bias': f['like_params'].attrs['bias'][iz_choice],
#     'beta': f['like_params'].attrs['beta'][iz_choice]
# }\
true_vals = {}
for par in pars_varied:
    true_vals[par] = f['like_params'].attrs[par][iz_choice]

f.close()
forecast = DESI_DR2(forecast_file)

# %%

for itheta in range(len(forecast.theta_max_A_arcmin)):
    errors = np.diag(np.squeeze(forecast.cov_ZAM[iz_choice, itheta, :, :]))**0.5
    plt.errorbar(forecast.k_M_edges[iz_choice,1:], forecast.Px_ZAM[0, itheta, :], errors, label=f'theta={forecast.theta_max_A_arcmin[itheta]:.1f}')


# %%
true_vals

# %%
par1_err, par2_err

# %%
plot_ellipses('mF','gamma',par1,par2,par1_err,par2_err, cov, r) # 


# %%
plt.rc('font', size=16) 
plot_ellipses('mF','gamma',par1,par2,par1_err,par2_err, cov, r, true_vals=true_vals, true_val_label="Input truth") # 
plt.plot(par1,par2,'*', label='best fit')
# plt.ylabel(r"$\beta$")
# plt.xlim([-.15,-.10])
# plt.ylim([1.3,1.9])
plt.legend()

# %%
plt.rc('font', size=16) 
plot_ellipses('bias','beta',bias,beta,bias_err,beta_err, cov, r, true_vals=true_vals, true_val_label="Input truth") # 
plt.plot(bias,beta,'*', label='best fit')
# plt.ylabel(r"$\beta$")
# plt.xlim([-.15,-.10])
# plt.ylim([1.3,1.9])
plt.legend()

# %% [markdown]
# # Version 2: run a forecast in a notebook.
# ## Step 1: Import a noiseless forecast

# %%
# forecast = DESI_DR2("../../data/px_measurements/forecast/forecast_binned_out_px-zbins_4-thetabins_20_noiseless.hdf5", theta_min_cut_arcmin=0, kmax_cut_AA=1)
forecast_file = "/global/common/software/desi/users/mlokken/cupix/data/px_measurements/forecast//forecast_ffcentral_arinyo_binned_out_px-zbins_2-thetabins_18_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kmax_cut_AA=1)

# %%
f = h5.File(forecast_file)
for attr in f['like_params'].attrs.keys():
    print(attr, f['like_params'].attrs[attr])
f.close()

# %%
iz_choice = np.array([0])

# %%
# Load emulator
z = forecast.z
print(z)
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
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)

ffemu = FF_emulator(z, fid_cosmo, cc)
ffemu.kp_Mpc = 1 # set pivot point

theory_AA = set_theory(ffemu, k_unit='iAA', verbose=True)
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu
dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array(z), camb_results=cc)

# %%
# get the minimum and maximium parameter values from original simulations for emulator
# Figure out the ForestFlow training range
path_program = os.path.dirname(forestflow.__path__[0]) + '/'
path_program
folder_lya_data = path_program + "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    average="both",
)
print(len(Archive3D.training_data))
training_data = Archive3D.training_data

# this info is also saved in the file below:
import pandas as pd
info_sims = pd.read_csv("../../data/emulator/ff_training_info.csv")

# %%
import pandas as pd
info_sims = pd.read_csv("/pscratch/sd/m/mlokken/desi-lya/px/data/emulator/ff_training_info.csv")

# %%
info_sims['mF_min'], info_sims['mF_max'], info_sims['z']

# %%
param_names = ["bias","beta","q1","kvav","av","bv","kp","q2"]
par_training_vals = {par:[] for par in param_names}

for i in range(len(training_data)):
    if training_data[i]['z']==z[iz_choice]:
        for par in training_data[i]['Arinyo'].keys():
            if par in param_names:
                par_training_vals[par].append(training_data[i]['Arinyo'][par])

for par in param_names:
    par_training_vals[par] = np.array(par_training_vals[par])
    plt.hist(par_training_vals[par], bins=10)
    plt.title(par)
    plt.show()
    plt.clf()


# %% [markdown]
# ## Step 2: Run this through the minimizer pipeline for first z bin with arinyo values

# %%
# read the likelihood params from forecast file
like_params = []
with h5.File(forecast_file, "r") as f:
    params = f['like_params']
    attrs = params.attrs
    for key in param_names: # important to get the sorting right
        if key in attrs:
            val = attrs[key][:]
            if len(val)>0:
                min_training = np.amin(par_training_vals[key])
                max_training = np.amax(par_training_vals[key])
                like_params.append(LikelihoodParameter(
                    name=key,
                    value=val[iz_choice],
                    ini_value= np.random.uniform(min_training, max_training), # random initial value within training range
                    min_value=min_training,
                    max_value=max_training,
                    # Gauss_priors_width=(max_training - min_training) # broad priors
                ))
                print(key,max_training - min_training)
# check the parameters
for p in like_params:
    print(p.name, p.value)

# %%
like = Likelihood(forecast, theory_AA, free_param_names=["bias"], iz_choice=iz_choice, like_params=like_params, verbose=True)

# %%
like.plot_px(iz_choice, like_params, multiply_by_k=False, ylim2=[-.05,.05], ylim=[-.005,0.25], every_other_theta=False, show=True, title=f"Redshift 2.2", theorylabel='Theory: central sim', datalabel=f'Forecast at central sim', xlim=[0,.7], residual_to_theory=True)

# %%
mini = IminuitMinimizer(like, verbose=False)

# %%
# %%time
mini.minimize(compute_hesse=False)

# %%
mini.minimizer.nfcn

# %%
# %%time
mini.minimizer.hesse()

# %%
true_vals={par.name:par.value for par in like_params}
if true_vals['bias']>0:
    true_vals['bias'] *= -1

# %%
bias_index = [i for i, p in enumerate(like_params) if p.name=='bias'][0]
beta_index = [i for i, p in enumerate(like_params) if p.name=='beta'][0]
mini.plot_ellipses("bias", "beta", nsig=2, cube_values=False, true_vals=true_vals)
plt.axvline(x=like_params[bias_index].ini_value, color='orange', linestyle=':')
plt.axhline(y=like_params[beta_index].ini_value, color='orange', linestyle=':')
# add true value label
plt.ylim([1.35, 1.8])
plt.xlim([-0.145, -0.105])
plt.text(like_params[bias_index].ini_value, like_params[beta_index].ini_value, 'Initial Value', color='orange')
plt.savefig(f"../../plots/forecast_bias_beta_contours_inibias_{like_params[bias_index].ini_value:.3f}_inibeta_{like_params[beta_index].ini_value:.3f}.pdf")

# %%
# mini.minimizer.params
# like.fit_probability(values=[like_params[bias_index].value_in_cube(), like_params[beta_index].value_in_cube()])
# like_params[bias_index].value_from_cube(mini.minimizer.params['bias'].value), like_params[beta_index].value_from_cube(mini.minimizer.params['beta'].value)
# like_params[bias_index].ini_value

# %% [markdown]
# ## Repeat for IGM parameters

# %%
iz_choice = np.array([0])

# %%
param_names = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
par_training_vals = {par:[] for par in param_names}

for i in range(len(training_data)):
    if training_data[i]['z']==z[iz_choice]:
        for par in training_data[i].keys():
            if par in param_names:
                par_training_vals[par].append(training_data[i][par])

for par in param_names:
    par_training_vals[par] = np.array(par_training_vals[par])
    plt.hist(par_training_vals[par], bins=10)
    plt.title(par)
    plt.show()
    plt.clf()


# %%
forecast_file = "/global/common/software/desi/users/mlokken/cupix/data/px_measurements/forecast//forecast_ffcentral_cosmo_igm_binned_out_px-zbins_2-thetabins_18_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kmax_cut_AA=1)

# %%
# read the likelihood params from forecast file
like_params = []
with h5.File(forecast_file, "r") as f:
    params = f['like_params']
    attrs = params.attrs
    for key in param_names: # important to get the sorting right
        if key in attrs:
            val = attrs[key][:]
            if len(val)>0:
                min_training = np.amin(par_training_vals[key])
                max_training = np.amax(par_training_vals[key])
                like_params.append(LikelihoodParameter(
                    name=key,
                    value=val[iz_choice],
                    ini_value= np.random.uniform(min_training, max_training), # random initial value within training range
                    min_value=min_training,
                    max_value=max_training,
                    # Gauss_priors_width=(max_training - min_training) # broad priors
                ))
# check the parameters
for p in like_params:
    print(p.name, p.value, p.min_value, p.max_value)

# %%
like = Likelihood(forecast, theory_AA, free_param_names=["mF"], iz_choice=iz_choice, like_params=like_params, verbose=True)
like.plot_px(iz_choice, like_params, multiply_by_k=False, ylim2=[-.05,.05], ylim=[-.005,0.25], every_other_theta=False, show=True, title=f"Redshift 2.2", theorylabel='Theory: central sim', datalabel=f'Forecast at central sim', xlim=[0,.7], residual_to_theory=True)

# %%
# %%time
mini = IminuitMinimizer(like, verbose=True)
mini.minimize(compute_hesse=False)

# %%
mini.minimizer.nfcn

# %%
mini.minimizer.nfcn * 1.5

# %%
60*60/mini.minimizer.nfcn # average time per function evaluation in seconds

# %%
# %load_ext line_profiler

# %%
# %lprun -f mini.minimize mini.minimize(compute_hesse=True)

# %%
# %%time
mini.minimizer.hesse()

# %%
forestflow.__path__

# %%
mini.minimizer.nfcn

# %%
mF_bestfit,err = mini.best_fit_value("mF", return_hesse=True)
plt.errorbar("mF", mF_bestfit, yerr=err, fmt='o', label='best fit')
plt.plot("mF", like_params[2].value, color='k', marker='*',ms=20, label='Truth')
# plt.plot("bias", like_params[0].ini_value, color='orange', marker='^',ms=10, label='initial value')
plt.legend()

# %%
true_vals={par.name:par.value for par in like_params}
mF_index = [i for i, p in enumerate(like_params) if p.name=='bias'][0]
gamma_index = [i for i, p in enumerate(like_params) if p.name=='beta'][0]
mini.plot_ellipses("mF", "gamma", nsig=2, cube_values=False, true_vals=true_vals)
plt.axvline(x=like_params[mF_index].ini_value, color='orange', linestyle=':')
plt.axhline(y=like_params[gamma_index].ini_value, color='orange', linestyle=':')
# add true value label
# plt.ylim([1.35, 1.8])
# plt.xlim([-0.145, -0.105])
plt.text(like_params[mF_index].ini_value, like_params[gamma_index].ini_value, 'Initial Value', color='orange')
# plt.savefig(f"../../plots/forecast_mF_gamma_contours_inimF_{like_params[mF_index].ini_value:.3f}_inigamma_{like_params[gamma_index].ini_value:.3f}.pdf")

# %%
# Walther+ constriaints
T0_z2p2 = 1.014*1e4 # Kelvin
gamma_z2p2 = 1.74
mF_z2p2 = 0.825
lambdap_z2p2 = 79.4 # [kpc]

T0_z2p4 = 1.165*1e4
gamma_z2p4 = 1.63
mF_z2p4 = 0.799
lambdap_z2p4 = 81.1 # [kpc]

sigma_T_kms_z2p2 = thermal_broadening_kms(T0_z2p2)
sigT_Mpc_z2p2 = sigma_T_kms_z2p2 / dkms_dMpc_zs[0]
kF_Mpc_z2p2 = 1/(lambdap_z2p2/1000)

sigma_T_kms_z2p4 = thermal_broadening_kms(T0_z2p4)
sigT_Mpc_z2p4 = sigma_T_kms_z2p4 / dkms_dMpc_zs[0]
kF_Mpc_z2p4 = 1/(lambdap_z2p4/1000)

# %%
# compute linear power parameters at each z (in Mpc units)
linP_zs = fit_linP.get_linP_Mpc_zs(
    sim_cosmo, z, 0.7
)

# %% [markdown]
# ## Step 2: Run this through the minimizer pipeline for first z bin with arinyo values

# %%
# set the likelihood parameters as the Arinyo params with the fiducial values being the true initial inputs:

# {'bias': array([0.10722388, 0.12911841]), 'beta': array([1.62629604, 1.51886928]), 'q1': array([0.24203433, 0.2228016 ]), 'kvav': array([0.51980138, 0.53986251]), 'av': array([0.39633834, 0.45435816]), 'bv': array([1.70459318, 1.74364209]), 'kp': array([14.37836838, 14.50495338]), 'q2': array([0.29897869, 0.3543148 ])}
#z = 2.2
like_params = []
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=1.0,
    ini_value=-0.05,
    value =0.10722388,
    Gauss_priors_width=.05
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.0,
    max_value=2.0,    
    ini_value = .5,
    value=1.62629604,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='q1',
    min_value=0.0,
    max_value=2.0,
    ini_value = 1.0,
    value = 0.24203433, # 0.1112,
    Gauss_priors_width=0.111
    ))
like_params.append(LikelihoodParameter(
    name='kvav',
    min_value=0.0,
    max_value=1.0,
    ini_value = .5,
    value=0.51980138,
    Gauss_priors_width=0.0003**0.2694,
    ))
like_params.append(LikelihoodParameter(
    name='av',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.3,
    value=0.39633834, #0.2694,
    Gauss_priors_width=0.27
    ))
like_params.append(LikelihoodParameter(
    name='bv',
    min_value=0.0,
    max_value=3.0,
    ini_value = 0.2,
    value=1.70459318,
    Gauss_priors_width=0.0002 
    ))
like_params.append(LikelihoodParameter(
    name='kp',
    min_value=0.0,
    max_value=20.0,
    ini_value = 0.5,
    value=14.37836838,
    Gauss_priors_width=0.5
    ))
like_params.append(LikelihoodParameter(
    name='q2',
    min_value = 0,
    max_value = 1.0,
    ini_value = 0.5,
    value = 0.29897869,
    Gauss_priors_width=0.5
    ))


# %%
like = Likelihood(forecast, theory_AA, free_param_names=["bias", "beta"], iz_choice=[0], like_params=like_params, verbose=True)

# %%
like.plot_px([0], like_params, multiply_by_k=False, ylim2=[-.05,.05], ylim=[-.005,0.25], every_other_theta=False, show=True, title=f"Redshift 2.2", theorylabel='Theory: P18 (cosmo)\n + Chaves-Montero+25 (IGM)', datalabel=f'DR2 measurement', xlim=[0,.7], residual_to_theory=True)

# %%
mini = IminuitMinimizer(like, verbose=False)

# %%
# %%time
mini.minimize()

# %%
# timing of the above cell:
# CPU times: user 57.4 s, sys: 8.51 s, total: 1min 5s
# Wall time: 56.5 s

# %%
true_vals={par.name:par.value for par in like_params}
true_vals['bias'] *= -1
mini.plot_ellipses("bias", "beta", nsig=2, cube_values=False, true_vals=true_vals)

# %%
bias_bestfit,err = mini.best_fit_value("bias", return_hesse=True)
plt.errorbar("bias", bias_bestfit, yerr=err, fmt='o', label='best fit')
plt.plot("bias", like_params[0].value*-1, color='k', marker='*',ms=20, label='Truth')
# plt.plot("bias", like_params[0].ini_value, color='orange', marker='^',ms=10, label='initial value')
plt.legend()

# %% [markdown]
# ## Step 3: Run through the minimizer pipeline for the first z bin with IGM values

# %%
sigT_Mpc_z2p2

# %%
mF_z2p2+.5*mF_z2p2

# %%
# 2.2
like_params = []
like_params.append(LikelihoodParameter(
    name='sigT_Mpc',
    min_value=-1.0,
    max_value=1.0,
    value=sigT_Mpc_z2p2,
    ini_value=sigT_Mpc_z2p2+.5*sigT_Mpc_z2p2,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='gamma',
    min_value=0.0,
    max_value=3.0,
    value = gamma_z2p2,
    ini_value = gamma_z2p2-.5*gamma_z2p2,
    Gauss_priors_width=1
    ))
like_params.append(LikelihoodParameter(
    name='mF',
    min_value=0.0,
    max_value=2.0,
    value = mF_z2p2,
    ini_value = mF_z2p2+.5*mF_z2p2,
    Gauss_priors_width=0.5
    ))
like_params.append(LikelihoodParameter(
    name='Delta2_p',
    min_value=0.0,
    max_value=1.0,
    ini_value = linP_zs[0]['Delta2_p'],
    value = linP_zs[0]['Delta2_p']
    ))
like_params.append(LikelihoodParameter(
    name='n_p',
    min_value=-5,
    max_value=1.0,
    ini_value = linP_zs[0]['n_p'],
    value = linP_zs[0]['n_p']
    ))
like_params.append(LikelihoodParameter(
    name='kF_Mpc',
    min_value=0.0,
    max_value=20,
    ini_value = kF_Mpc_z2p2,
    value = kF_Mpc_z2p2
    ))
like_params.append(LikelihoodParameter(
    name='lambda_P',
    min_value=0.0,
    max_value=150.,
    ini_value = lambdap_z2p2,
    value = lambdap_z2p2
    ))

# %%
like = Likelihood(forecast, theory_AA, free_param_names=["mF", "gamma"], iz_choice=[0], like_params=like_params, verbose=False)

# %%
mini = IminuitMinimizer(like, verbose=False)

# %% jupyter={"outputs_hidden": true}
# %%time
mini.minimize()

# %%
true_vals={par.name:par.value for par in like_params}
mini.plot_ellipses("mF", "gamma", nsig=2, cube_values=False, true_vals=true_vals)

# %%
# timing of the above cell:
# CPU times: user 5min 43s, sys: 1min, total: 6min 43s
# Wall time: 5min 40s

# another run:
# 7 min 35 s

# %%
mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, xlim=[-.01, .4], ylim2 = [-3,3], datalabel="Mock ``Forecast'' Data", show=True)

# %%
mF_bestfit,err = mini.best_fit_value("mF", return_hesse=True)
plt.errorbar("mF", mF_bestfit, yerr=err, fmt='o', label='best fit')
plt.plot("mF", like_params[2].value, color='k', marker='*',ms=20, label='Truth')
# plt.plot("mF", like_params[2].ini_value, color='orange', marker='^',ms=10, label='initial value')
plt.legend()

# %%
mF_z2p2, mF_bestfit

# %%
err/mF_bestfit*100

# %%
mini.plot_ellipses("bias", "bias", nsig=2, cube_values=False, true_vals={'bias': -0.115, 'beta': 1.55, 'q1': 0.1112, 'av':0.2694})

# %%
prob = like.fit_probability(mini.minimizer.values)
print(prob)

# %% [markdown]
# ## Step 4: Import a noisy forecast

# %% [markdown]
# ## Step 5: Run this through the minimizer for 1 z bin many times 

# %% [markdown]
# ## Step 6: ensure the noisy forecast scatter makes sense with the pipeline errors

# %% [markdown]
# ### Step 7: repeat for the MCMC pipeline

# %%
