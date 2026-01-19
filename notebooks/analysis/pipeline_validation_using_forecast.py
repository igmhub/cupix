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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Step 1: Import a noiseless forecast

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
from lace.cosmo.thermal_broadening import thermal_broadening_kms

from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
# %load_ext autoreload
# %autoreload 2

# %%
forecast = DESI_DR2("../../data/px_measurements/forecast/forecast_binned_out_px-zbins_4-thetabins_20_noiseless.hdf5", theta_min_cut_arcmin=0, kmax_cut_AA=1)

# %%
# Load emulator
z = forecast.z[0:1]
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

theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu
dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array(z))

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
