# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

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
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
# %load_ext autoreload
# %autoreload 2

# %%
forecast = DESI_DR2("/Users/mlokken/research/lyman_alpha/software/cupix/data/px_measurements/forecast/forecast_binned_out_trucont_px-zbins_2-thetabins_18.hdf5", theta_min_cut_arcmin=0, kmax_cut_AA=1)

# %%
# Load emulator
z = [forecast.z[0]] # choose redshift bin 2.2
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

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values

like_params = []
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=0,
    ini_value=-0.05,
    value =-0.2,
    Gauss_priors_width=.05
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.0,
    max_value=2.0,    
    ini_value = .5,
    value=1.55,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='q1',
    min_value=0.0,
    max_value=2.0,
    ini_value = 1.0,
    value=0.1112,
    Gauss_priors_width=0.111
    ))
like_params.append(LikelihoodParameter(
    name='kvav',
    min_value=0.0,
    max_value=1.0,
    ini_value = .5,
    value=0.0001**0.2694,
    Gauss_priors_width=0.0003**0.2694,
    ))
like_params.append(LikelihoodParameter(
    name='av',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.3,
    value=0.2694,
    Gauss_priors_width=0.27
    ))
like_params.append(LikelihoodParameter(
    name='bv',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.2,
    value=0.0002,
    Gauss_priors_width=0.0002
    ))
like_params.append(LikelihoodParameter(
    name='kp',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.5,
    value=0.5740,
    Gauss_priors_width=0.5
    ))


# likelihood_params = []
# likelihood_params.append(LikelihoodParameter(
#     name='Delta2_p',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='n_p',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='mF',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='gamma',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='kF_Mpc',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='sigT_Mpc',
#     min_value=-1.0,
#     max_value=1.0,
#     ))

# %%
like = Likelihood(forecast, theory_AA, free_param_names=["bias", "beta", "q1", "av"], iz_choice=0, like_params=like_params, verbose=False)

# %%
mini = IminuitMinimizer(like, verbose=False)

# %%
mini.minimizer.params

# %%
mini.minimize()

# %%
prob = like.fit_probability(mini.minimizer.values)


# %%
prob

# %%
mini.plot_best_fit(multiply_by_k=False, every_other_theta=True, xlim=[-.01, .4], datalabel="Mock Data", show=True)

# %%
mini.plot_ellipses("bias", "beta", nsig=2, cube_values=False, true_vals={'bias': -0.115, 'beta': 1.55, 'q1': 0.1112, 'av':0.2694})

# %%
mini.plot_ellipses("av", "q1", nsig=2, cube_values=False, true_vals={'bias': -0.115, 'beta': 1.55, 'q1': 0.1112, 'av':0.2694})

# %%
