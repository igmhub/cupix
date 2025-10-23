# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: python3
# ---

# Minuit minimizer
#

import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from lace.cosmo import camb_cosmo, fit_linP
import matplotlib.pyplot as plt
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta
from cupix.px_data.data_lyacolore import Px_Lyacolore
import scipy
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
# %load_ext autoreload
# %autoreload 2

# +

# Load emulator
z = np.array([2.2])
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
# -

ffemu = FF_emulator(z, fid_cosmo, cc)
ffemu.kp_Mpc = 1 # set pivot point

emu_params = Args()
emu_params.set_baseline()
print(emu_params)

theory_AA = set_theory(emu_params, ffemu, free_parameters=['ln_tau_0'], k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu

MockData = Px_Lyacolore("binned_out_truecont_px-zbins_2-thetabins_9.hdf5", theta_min_cut_arcmin=10)


# +
# set the likelihood parameters as the Arinyo params with some fiducial values

like_params = []
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=1.0,
    value=-0.115,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.0,
    max_value=2.0,    
    value = 1.55,
    Gauss_priors_width=1
    ))
like_params.append(LikelihoodParameter(
    name='q1',
    min_value=0.0,
    max_value=1.0,
    value = 0.1112
    ))
like_params.append(LikelihoodParameter(
    name='kvav',
    min_value=0.0,
    max_value=1.0,
    value = 0.0001**0.2694
    ))
like_params.append(LikelihoodParameter(
    name='av',
    min_value=0.0,
    max_value=1.0,
    value = 0.2694
    ))
like_params.append(LikelihoodParameter(
    name='bv',
    min_value=0.0,
    max_value=1.0,
    value = 0.0002
    ))
like_params.append(LikelihoodParameter(
    name='kp',
    min_value=0.0,
    max_value=1.0,
    value = 0.5740
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
# -

like = Likelihood(MockData, theory_AA, free_param_names=["bias","beta"], iz_choice=0, like_params=like_params)

mini = IminuitMinimizer(like, like_params, verbose=True)

mini.minimize()

mini.best_fit_value("bias")

mini.minimizer.params

mini.minimizer.params.

import copy
like_params_to_plot = copy.deepcopy(like_params)
like_params_to_plot[0].value = -0.1204
like_params_to_plot[1].value = 1.46

for p in like_params_to_plot:
    print(f"{p.name}: {p.value}")   

like.plot_px(0, like_params, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None)


like.plot_px(0, like_params_to_plot, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None)



