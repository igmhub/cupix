# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from lace.cosmo import camb_cosmo
import matplotlib.pyplot as plt

from cupix.px_data.data_lyacolore import Px_Lyacolore
# %load_ext autoreload
# %autoreload 2
# %load_ext line_profiler

# %%

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

# %%
ffemu = FF_emulator(z, fid_cosmo, cc)
ffemu.kp_Mpc = 1 # set pivot point

# %%
theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu

# %%
# MockData = Px_Lyacolore("binned_out_trucont_px-zbins_2-thetabins_10_res.hdf5", theta_min_cut_arcmin=40)
MockData = Px_Lyacolore("binned_out_trucont_px-zbins_2-thetabins_10_res.hdf5", theta_min_cut_arcmin=10, kmax_cut_AA=1)

# %%
MockData.theta_centers_arcmin

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values

like_params = []
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=1.0,
    ini_value=-0.115,
    value =-0.115,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.0,
    max_value=2.0,    
    ini_value = 1.55,
    value=1.55,
    Gauss_priors_width=1
    ))
like_params.append(LikelihoodParameter(
    name='q1',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.1112,
    value=0.1112,
    Gauss_priors_width=0.05
    ))
like_params.append(LikelihoodParameter(
    name='kvav',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.0001**0.2694,
    value=0.0001**0.2694,
    Gauss_priors_width=0.0002**0.2694,
    ))
like_params.append(LikelihoodParameter(
    name='av',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.2694,
    value=0.2694,
    Gauss_priors_width=0.2
    ))
like_params.append(LikelihoodParameter(
    name='bv',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.0002,
    value=0.0002,
    Gauss_priors_width=0.0001
    ))
like_params.append(LikelihoodParameter(
    name='kp',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.5740,
    value=0.5740,
    Gauss_priors_width=0.2
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
like = Likelihood(MockData, theory_AA, free_param_names=["beta"], iz_choice=0, like_params=like_params, verbose=True)

# %%
like.get_log_like([lp.value for lp in like_params])

# %%
# %lprun -f like.get_log_like like.get_log_like([lp.value for lp in like_params])

# %%
# %lprun -f like.get_convolved_Px_AA like.get_convolved_Px_AA(0, np.arange(len(MockData.theta_max_A_arcmin)), like_params) # compare to .897s total, 4 calls per get_Px

# %%
# %lprun -f like.theory.get_px_AA like.theory.get_px_AA([2.2],MockData.k_m, 10, like_params)


# %%
like.plot_px(0, like_params, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None)

# %%
