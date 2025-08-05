# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: pcross
#     language: python
#     name: python3
# ---

# %%
import sys
import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
# %load_ext autoreload
# %autoreload 2
from lace.cosmo import camb_cosmo, fit_linP
import matplotlib.pyplot as plt


# %% [markdown]
# Load the emulator

# %%
# Load emulator
z = np.array([2.2,2.4])
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

# %% [markdown]
# Set the theory with some default parameters

# %%
emu_params = Args()
emu_params.set_baseline()

theory_AA = set_theory(emu_params, ffemu, free_parameters=['ln_tau_0'], k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu

# %% [markdown]
# ## Let's start with an example where we allow the likelihood parameters to be determined by fiducial values of underlying models.

# %% [markdown]
# All we have to do in the followwing is to set the names and min/max values of the parameters we want to input.

# %%

# make some likelihood parameter objects
likelihood_params = []


likelihood_params.append(LikelihoodParameter(
    name='Delta2_p',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='n_p',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='mF',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='gamma',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='kF_Mpc',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='sigT_Mpc',
    min_value=-1.0,
    max_value=1.0,
    ))


# %%
[par.name for par in likelihood_params]

# %% [markdown]
# Current working version includes option to have different k for different redshift bins, but not for different theta (might want to make this fully generic)
#

# %%
theta_bin_deg = np.asarray([[[0,0.02],[0.02,0.1],[.1,.5],[.5,1]], [[0,0.02],[0.02,0.1],[.1,.5],[.5,1]]])
k_AA = np.array([np.linspace(0.01,.8,100), np.linspace(0.01,.8,100)])
k_AA.shape, z.shape, theta_bin_deg.shape

# %%
out_AA = theory_AA.get_px_AA(
        zs = z,
        k_AA=k_AA,
        theta_bin_deg=theta_bin_deg,
        like_params=likelihood_params,
        return_blob=False
    )

# %%
for iz, zbin in enumerate(out_AA):
    print("z=", z[iz])
    if iz==0:
        linestyle='dashed'
    else:
        linestyle='solid'
    
    for itheta, theta in enumerate(zbin):
        # print(out[iz][itheta])
        plt.plot(k_AA[iz], out_AA[iz][itheta], label='theta={}, z={}'.format(theta_bin_deg[iz][itheta], z[iz]), linestyle=linestyle)
    
plt.legend()


# %% [markdown]
# Let's do the same thing, but for a case of only 1 z, to make sure it works 

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
cosmo = {
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
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)
ffemu = FF_emulator(z, cosmo, cc)
ffemu.kp_Mpc = 1 # pivot point

# %%

emu_params = Args()
emu_params.set_baseline()

theory_AA = set_theory(emu_params, ffemu, free_parameters=['ln_tau_0'], k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu

# %%
theta_bin_deg = np.asarray([[[0,0.02],[0.02,0.1],[.1,.5],[.5,1]]])
k_AA = np.array([np.linspace(0.01,.8,100)])
k_AA.shape, z.shape, theta_bin_deg.shape

# %%

likelihood_params.append(LikelihoodParameter(
    name='Delta2_p',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='n_p',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='mF',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='gamma',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='kF_Mpc',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='sigT_Mpc',
    min_value=-1.0,
    max_value=1.0,
    ))


# %%
out_AA = theory_AA.get_px_AA(
        zs = z,
        k_AA=k_AA,
        theta_bin_deg=theta_bin_deg,
        like_params=likelihood_params,
        return_blob=False
    )

# %%
for iz, zbin in enumerate(out_AA):
    print("z=", z[iz])
    if iz==0:
        linestyle='dashed'
    else:
        linestyle='solid'
    
    for itheta, theta in enumerate(zbin):
        # print(out[iz][itheta])
        plt.plot(k_AA[iz], out_AA[iz][itheta], label=r'$\theta$={}, z={}'.format(theta_bin_deg[iz][itheta], z[iz]), linestyle=linestyle)
    
plt.legend()
plt.ylim([0,0.18])

# %% [markdown]
# ## Now let's try inputting directly the Arinyo parameters

# %%
# Load emulator
z = np.array([2.2,2.4])
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

# %% [markdown]
# Set the theory with some default parameters

# %%
emu_params = Args()
emu_params.set_baseline()

theory_AA = set_theory(emu_params, ffemu, free_parameters=['ln_tau_0'], k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values

likelihood_params = []
likelihood_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=1.0,
    value=[-0.115,-0.115]
    ))
likelihood_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.0,
    max_value=2.0,    
    value = [1.55, 1.55]
    ))
likelihood_params.append(LikelihoodParameter(
    name='q1',
    min_value=0.0,
    max_value=1.0,
    value = [0.1112, 0.1112]
    ))
likelihood_params.append(LikelihoodParameter(
    name='kvav',
    min_value=0.0,
    max_value=1.0,
    value = [0.0001**0.2694, 0.0001**0.2694]
    ))
likelihood_params.append(LikelihoodParameter(
    name='av',
    min_value=0.0,
    max_value=1.0,
    value = [0.2694, 0.2694]
    ))
likelihood_params.append(LikelihoodParameter(
    name='bv',
    min_value=0.0,
    max_value=1.0,
    value = [0.0002, 0.0002]
    ))
likelihood_params.append(LikelihoodParameter(
    name='kp',
    min_value=0.0,
    max_value=1.0,
    value = [0.5740, 0.5740]
    ))
likelihood_params.append(LikelihoodParameter(
    name='q2',
    min_value=0.0,
    max_value=1.0,
    value = [0.5740, 0.5740]
    ))


# initial_mF = 0.825
# initial_T0 = 1.014 * 1e4
# initial_gamma = 1.74
# initial_lambda_pressure = 79.4
# linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, z, ffemu.kp_Mpc) # here z could be an array of zs if desired (e.g., [2.2, 2.4, 2.6])


# %%
[par.name for par in likelihood_params], [par.value for par in likelihood_params]

# %% [markdown]
# Current working version includes option to have different k for different redshift bins, but not for different theta (might want to make this fully generic)
#

# %%
theta_bin_deg = np.asarray([[[0,0.02],[0.02,0.1],[.1,.5],[.5,1]], [[0,0.02],[0.02,0.1],[.1,.5],[.5,1]]])
k_AA = np.array([np.linspace(0.01,.8,100), np.linspace(0.01,.8,100)])
k_AA.shape, z.shape, theta_bin_deg.shape

# %%
out_AA = theory_AA.get_px_AA(
        zs = z,
        k_AA=k_AA,
        theta_bin_deg=theta_bin_deg,
        like_params=likelihood_params,
        return_blob=False
    )

# %%
for iz, zbin in enumerate(out_AA):
    print("z=", z[iz])
    if iz==0:
        linestyle='dashed'
    else:
        linestyle='solid'
    
    for itheta, theta in enumerate(zbin):
        # print(out[iz][itheta])
        plt.plot(k_AA[iz], out_AA[iz][itheta], label='theta={}, z={}'.format(theta_bin_deg[iz][itheta], z[iz]), linestyle=linestyle)
    
plt.legend()


# %% [markdown]
# ## Now try inputting Silicon model

# %%
likelihood_params.append(LikelihoodParameter(
    name='bias_SiIII',
    min_value=0.0,
    max_value=1.0,
    value = [-9.79e-3, -9.79e-3]
    ))

# %%
out_AA = theory_AA.get_px_AA(
        zs = z,
        k_AA=k_AA,
        theta_bin_deg=theta_bin_deg,
        like_params=likelihood_params,
        return_blob=False,
        add_silicon=True
    )

# %%
for iz, zbin in enumerate(out_AA):
    print("z=", z[iz])
    if iz==0:
        linestyle='dashed'
    else:
        linestyle='solid'
    
    for itheta, theta in enumerate(zbin):
        # print(out[iz][itheta])
        plt.plot(k_AA[iz], out_AA[iz][itheta], label='theta={}, z={}'.format(theta_bin_deg[iz][itheta], z[iz]), linestyle=linestyle)
    
plt.legend()


# %%
