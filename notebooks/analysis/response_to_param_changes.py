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
#     display_name: forestflow
#     language: python
#     name: forestflow
# ---

# %%
# %load_ext autoreload
# %autoreload 1

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
import copy
from lace.cosmo.thermal_broadening import thermal_broadening_kms
import time

# %%
# plot styles
plt.style.use('seaborn-v0_8-colorblind')
plt.style.use('../../plots/pxpaper.mplstyle')
from matplotlib import lines

# %%
# Load emulator
z = [2.2]
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
ffemu.emu.Nrealizations

# %%
# set some default likelihood parameters
like_params = []
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=0,
    ini_value=-0.05,
    value =-0.115,
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

# %%
# set some default likelihood parameters
like_params_bx2 = copy.deepcopy(like_params)
for param in like_params_bx2:
    if param.name == 'bias':
        param.value *= 2
like_params_betax2 = copy.deepcopy(like_params)
for param in like_params_betax2:
    if param.name == 'beta':
        param.value *= 2

like_params_q1x2 = copy.deepcopy(like_params)
for param in like_params_q1x2:
    if param.name == 'q1':
        param.value *= 2

# %%
iz = 0 # choice of z bin
k = np.linspace(0.001,1,100)
thetabins = np.logspace(-.5,2,8)
res = theory_AA.get_px_AA(
            zs = [z[iz]],
            k_AA=k,
            theta_arcmin=thetabins,
            like_params=like_params
        )

res2 = theory_AA.get_px_AA(
            zs = [z[iz]],
            k_AA=k,
            theta_arcmin=thetabins,
            like_params=like_params_bx2
        )
res2xbeta = theory_AA.get_px_AA(
            zs = [z[iz]],
            k_AA=k,
            theta_arcmin=thetabins,
            like_params=like_params_betax2
        )

res2xq1 = theory_AA.get_px_AA(
            zs = [z[iz]],
            k_AA=k,
            theta_arcmin=thetabins,
            like_params=like_params_q1x2
        )

# %%
colors = plt.cm.tab10(np.linspace(0,1,len(thetabins)))


# %%

for theta in range(len(res)):
    plt.plot(k, res[theta], label=rf"$\theta$={thetabins[theta]:.1f}$\prime$", color=colors[theta])
for theta in range(len(res2)):
    plt.plot(k, res2[theta], linestyle='dashed', color=colors[theta])
plt.xlim([0,1])
plt.legend()
plt.ylabel(r"$P_\times$ [$\AA$]")
plt.xlabel(r"$k$ [$\AA^{-1}$]")

# %%

for theta in range(len(res)):
    plt.plot(k, res[theta], label=rf"$\theta$={thetabins[theta]:.1f}$\prime$", color=colors[theta])
for theta in range(len(res2)):
    plt.plot(k, res2xbeta[theta], linestyle='dashed', color=colors[theta])

plt.legend()
plt.ylabel(r"$P_\times$ [$\AA$]")
plt.xlabel(r"$k$ [$\AA^{-1}$]")
plt.xlim([0,1])

# %%

for theta in range(len(res)):
    plt.plot(k, res[theta], label=rf"$\theta$={thetabins[theta]:.1f}$\prime$", color=colors[theta])
for theta in range(len(res2xq1)):
    plt.plot(k, res2xq1[theta], linestyle='dashed', color=colors[theta])

plt.legend()
plt.ylabel(r"$P_\times$ [$\AA$]")
plt.xlabel(r"$k$ [$\AA^{-1}$]")
plt.xlim([0,1])

# %% [markdown]
# Do it for the input parameters

# %%
compare_to = "W18"
# compare_to = "CM25"
# compare_to = "AK25"
if compare_to == "W18":
    if z[0] == 2.2:
        T0 = 1.014*1e4 # Kelvin
        gamma = 1.74
        mF = 0.825
        lambdap = 79.4 # [kpc]
    # Walther+ constriaints
    elif z[0] == 2.4:
        T0 = 1.165*1e4
        gamma = 1.63
        mF = 0.799
        lambdap = 81.1 # [kpc]
    else:
        print("Need to input the other redshift values.")

sigma_T_kms = thermal_broadening_kms(T0)
sigT_Mpc = sigma_T_kms / dkms_dMpc_zs[0]
kF_Mpc = 1/(lambdap/1000)

# %%
# compute linear power parameters at each z (in Mpc units)
linP_zs = fit_linP.get_linP_Mpc_zs(
    sim_cosmo, z, 0.7
)

# %%
like_params = []
like_params.append(LikelihoodParameter(
    name='sigT_Mpc',
    min_value=-1.0,
    max_value=1.0,
    value=sigT_Mpc,
    ini_value=sigT_Mpc,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='gamma',
    min_value=0.0,
    max_value=3.0,    
    value = gamma,
    ini_value = gamma,
    Gauss_priors_width=1
    ))
like_params.append(LikelihoodParameter(
    name='mF',
    min_value=0.0,
    max_value=1.0,
    ini_value = mF,
    value = mF,
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
    min_value=0.0,
    max_value=1.0,
    ini_value = linP_zs[0]['n_p'],
    value = linP_zs[0]['n_p']
    ))
like_params.append(LikelihoodParameter(
    name='kF_Mpc',
    min_value=0.0,
    max_value=1.0,
    ini_value = kF_Mpc,
    value = kF_Mpc
    ))
like_params.append(LikelihoodParameter(
    name='lambda_P',
    min_value=0.0,
    max_value=1.0,
    ini_value = lambdap,
    value = lambdap
    ))

# %%
np.log10(2)

# %%
# set some default likelihood parameters
like_params_mFx2 = copy.deepcopy(like_params)
for param in like_params_mFx2:
    if param.name == 'mF':
        param.value *= 2
like_params_gammax2 = copy.deepcopy(like_params)
for param in like_params_gammax2:
    if param.name == 'gamma':
        param.value *= 2

        
iz = 0 # choice of z bin
k = np.logspace(-3,0.3,100)
# thetabins = np.logspace(-.5,2,4)
thetabins = [0.5, 2, 10, 30, 80]

start1 = time.time()
res = theory_AA.get_px_AA(
            zs = [z[iz]],
            k_AA=k,
            theta_arcmin=thetabins,
            like_params=like_params
        )
end1 = time.time()
start2 = time.time()
res2xmF = theory_AA.get_px_AA(
            zs = [z[iz]],
            k_AA=k,
            theta_arcmin=thetabins,
            like_params=like_params_mFx2
        )
end2 = time.time()
start3 = time.time()
res2xgamma = theory_AA.get_px_AA(
            zs = [z[iz]],
            k_AA=k,
            theta_arcmin=thetabins,
            like_params=like_params_gammax2
        )
end3 = time.time()
print(end3-start3, end2-start2, end1-start1)

# %%

for theta in range(len(res)):
    plt.plot(k, res[theta], label=rf"$\theta$={thetabins[theta]:.1f}$\prime$", color=colors[theta])
for theta in range(len(res2xmF)):
    plt.plot(k, res2xmF[theta], linestyle='dashed', color=colors[theta])
plt.yscale('symlog', linthresh=1e-5)
plt.xscale('log')

plt.legend()
plt.ylabel(r"$P_\times\, [\AA]$")
plt.xlabel(r"$k\, [\AA^{-1}]$")
plt.xlim([0.001,2])
plt.ylim([-3e-5,.5])
handles, labels = plt.gca().get_legend_handles_labels()

dashed = lines.Line2D([0],[0],color='grey', linestyle='dashed', label=rf'$2 \times \bar{{F}}$')
plt.legend(handles=handles + [dashed])
plt.savefig("../../plots/forecast_mfx2.pdf", bbox_inches='tight')

# %%
plt.rc('font', size=16) 
for theta in range(len(res)):
    plt.plot(k, res[theta], label=rf"$\theta$={thetabins[theta]:.1f}$\prime$", color=colors[theta])
for theta in range(len(res2xgamma)):
    plt.plot(k, res2xgamma[theta], linestyle='dashed', color=colors[theta])
plt.yscale('symlog', linthresh=1e-5)
plt.xscale('log')

plt.legend()
plt.ylabel(r"$P_\times\, [\AA]$")
plt.xlabel(r"$k\, [\AA^{-1}]$")
plt.xlim([0.001,2])
plt.ylim([-3e-5,.5])
handles, labels = plt.gca().get_legend_handles_labels()

dashed = lines.Line2D([0],[0],color='grey', linestyle='dashed', label=rf'$2 \times \gamma$')
plt.legend(handles=handles + [dashed])
plt.savefig("../../plots/forecast_gammax2.pdf", bbox_inches='tight')

# %%
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# %%
mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc/u.arcmin)
mpc_per_arcmin * 0.5 * u.arcmin, mpc_per_arcmin * 80 * u.arcmin
