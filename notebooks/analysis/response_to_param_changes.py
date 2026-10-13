# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 1

# %%
from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.inference.free_parameter import FreeParameter
from cupix.inference.posterior import Posterior
from cupix.inference.minimize_posterior import Minimizer
import matplotlib.pyplot as plt
import numpy as np

# %%
# plot styles
plt.style.use('seaborn-v0_8-colorblind')
plt.style.use('../../plots/pxpaper.mplstyle')
from matplotlib import lines

# %%
basedir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/"
fname = basedir + "/wp1d/BALs_weighted_to_zero/bf3_binned_out_px-zbins_4-thetabins_20_w_res_wp1d.hdf5"
data = DESI_DR2(config={'data_file':fname, 'kM_min_cut_AA':0.5, 'kM_max_cut_AA':1.0, 'km_max_cut_AA':1.2, 'theta_min_cut_arcmin':10.0})
zs = data.z

# %% [markdown]
# ## Begin with uncontaminated theory

# %%
cosmo = cosmology.Cosmology()
config_lya={'verbose': True, 'include_hcd': False, 'include_metal': False,
        'include_sky': False, 'include_continuum': False, 'default_lya_model':'best_fit_arinyo_from_p1d'}
config_igm={'verbose': True, 'include_hcd': False, 'include_metal': False,
        'include_sky': False, 'include_continuum': False, 'default_lya_model':'best_fit_igm_from_p1d'}
theories_lya = []
theories_igm = []
for iz, z in enumerate(data.z): 
    theory = Theory(z=z, fid_cosmo=cosmo, config=config_lya)
    theories_lya.append(theory)
    theory = Theory(z=z, fid_cosmo=cosmo, config=config_igm)
    theories_igm.append(theory)

# %%
iz = 0 # choice of z bin
k = np.linspace(0.001,1,100)
thetabins = [(6.31+10)/2., (15+20)/2., (30+35)/2.]

bias_fid = theories_lya[iz].get_param('bias')
bias_new = bias_fid*1.1
px_changed_bias = theories_lya[iz].get_px_obs(
            k_AA=k,
            theta_arc=thetabins,
            params={'bias':bias_new}
        )

px_fid_bias = theories_lya[iz].get_px_obs(
            k_AA=k,
            theta_arc=thetabins
        )
mF_fid = theories_igm[iz].get_param('mF')
mF_new = mF_fid*1.05
px_changed_mF = theories_igm[iz].get_px_obs(
            k_AA=k,
            theta_arc=thetabins,
            params={'mF':mF_new}
        )

px_fid_mF = theories_igm[iz].get_px_obs(
            k_AA=k,
            theta_arc=thetabins
        )


# %%
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6,3))

colors = plt.cm.tab10(np.linspace(0,1,len(thetabins)))
for theta in range(0,len(px_fid_bias)):
    ax[0].plot(k, px_fid_bias[theta], label=rf"$\theta$={thetabins[theta]:.1f}$^\prime$", color=colors[theta])
    ax[1].plot(k, px_fid_mF[theta], color=colors[theta])
for theta in range(0,len(px_changed_bias)):
    ax[0].plot(k, px_changed_bias[theta], linestyle='dashed', color=colors[theta])
    ax[1].plot(k, px_changed_mF[theta], linestyle='dashed', color=colors[theta])
dashed = lines.Line2D([0],[0],color='grey', linestyle='dashed', label=rf'$b_\alpha={bias_new:.2f}$')
solid = lines.Line2D([0],[0],color='grey', linestyle='solid', label=rf'$b_\alpha={bias_fid:.2f}$')
dashed_mF = lines.Line2D([0],[0],color='grey', linestyle='dashed', label=rf'$\overline{{F}}={mF_new:.2f}$')
solid_mF = lines.Line2D([0],[0],color='grey', linestyle='solid', label=rf'$\overline{{F}}={mF_fid:.2f}$')
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles=handles, loc='outside upper center', fontsize=8, ncol=2, framealpha=1)
ax[0].legend(handles=[dashed, solid], loc='upper center', fontsize=8)
ax[1].legend(handles=[dashed_mF,solid_mF],loc='upper center', fontsize=8)
ax[0].set_xlim([0,1])
ax[0].set_ylabel(r"$P_\times$ [$\AA$]")
ax[0].set_xlabel(r"$k$ [$\AA^{-1}$]")
ax[1].set_xlabel(r"$k$ [$\AA^{-1}$]")
for i in range(2):
    ax[i].set_xlim([0,0.7])
# reduce hspace
plt.subplots_adjust(wspace=0.05)
plt.savefig("../../plots/bias_mF_changes.pdf", bbox_inches='tight', dpi=300)

# %% [markdown]
# ## Now various contaminants

# %%
cosmo = cosmology.Cosmology()
config_hcd={'verbose': True, 'include_hcd': True, 'include_metal': False,
        'include_sky': False, 'include_continuum': False, 'default_lya_model':'best_fit_arinyo_from_p1d'}
config_metal={'verbose': True, 'include_hcd': False, 'include_metal': True,
        'include_sky': False, 'include_continuum': False, 'default_lya_model':'best_fit_arinyo_from_p1d'}
config_continuum={'verbose': True, 'include_hcd': False, 'include_metal': False,
        'include_sky': False, 'include_continuum': True, 'default_lya_model':'best_fit_arinyo_from_p1d'}
config_sky={'verbose': True, 'include_hcd': False, 'include_metal': False,
        'include_sky': True, 'include_continuum': False, 'default_lya_model':'best_fit_arinyo_from_p1d'}
theory_hcd = Theory(z=data.z[0], fid_cosmo=cosmo, config=config_hcd)
theory_metal = Theory(z=data.z[0], fid_cosmo=cosmo, config=config_metal)
theory_continuum = Theory(z=data.z[0], fid_cosmo=cosmo, config=config_continuum)
theory_sky = Theory(z=data.z[0], fid_cosmo=cosmo, config=config_sky)

# %%
px_base = theories_lya[0].get_px_obs(k_AA=k, theta_arc=thetabins)
px_hcd  = theory_hcd.get_px_obs(k_AA=k, theta_arc=thetabins)
px_SiII = theory_metal.get_px_obs(k_AA=k, theta_arc=thetabins, params={'b_SiIII':0})
px_SiIII = theory_metal.get_px_obs(k_AA=k, theta_arc=thetabins, params={'b_SiII':0})
px_continuum = theory_continuum.get_px_obs(k_AA=k, theta_arc=thetabins)
px_sky = theory_sky.get_px_obs(k_AA=k, theta_arc=thetabins)

# %%
plt.figure(figsize=[4,6]) # fig, ax = plt.subplots(ncols=2, nrows=3)
# plot px_sky as full row first, and follow with every axis for the other contaminants, residuals only
xlim=[0,0.7]
for theta in range(len(px_base)):
    stable = abs(px_base[theta])>0.0005

    if theta==2:
        skylabel="sky residuals"
    else:
        skylabel=None
    plt.subplot(3,1,1)
    plt.plot(k, px_base[theta], color=colors[theta], label=rf"$\theta$={thetabins[theta]:.1f}$^\prime$", lw=1)
    plt.plot(k, px_sky[theta]-px_base[theta], color=colors[theta], linestyle='dashed', label=skylabel, lw=1)
    plt.ylabel(r"$P_\times$ [$\AA$]")
    plt.xlim(xlim)
    plt.legend()
    plt.subplot(3, 2, 3)
    plt.plot(k[stable], px_hcd[theta][stable]/px_base[theta][stable]-1, color=colors[theta], lw=1)
    plt.title("HCDs")
    plt.ylabel(r"$\Delta P_\times/P_\times$")
    plt.ylim([-.1,.5])
    plt.xlim(xlim)
    plt.subplot(3, 2, 4)
    plt.plot(k[stable], px_SiII[theta][stable]/px_base[theta][stable]-1, color=colors[theta], lw=1)
    plt.title("Si-II")
    plt.ylim([-.3,.3])
    plt.xlim(xlim)
    plt.subplot(3, 2, 5)
    plt.plot(k[stable], px_SiIII[theta][stable]/px_base[theta][stable]-1, color=colors[theta], lw=1)
    plt.title("Si-III")
    plt.xlabel(r"$k$ [$\AA^{-1}$]")
    plt.ylim([-.3,.3])
    plt.xlim(xlim)
    plt.ylabel(r"$\Delta P_\times/P_\times$")
    plt.subplot(3, 2, 6)
    plt.plot(k[stable], px_continuum[theta][stable]/px_base[theta][stable]-1, color=colors[theta], lw=1)
    plt.title("Continuum")
    plt.ylim([-.5,.1])
    plt.xlim(xlim)
    plt.xlabel(r"$k$ [$\AA^{-1}$]")

# add vspace
plt.subplots_adjust(hspace=0.4)
plt.savefig("../../plots/contaminants.pdf", bbox_inches='tight', dpi=300)


# %%
dir(cosmo)

# %%
# get the comoving size of the theta bins
theta_comoving_Mpc = 1/cosmo.get_darc_dMpc(data.z[0]) * np.array(thetabins)

# %%
theta_comoving_Mpc

# %%
