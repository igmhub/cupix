# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: cupix
# ---

# %% [markdown]
# # copied from ForestFlow: Tutorial for how to calculate $P_\times$

# %% [markdown]
#
# Reproduce results with the new Px functions in cupix

# %%
import numpy as np
from scipy import special
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
# %load_ext autoreload
# %autoreload 2


# %%
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
# import P3D theory
from lace.cosmo import cosmology
from forestflow.model_p3d_arinyo import ArinyoModel
from cupix.likelihood import theory
from forestflow.pcross import Px_Mpc, Px_Mpc_detailed

# %% [markdown]
# First, choose a redshift and $k$ range. Initialize an instance of the Arinyo class for this redshift given cosmology calculations from Camb.

# %%
zs = np.array([2, 2.5])  # set target redshift


# %%
cosmo_params = {
    "H0": 67.66,
    "mnu": 0,
    "omch2": 0.119,
    "ombh2": 0.0224,
    "omk": 0,
    'As': 2.105e-09,
    'ns': 0.9665,
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}
cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params)
model_Arinyo = ArinyoModel(fid_cosmo=cosmo)
model_Arinyo.default_params

# %% [markdown]
# ## Plot the 3D power spectrum

# %%
nn_k = 200  # number of k bins
nn_mu = 10  # number of mu bins
k = np.logspace(-3, 2, nn_k)
mu = np.linspace(0, 1, nn_mu)
k2d = np.tile(k[:, np.newaxis], nn_mu)  # k grid for P3D
mu2d = np.tile(mu[:, np.newaxis], nn_k).T  # mu grid for P3D

kpar = np.logspace(-1, np.log10(5), nn_k)  # kpar for P1D

plin = model_Arinyo.linP_Mpc(zs[0], k)  # get linear power spectrum at target z
p3d = model_Arinyo.P3D_Mpc_k_mu(
    zs[0], k2d, mu2d, ari_pp=model_Arinyo.default_params
)  # get P3D at target z
p1d = model_Arinyo.P1D_Mpc(
    zs[0], kpar, ari_pp=model_Arinyo.default_params
)  # get P1D at target z

# %%
for ii in range(p3d.shape[1]):
    plt.loglog(
        k, p3d[:, ii] / plin, label=r"$<\mu>=$" + str(np.round(mu[ii], 2))
    )
plt.xlabel(r"$k$ [Mpc]")
plt.ylabel(r"$P/P_{\rm lin}$")
plt.xlim([10**-1, 10**2])
plt.ylim([10**-3, 1])
plt.legend()

# %%

# %%
new_theory = theory.Theory(zs=zs, fid_cosmo=cosmo, config={'verbose':True})

# %%
# we use slightly different conventions
new_params = {}
for par in ['bias', 'beta', 'q1', 'q2', 'av', 'bv']:
    new_params[par] = model_Arinyo.default_params[par]
new_params['kp_Mpc'] = model_Arinyo.default_params['kp']
new_params['kv_Mpc'] = np.exp(np.log(model_Arinyo.default_params['kvav'])/model_Arinyo.default_params['av'])

# %%
new_p3d = new_theory.get_p3d_lya_Mpc(iz=0, k=k2d, mu=mu2d, params=new_params)

# %%
for ii in [-1, 0]:
    plt.loglog(k, p3d[:, ii] / plin, label=r"$<\mu>=$" + str(np.round(mu[ii], 2)))
    plt.loglog(k, new_p3d[:, ii] / plin, ls=':')
plt.xlabel(r"$k$ [Mpc]")
plt.ylabel(r"$P/P_{\rm lin}$")
plt.xlim([10**-1, 40])
plt.ylim([10**-3, 1])
plt.legend()

# %%
test_p3d = new_theory.get_p3d_lya_hcd_Mpc(iz=0, k=k2d, mu=mu2d, params=new_params)

# %%
for ii in [-1, 0]:
    plt.loglog(k, new_p3d[:, ii] / plin, label=r"$<\mu>=$" + str(np.round(mu[ii], 2)))
    plt.loglog(k, test_p3d[:, ii] / plin, ls=':')
plt.xlabel(r"$k$ [Mpc]")
plt.ylabel(r"$P/P_{\rm lin}$")
plt.xlim([10**-3, 40])
plt.ylim([10**-3, 1])
plt.legend()

# %% [markdown]
# ### Now test Px (in comoving Mpc)

# %%
rperp = np.logspace(-2,2,100) # use the same rperp for each z. We could also input this as a list of [rperp, rperp] for each z.

# %%
# we can compute Px from within the Arinyo class using default parameters,
Px_Mpc_1 = model_Arinyo.Px_Mpc(z=zs[0], kpar_iMpc = kpar, rperp_Mpc = rperp, ari_pp=model_Arinyo.default_params)

# we could have also done it outside of the class with the function Px_Mpc:
Px_Mpc_2 = Px_Mpc(
    zs[0], kpar, rperp, model_Arinyo.P3D_Mpc_k_mu, p3d_params=model_Arinyo.default_params
)
print("Detailed method is equal to previous method:", np.allclose(Px_Mpc_1, Px_Mpc_2, atol=1e-15))

# %%
Px_Mpc_3 = new_theory.get_px_lya_Mpc(iz=0, rt_Mpc=rperp, kp_Mpc=kpar, params=new_params)

# %%
print("New method:", np.allclose(Px_Mpc_1, Px_Mpc_3, atol=1e-15))

# %% [markdown]
# ### Play with contaminants (comoving Mpc)

# %%
px_lya = new_theory.get_px_lya_Mpc(iz=0, rt_Mpc=rperp, kp_Mpc=kpar, params=new_params)

# %%
px_lya_hcd = new_theory.get_px_lya_hcd_Mpc(iz=0, rt_Mpc=rperp, kp_Mpc=kpar, params=new_params)

# %%
for irt in [0, 50, 70]:
    plt.semilogx(kpar, px_lya[irt], label='rt = {:.3f} Mpc'.format(rperp[irt]))
    plt.semilogx(kpar, px_lya_hcd[irt], ls=':')
plt.legend()
plt.ylabel(r'$P_\times(r_\perp, k_\parallel)$ [Mpc]')
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
plt.title('Impact of HCD contamination')

# %%
px_metal_auto = new_theory.get_px_metal_auto_Mpc(iz=0, rt_Mpc=rperp, kp_Mpc=kpar, params=new_params)
px_metal_cross = new_theory.get_px_metal_cross_Mpc(iz=0, rt_Mpc=rperp, kp_Mpc=kpar, params=new_params)

# %%
for irt in [0, 50, 70]:
    plt.semilogx(kpar, px_lya[irt], label='rt = {:.3f} Mpc'.format(rperp[irt]))
    plt.semilogx(kpar, px_lya[irt] + px_metal_auto[irt], ls=':')
    plt.semilogx(kpar, px_lya[irt] + px_metal_cross[irt], ls='--')
plt.legend()
plt.ylabel(r'$P_\times(r_\perp, k_\parallel)$ [Mpc]')
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
plt.title('Impact of metal contamination')

# %%
px_sky = new_theory.get_px_sky_Mpc(iz=0, rt_Mpc=rperp, kp_Mpc=kpar, params=new_params)

# %%
for irt in [0, 50, 70]:
    plt.semilogx(kpar, px_lya[irt], label='rt = {:.3f} Mpc'.format(rperp[irt]))
    plt.semilogx(kpar, px_lya[irt] + 10*px_sky[irt], ls=':')
plt.legend()
plt.ylabel(r'$P_\times(r_\perp, k_\parallel)$ [Mpc]')
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
plt.title('Impact of correlated sky residuals (x10)')

# %% [markdown]
# ### Now test Px (in observing units)

# %%
theta_arc = 5.0
k_AA = np.linspace(0.01,1.0, 1000)

# %%
px_lya = new_theory.get_px_lya_obs(iz=0, theta_arc=theta_arc, k_AA=k_AA, params=new_params)

# %%
px_lya_hcd = new_theory.get_px_lya_hcd_obs(iz=0, theta_arc=theta_arc, k_AA=k_AA, cosmo=cosmo, params=new_params)

# %%
px_metal_auto = new_theory.get_px_metal_auto_obs(iz=0, theta_arc=theta_arc, k_AA=k_AA, params=new_params)

# %%
px_metal_cross = new_theory.get_px_metal_cross_obs(iz=0, theta_arc=theta_arc, k_AA=k_AA, params=new_params)

# %%
px_sky = new_theory.get_px_sky_obs(iz=0, theta_arc=theta_arc, k_AA=k_AA, params=new_params)
if px_sky.shape[0]==1:
    px_sky = px_sky.squeeze()

# %%
plt.semilogx(k_AA, px_lya, label='Lya')
plt.semilogx(k_AA, px_lya_hcd, label='Lya + HCD')
plt.semilogx(k_AA, px_lya + px_sky, label='Lya + sky')
plt.semilogx(k_AA, px_lya_hcd + px_metal_auto + px_metal_cross, label='Lya + HCD + metals')

plt.legend()
plt.ylabel(r'$P_\times(\theta, q)$ [AA]')
plt.xlabel(r'q [1/AA]')
plt.title(r'Impact of contamination at z = {:.2f}, $\theta={:.2f}$ arcmin'.format(zs[0], theta_arc))

# %%
plt.semilogx(k_AA, px_lya_hcd / px_lya, label='Lya + HCD')
plt.semilogx(k_AA, (px_lya + px_sky) / px_lya, label='Lya + sky')
plt.semilogx(k_AA, (px_lya_hcd + px_metal_auto + px_metal_cross) / px_lya, label='Lya + HCD + metals')
plt.axhline(y=1, ls=':', color='gray')
plt.legend()
plt.ylim([0.9,1.3])
plt.ylabel(r'$P_\times(\theta, q) ~/ ~ P_\times^{\alpha}(\theta, q) $')
plt.xlabel(r'q [1/AA]')
plt.title(r'Impact of contamination at z = {:.2f}, $\theta={:.2f}$ arcmin'.format(zs[0], theta_arc))
plt.savefig('px_cont.png')

# %%
