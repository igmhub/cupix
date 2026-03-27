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
# # Set up new theory object and play with some plots

# %%
import numpy as np
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.likelihood.theory import Theory

# %%
cosmo_params = {"H0": 67.66}
cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params)

# %%
z=2.25
theory = Theory(z=z, fid_cosmo=cosmo, config={'verbose':True})

# %%
# numpy array of kpar values (in inverse AA)
kp_AA = np.linspace(0.01, 2.0, 1000)
# numpy array of theta values (in arcmin)
theta_arc = np.linspace(0.1, 60.0, 100)
# get a 2D array prediction
px_obs = theory.get_px_lya_obs(theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params={})

# %%
# plot the prediction for a couple of theta values
for it_A in [10, 20, 40]:
    label = "theta = {:.2f}' ".format(theta_arc[it_A])
    plt.plot(kp_AA, px_obs[it_A], label=label)
plt.xlim([0,1])
plt.xlabel('k [1/A]')
plt.ylabel('Px [A]')
plt.legend();

# %% [markdown]
# ### Now make predictions for different parameter values

# %%
theory.lya_model.default_lya_params

# %%
# this can be a list of likelihood parameters or a dictionary
params = {'bias': -0.12, 'beta': 1.6, 'q1': .3}
px_obs = theory.get_px_lya_obs(theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params=params)

# %%
for beta in [0.0, 1.0, 2.0]:
    px_obs = theory.get_px_lya_obs(theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params={'beta':beta})
    for it_A in [5]:
        label = 'theta = {:.2f} arcmin, beta={:.2f}'.format(theta_arc[it_A], beta)
        plt.plot(kp_AA, px_obs[it_A], label=label)
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')

# %%
for q1 in [0.0, 0.5, 0.9]:
    px_obs = theory.get_px_lya_obs(theta_arc=theta_arc, k_AA=kp_AA, cosmo=cosmo, params={'q1':q1})
    
    for it_A in [5]:
        label = 'theta = {:.2f} arcmin, q1={:.2f}'.format(theta_arc[it_A], q1)
        plt.plot(kp_AA, px_obs[it_A], label=label)
plt.legend()
plt.xlabel(r'$k_\parallel$ (1/Ang)')
plt.ylabel('Px [Ang]')

# %% [markdown]
# ### Vary cosmology and check that everything works

# %%
N=5
nss = np.linspace(0.94, 0.98, N)
k = np.logspace(-4, 2, 100)


# %%
def plot_linP_ratios():
    plins = []
    for i in range(N):
        cosmo = theory.get_cosmology(cosmo=None, params={'ns':nss[i]})
        print('new cosmo', cosmo.new_params)
        plins.append(cosmo.get_linP_Mpc(theory.z, k))
    iref=2
    plin_ref=plins[iref]
    for i in range(N):
        plt.semilogx(k, plins[i] / plins[iref], label=r'$n_s={:.3f}$'.format(nss[i]))
    plt.xlabel('k [1/Mpc]')
    plt.ylabel(r'ratio of $P_L (k)$')
    plt.legend()
    plt.ylim([0.95,1.05])


# %%
plot_linP_ratios()


# %%
def plot_p3d_ratios(mu):
    p3ds = []
    for i in range(N):
        p3ds.append(theory.get_p3d_lya_Mpc(k, mu, params={'ns':nss[i]}))
    iref=2
    p3d_ref=p3ds[iref]
    for i in range(N):
        plt.semilogx(k, p3ds[i] / p3ds[iref], label=r'$n_s={:.3f}$'.format(nss[i]))
    plt.xlabel('k [1/Mpc]')
    plt.ylabel(r'ratio of $P_{{3D}} (k, \mu={:.1f})$'.format(mu))
    plt.legend()
    plt.ylim([0.8,1.2])


# %%
plot_p3d_ratios(mu=0.0)

# %%
plot_p3d_ratios(mu=1.0)


# %%
def plot_px_ratios(rt=10.0):
    pxs = []
    for i in range(N):
        pxs.append(theory.get_px_lya_Mpc(rt_Mpc=rt, kp_Mpc=k, params={'ns':nss[i]}))
    iref=2
    px_ref=pxs[iref]
    for i in range(N):
        plt.semilogx(k, pxs[i] / pxs[iref], label=r'$n_s={:.3f}$'.format(nss[i]))
    plt.xlabel('k [1/Mpc]')
    plt.ylabel(r'ratio of $P_{{3D}} (k, \mu={:.1f})$'.format(rt))
    plt.legend()
    plt.ylim([0.8,1.2])


# %%
plot_px_ratios(rt=1.0)

# %%
plot_px_ratios(rt=20.0)

# %%
