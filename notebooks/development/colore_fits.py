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
# # Study the best-fit models to CoLoRe mocks
#
# They are quite crazy, with effectively Diract delta functions at mu=0
#
# We should use pressure-only models for now...

# %%
import numpy as np
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.likelihood.theory import Theory

# %%
cosmo = cosmology.Cosmology()

# %%
#default_lya_model = 'pressure_only_fits_from_colore'
default_lya_model = 'best_fit_arinyo_from_colore'
config={'verbose':False, 'default_lya_model': default_lya_model}
zs = [2.2, 2.4, 2.6, 2.8]
theories = []
for z in zs:
    print('---------------------')
    print('theory for z =', z)
    theory = Theory(z=z, fid_cosmo=cosmo, config=config)
    kv_Mpc = theory.lya_model.default_lya_params['kv_Mpc']
    bv = theory.lya_model.default_lya_params['bv']
    print('kv_Mpc = {}, bv = {}'.format(kv_Mpc, bv))
    theories.append(theory)


# %%
def plot_DNL(theory, params={}):
    k = np.logspace(-3, 1, 100)
    z = theory.z
    linP = cosmo.get_linP_Mpc(z, k)
    lya_params = theory.lya_model.get_lya_params(cosmo, params)  
    DNL_mu_0 = theory._compute_DNL_Arinyo(k, mu=0.0, linP=linP, lya_params=lya_params)
    DNL_mu_00001 = theory._compute_DNL_Arinyo(k, mu=0.0001, linP=linP, lya_params=lya_params)
    DNL_mu_1 = theory._compute_DNL_Arinyo(k, mu=1.0, linP=linP, lya_params=lya_params)
    plt.figure()
    plt.semilogx(k, DNL_mu_0, label='mu = 0')
    plt.semilogx(k, DNL_mu_1, label='mu = 1')
    plt.semilogx(k, DNL_mu_00001, ls='--', label='mu = 0.0001')
    plt.legend()
    plt.xlabel('k [1/Mpc]')
    plt.ylabel(r'$D_{\rm NL}(k, \mu)$')
    plt.ylim(0.8, 1.1)
    plt.xlim(0.01, 0.5)
    plt.axhline(y=1, ls=':', color='gray')
    plt.title('z = {}'.format(z))


# %%
for theory in theories:
    plot_DNL(theory, params={})


# %%
def compare_DNL_at_mu_0(theory, tiny_mu=1e-10):
    k = np.logspace(-3, 1, 100)
    z = theory.z
    linP = cosmo.get_linP_Mpc(z, k)
    default_lya_params = theory.lya_model.get_lya_params(cosmo, params={})
    print('default', default_lya_params['bv'], tiny_mu**default_lya_params['bv'])
    default_DNL = theory._compute_DNL_Arinyo(k, mu=tiny_mu, linP=linP, lya_params=default_lya_params)
    params = {'bv': 0.0}
    new_lya_params = theory.lya_model.get_lya_params(cosmo, params=params)
    print('new', new_lya_params['bv'], tiny_mu**new_lya_params['bv'])
    new_DNL = theory._compute_DNL_Arinyo(k, mu=tiny_mu, linP=linP, lya_params=new_lya_params)
    plt.figure()
    plt.semilogx(k, default_DNL, label=r'$b_v = {}$'.format(default_lya_params['bv']))
    plt.semilogx(k, new_DNL, ls=':', label=r'$b_v = 0.0$')
    plt.legend()
    plt.xlabel('k [1/Mpc]')
    plt.ylabel(r'$D_{\rm NL}(k, \mu=0)$')
    plt.ylim(0.8, 1.1)
    plt.xlim(0.01, 0.5)
    plt.axhline(y=1, ls=':', color='gray')
    plt.title('z = {}'.format(z))


# %%
for theory in theories:
    compare_DNL_at_mu_0(theory, tiny_mu=1e-100)


# %%
def compare_Px(theory, kp_Mpc=0.1):
    rt_Mpc = np.logspace(-3, 3, 1000)
    default_Px = theories[0].get_px_lya_Mpc(rt_Mpc=rt_Mpc, kp_Mpc=kp_Mpc)
    new_Px = theories[0].get_px_lya_Mpc(rt_Mpc=rt_Mpc, kp_Mpc=kp_Mpc, params={'bv':0.0})
    plt.figure()
    plt.semilogx(rt_Mpc, default_Px, label='0.01 * default Px')
    plt.semilogx(rt_Mpc, new_Px / default_Px - 1.0, label='relative effect new / Px')
#    plt.semilogx(k, default_DNL, label=r'$b_v = {}$'.format(default_lya_params['bv']))
#    plt.semilogx(k, new_DNL, ls=':', label=r'$b_v = 0.0$')
    plt.legend()
    plt.xlabel(r'$r_\perp$ [Mpc]')
    plt.ylabel(r'ratio $P_\times(k_\parallel, r_\perp)$ [Mpc]')
    plt.ylim(-0.01, 0.01)
#    plt.xlim(0.01, 0.5)
    plt.axhline(y=0, ls=':', color='gray')
    plt.title('z = {}, kp = {} [1/Mpc]'.format(theory.z, kp_Mpc))


# %%
for theory in theories:
    compare_Px(theory, kp_Mpc=0.1)

# %%
for theory in theories:
    compare_Px(theory, kp_Mpc=0.001)

# %%
for theory in theories:
    compare_Px(theory, kp_Mpc=0.01)

# %%
