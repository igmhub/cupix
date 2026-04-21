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
