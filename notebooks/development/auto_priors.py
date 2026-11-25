# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
from cupix.likelihood.model_lya import get_priors_colore, get_priors_gadget, get_priors_gaikal
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
# %load_ext autoreload
# %autoreload 2
import numpy as np

# %%
gadget_short_info_file = '../../data/emulator/ff_training_info.csv'
train_test_info = pd.read_csv(gadget_short_info_file)


# %%
z_all = train_test_info['z'].to_numpy()
sortidx = np.argsort(z_all)
z_all = z_all[sortidx]

par = 'mF'
if par+"_central" in train_test_info.columns:
    plt.plot(z_all, train_test_info[par+"_central"][sortidx])
    cs_central = CubicSpline(z_all, train_test_info[par+"_central"][sortidx])
    # cs_central = PchipInterpolator(z_all, train_test_info[par+"_central"][sortidx])
    plt.plot(z_all, cs_central(z_all), ls='--')


# %%
# plot T0 values
plt.plot(z_all, train_test_info['T0_min'][sortidx]/10**4)
plt.plot(z_all, train_test_info['T0_max'][sortidx]/10**4)
plt.ylim([0,3])

# %%
# plot kF values
plt.plot(z_all, train_test_info['kF_Mpc_min'][sortidx])
plt.plot(z_all, train_test_info['kF_Mpc_max'][sortidx])
# plt.ylim([0,3])

# %%
train_test_info['T0_min'][1]

# %%
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# %%
np.sqrt(6205/10**4)*(1+2.4)/cosmo.H(2.4)*9.1 * u.km/u.s

# %%
plt.plot(z_all, train_test_info['sigT_Mpc_min'][sortidx])
plt.plot(z_all, train_test_info['sigT_Mpc_max'][sortidx])
plt.plot(z_all, train_test_info['sigT_Mpc_central'][sortidx])

# %%
for z in [2.2, 2.4, 2.6, 2.8]:
    print("z=",z)
    priors = get_priors_gadget(z, 'igm', extra_params=['T0'])
    print("T0:", priors['T0']['min']/10**4, priors['T0']['max']/10**4)
    print("sigT_Mpc", priors['sigT_Mpc']['min'], priors['sigT_Mpc']['max'])
    print("gamma:", priors['gamma']['min'], priors['gamma']['max'])
    print("mF:", priors['mF']['min'], priors['mF']['max'])
    print("kF_Mpc:", priors['kF_Mpc']['min'], priors['kF_Mpc']['max'])

# %%
z = 2.2
priors = get_priors_gadget(z, 'arinyo')
print("bias:", priors['bias']['min'], priors['bias']['max'])
print("beta:", priors['beta']['min'], priors['beta']['max'])

# %%
