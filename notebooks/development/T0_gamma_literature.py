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

# %% [markdown]
# # Get $T_0$ and $\gamma$ from Gaikal et al

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# From Table 2 in https://arxiv.org/pdf/2009.00016:
#
# 2.0 ± 0.1 9500 ± 1393 1.500 ± 0.096
# 2.2 ± 0.1 11000 ± 1028 1.425 ± 0.133
# 2.4 ± 0.1 12750 ± 1132 1.325 ± 0.122
# 2.6 ± 0.1 13500 ± 1390 1.275 ± 0.122
# 2.8 ± 0.1 14750 ± 1341 1.250 ± 0.109
# 3.0 ± 0.1 14750 ± 1322 1.225 ± 0.120
# 3.2 ± 0.1 12750 ± 1493 1.275 ± 0.129
# 3.4 ± 0.1 11250 ± 1125 1.350 ± 0.108
# 3.6 ± 0.1 10250 ± 1070 1.400 ± 0.101
# 3.8 ± 0.1 9250 ± 876 1.525 ± 0.140

# %%
z = np.array([2.2, 2.4, 2.6, 2.8])
T0 = np.array([11000, 12750, 13500, 14750])
T0_errs = np.array([1028, 1132, 1390, 1341])
gamma = np.array([1.425, 1.325, 1.275, 1.250])
gamma_errs = np.array([0.133, 0.122, 0.122, 0.109])
plt.errorbar(z, T0/10**3, yerr=T0_errs/10**3, fmt='o', label='T0 vs z')
plt.ylabel(r'T0 [$10^3$ K]')

# %%
plt.errorbar(z, gamma, yerr=gamma_errs, fmt='o', label='gamma vs z')
plt.xlabel('z')
plt.ylabel(r'$\gamma$')

# %%
myzs = np.linspace(2.2, 2.8, 100)
for myz in myzs:
    T0_interp = np.interp(myz, z, T0)
    T0_err_interp = np.interp(myz, z, T0_errs)
    gamma_interp = np.interp(myz, z, gamma)
    gamma_err_interp = np.interp(myz, z, gamma_errs)
    plt.errorbar(myz, T0_interp/10**3, yerr=T0_err_interp/10**3, fmt='.', color='blue', alpha=0.1)
plt.errorbar(z, T0/10**3, yerr=T0_errs/10**3, fmt='o', color='blue', label='T0 vs z')

# %%
