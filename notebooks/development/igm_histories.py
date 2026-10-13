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
import lace
import numpy as np
import cupix
from cupix.utils.utils import get_path_repo
import os

# %%
import pandas as pd
igm = pd.read_csv("../../data/emulator/ff_training_info.csv")

# %%
igm['z']

# %%
igm['T0_min']

# %%
import matplotlib.pyplot as plt

plt.plot(igm['z'], igm['T0_min']/1e4, label='min')
plt.plot(igm['z'], igm['T0_max']/1e4, label='max')
plt.legend()

# %%
igm

# %%

plt.plot(igm['z'], igm['sigT_Mpc_min'], label='min')
plt.plot(igm['z'], igm['sigT_Mpc_max'], label='max')
plt.ylabel("$\sigma_T$ [Mpc]")
plt.legend()

# %%

plt.plot(igm['z'], igm['mF_min'], label='min')
plt.plot(igm['z'], igm['mF_max'], label='max')
plt.ylabel("$\sigma_T$ [Mpc]")
plt.legend()

# %%

plt.plot(igm['z'], igm['gamma_min'], label='min')
plt.plot(igm['z'], igm['gamma_max'], label='max')
plt.legend()

# %%
sim_suite = 'mpg'
repo = get_path_repo("lace")
cosmo_fname = os.path.join(
    repo, "data", "sim_suites", "Australia20", "mpg_emu_cosmo.npy"
)
igm_fname = os.path.join(
    repo, "data", "sim_suites", "Australia20", "IGM_histories.npy"
)

# %%
igm_fname

# %%
try:
    igm_all = np.load(igm_fname, allow_pickle=True).item()
except:
    script_fname = os.path.join(
        get_path_repo("lace"),
        "script",
        "developers",
        "save_" + sim_suite + "_IGM.py",
    )
    raise ValueError(
        f"{igm_fname} not found. You can produce it using {script_fname}"
    )

# %%
for par in igm_all:
    print(igm_all[par])
    break

# %%
b
