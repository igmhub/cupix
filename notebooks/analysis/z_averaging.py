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
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
# %load_ext autoreload
# %autoreload 2
from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.theory import Theory

# %%
zs = np.linspace(2.3, 2.5, 10)
print(zs)

# %%
cosmo_dict = {'H0': 67}
cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_dict)

theories = []
for z in zs:
    theories.append(Theory(z, fid_cosmo=cosmo))

# %%
# numpy array of kpar values (in inverse AA)
kp_AA = np.linspace(0.01, 2.0, 100)
# numpy array of theta values (in arcmin)
theta_arc = np.linspace(0.1, 60.0, 100)

# %%
models = []
for iz, z in enumerate(zs):
    models.append(theories[iz].get_px_obs(theta_arc=theta_arc, k_AA=kp_AA))

# %%
theory_center = Theory(z=2.4, fid_cosmo=cosmo)
model_center  = theory_center.get_px_obs(theta_arc=theta_arc, k_AA=kp_AA)

# %%
np.asarray(models).shape

# %%
np.where(kp_AA>.7)

# %%
it = 1
for iz, z in enumerate(zs):
    label = f'z = {iz}'
    plt.plot(kp_AA, models[iz][it], label=label, alpha=.2)
model_avg = np.average(np.asarray(models), axis=0)
plt.plot(kp_AA, model_center[it], ls='solid', color='k', label='at z=2.4')
plt.plot(kp_AA, model_avg[it], ls='dashed', color='k', label='average over bin')
plt.legend()
plt.show()
plt.clf()


for it,theta in enumerate(theta_arc):
    px_has_signal = model_avg[it]>.001
    print(f"For theta={theta:.2f}, iz={it}, Assumption is {np.abs(np.average((model_center[it][px_has_signal]/model_avg[it][px_has_signal]-1)*100)):.2f} percent off")
    

# %%
