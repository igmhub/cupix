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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## This notebook evaluates the likelihood in different regions of parameter space to test code timing and ensure good performance across the space.

# %%
import numpy as np
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from lace.cosmo import camb_cosmo
import matplotlib.pyplot as plt
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import h5py as h5
import cupix
import pandas as pd
cupixpath = cupix.__path__[0].rsplit('/', 1)[0]

# %% [markdown]
# ### Step 1: Import a noiseless forecast

# %%
# choose the mode
mode = 'cosmo_igm' # if the parameters you want to test are mean-flux, etc
# mode = 'arinyo' # if the parameters you want to test are Arinyo params

# %%
forecast_file = f"../../data/px_measurements/forecast/forecast_ffcentral_{mode}_binned_out_px-zbins_2-thetabins_18_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kmax_cut_AA=1)

# %% [markdown]
# ### Step 2: Load the emulator with Nrealizations = 1000

# %%
# Load emulator
z = forecast.z
print(z)
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

ffemu = FF_emulator(z, fid_cosmo, cc, Nrealizations=5000)
theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu

# %% [markdown]
# ### Step 3: Setup up the likelihood parameters

# %%
# read the likelihood params from forecast file
like_params = []
with h5.File(forecast_file, "r") as f:
    params = f['like_params']
    attrs = params.attrs
    for key in attrs:
        val = attrs[key][:]
        if len(val)>0:
            like_params.append(LikelihoodParameter(
                name=key,
                value=val,
                min_value=-10., # arbitrary for now
                max_value=10.
            ))
# check the parameters
for p in like_params:
    print(p.name, p.value)

# %% [markdown]
# ### Step 4: Profile likelihood

# %%
iz_choice = np.array([0])#  np.array([0,3])
like = Likelihood(forecast, theory_AA, free_param_names=[], iz_choice=iz_choice, like_params=like_params, verbose=False)

# %%
# make sure theory matches forecast data at central value
like.plot_px(iz_choice, like_params, multiply_by_k=False, ylim=[-0.00025,0.3], ylim2=[-10,10], every_other_theta=True, show=True,  title=f"Redshift {forecast.z[iz_choice]}", theorylabel=f'Model from best-fit $\chi$', datalabel='Stack on true-continuum mocks')


# %%
print(like.fit_probability(values=None))

# %%
like.get_chi2(values=None)

# %% [markdown]
# ## Choose the parameter you want to test

# %%
par_to_test = 'sigT_Mpc'

# %%
like_with_freepar = Likelihood(forecast, theory_AA, free_param_names=[par_to_test], iz_choice=iz_choice, like_params=like_params, verbose=False)

# %%
like.fit_probability(values = [])

# %%
par_index = [i for i, lp in enumerate(like_params) if lp.name==par_to_test][0]

# %% [markdown]
# Make sure the probability is high / chi2 is low if we input the truth for that parameter 

# %%
like_with_freepar.fit_probability(values=like_params[par_index].get_value_in_cube(like_params[par_index].value[iz_choice]))

# %%
like_with_freepar.get_chi2(values=like_params[par_index].get_value_in_cube(like_params[par_index].value[iz_choice]))

# %%
# get the min/ max values of parameters from the training simulations
gadget_short_info_file = cupixpath + '/data/emulator/ff_training_info.csv'
train_test_info = pd.read_csv(gadget_short_info_file)
train_test_z = np.where(z[iz_choice][0]==train_test_info["z"])[0][0]
print("Found min/max values at z=", train_test_info.iloc[train_test_z]['z'])
min_par_val = train_test_info.iloc[train_test_z][f'{par_to_test}_min']
max_par_val = train_test_info.iloc[train_test_z][f'{par_to_test}_max']
print(f"Min/max values of {par_to_test} in training sims at this z: {min_par_val}, {max_par_val}")

# %%
chi2_per_param = []

par_vals = np.linspace(min_par_val, max_par_val, 30)
print(par_vals)
for i in range(len(par_vals)):
    chi2_i = like_with_freepar.get_chi2(values=[like_params[par_index].get_value_in_cube(par_vals[i])])
    chi2_per_param.append(chi2_i)
    if i==0 or chi2_i < np.min(chi2_per_param[:-1]):
        best = par_vals[i]
        print(f"New best fit found with chi2={chi2_i} at mF={best}")
        best_chi2 = chi2_i
    

# %%
plt.plot(par_vals, chi2_per_param - np.amin(chi2_per_param))
plt.xlabel(par_to_test)

plt.ylabel(r'$\Delta \chi^2$')
plt.axvline(best, color='grey', label='best')
plt.axvline(like_params[par_index].value[iz_choice], color='red', label='truth')
plt.title(f'chi2 vs {par_to_test}')
# plt.ylim([0,1000000000])
plt.legend()

# %%
