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
from lace.cosmo import camb_cosmo, fit_linP
import matplotlib.pyplot as plt
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import os
import h5py as h5
from lace.cosmo.thermal_broadening import thermal_broadening_kms
from forestflow.archive import GadgetArchive3D
import forestflow
import copy
# %load_ext autoreload
# %autoreload 2

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

ffemu = FF_emulator(z, fid_cosmo, cc, Nrealizations=1000)
theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu
dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array(z), camb_results=cc)

# %% [markdown]
# ### Step 3: Setup up the likelihood parameters

# %%
# Figure out the ForestFlow training range
path_program = os.path.dirname(forestflow.__path__[0]) + '/'
path_program
folder_lya_data = path_program + "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    average="both",
)
print(len(Archive3D.training_data))
training_data = Archive3D.training_data

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

# %%
like_with_freepar = Likelihood(forecast, theory_AA, free_param_names=['mF'], iz_choice=iz_choice, like_params=like_params, verbose=False)

# %%
like.fit_probability(values = [])

# %% [markdown]
# ## Choose the parameter you want to test

# %%
par_to_test = 'mF'
par_index = [i for i, lp in enumerate(like_params) if lp.name==par_to_test][0]

# %% [markdown]
# Make sure the probability is high / chi2 is low if we input the truth for that parameter 

# %%
like_with_freepar.fit_probability(values=like_params[par_index].get_value_in_cube(like_params[par_index].value[iz_choice]))

# %%
like_with_freepar.get_chi2(values=like_params[par_index].get_value_in_cube(like_params[par_index].value[iz_choice]))

# %%
chi2_per_param = []

par_vals = np.linspace(like_params[par_index].value[iz_choice]-0.15, like_params[par_index].value[iz_choice]+0.15, 30)
print(par_vals)
for i in range(len(par_vals)):
    # make a deep copy of like_params
    # like_params_copy = copy.deepcopy(like_params)
    # for j, param in enumerate(like_params):
    #     if param.name=='mF':
    #         # replace the first element of the likelihood_params list
    #         like_params_copy[j] = LikelihoodParameter(
    #             name='mF',
    #             min_value=min_mF,
    #             max_value=max_mF,
    #             value=[mF_vals[i]]
    #             )
    # make a new likelihood
    # like_test = Likelihood(forecast, theory_AA, free_param_names=[], iz_choice=iz_choice, like_params=like_params_copy, verbose=False)
    chi2_i = like_with_freepar.get_chi2(values=[like_params[par_index].get_value_in_cube(par_vals[i])])
    chi2_per_param.append(chi2_i)
    if i==0 or chi2_i < np.min(chi2_per_param[:-1]):
        best = par_vals[i]
        print(f"New best fit found with chi2={chi2_i} at mF={best}")
        best_chi2 = chi2_i
    # # make sure it worked
    # for param in like_params:
    #     print(param.name, param.value)

# %%
plt.plot(par_vals, chi2_per_param - np.amin(chi2_per_param))
plt.xlabel('mF')

plt.ylabel(' chi2')
plt.axvline(best, color='grey', label='best')
plt.axvline(like_params[par_index].value[iz_choice], color='red', label='truth')
plt.title(f'Log likelihood vs {par_to_test}')
# plt.ylim([0,1000000000])
plt.legend()

# %%
