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

# %%
import numpy as np
from cupix.likelihood.generate_fake_data import FakeData
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
import matplotlib.pyplot as plt
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import forestflow
from forestflow.archive import GadgetArchive3D
import os
from pathlib import Path
import cupix
import pandas as pd
import h5py 
# %load_ext autoreload
# %autoreload 2


   # %%
   # Figure out the ForestFlow training central simulation
path_program = os.path.dirname(forestflow.__path__[0]) + '/'
path_program
folder_lya_data = path_program + "/data/best_arinyo/"
print("Loading archive.")
Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    # folder_data=folder_lya_data,
    # average="both",
)
sim_dict_central =  Archive3D.get_testing_data("mpg_central")
training_data = Archive3D.training_data



for sim_z in sim_dict_central:
    print(sim_z['Arinyo_minz'])
    break
    # for par in all_pars:
    #     if par in sim_z['Arinyo']:
    #         dict_save[par+"_central"].append(sim_z['Arinyo'][par])
    #     elif par in sim_z:
    #         dict_save[par+"_central"].append(sim_z[par])
    #     else:
    #         print("Parameter", par, "not found in central simulation at z=", sim_z['z'])
    #     # now find min, max values from training data
    #     all_training_values_iz = []
    #     for i in range(len(training_data)):
    #         if abs(training_data[i]['z']-sim_z['z'])<0.01:
    #             if par in training_data[i]['Arinyo']:
    #                 all_training_values_iz.append(training_data[i]['Arinyo'][par])
    #             elif par in training_data[i]:
    #                 all_training_values_iz.append(training_data[i][par])
    #     dict_save[par+"_min"].append(np.amin(np.asarray(all_training_values_iz)))
    #     dict_save[par+"_max"].append(np.amax(np.asarray(all_training_values_iz)))

# %%
for sim_z in sim_dict_central:
    print(sim_z['Arinyo_minz'])
    print(sim_z['Arinyo_min'])
    break

# %% [markdown]
# ## Set up the forecast configuration

# %%
# --------- settings ---------
forecast = "central" # "central" or "random"
param_mode = "igm" # "igm" or "arinyo"
add_noise = False
Nfree = 2 # number of free parameters to fit in validation
run_minimizer = True # if false, will only generate the forecast
# "central" uses the central simulation in the training set
# "random" randomly selects points within the minimum and maximum range of each training parameter
# --------------------------

# %%
data_file = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/binned_out_px-zbins_4-thetabins_9_w_res.hdf5"
# data_file = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-0/tru_cont/binned_out_px-zbins_2-thetabins_18.hdf5" # mock
data = DESI_DR2(data_file)
data_label = 'real'

# %%
igm_pars = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
arinyo_pars = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']
# set cosmo+IGM pars and Arinyo pars
if param_mode == "igm":
    pars = igm_pars
elif param_mode == "arinyo":
    pars = arinyo_pars

# %%
cupixpath = cupix.__path__[0].rsplit('/', 1)[0]
igm_pars = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
arinyo_pars = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']
gadget_short_info_file = cupixpath + '/data/emulator/ff_training_info.csv'
if not os.path.exists(gadget_short_info_file):
    # Figure out the ForestFlow training central simulation
    path_program = os.path.dirname(forestflow.__path__[0]) + '/'
    path_program
    folder_lya_data = path_program + "/data/best_arinyo/"
    print("Loading archive.")
    Archive3D = GadgetArchive3D(
        base_folder=path_program[:-1],
        folder_data=folder_lya_data,
        average="both",
    )
    sim_dict_central =  Archive3D.get_testing_data("mpg_central")
    training_data = Archive3D.training_data
    all_pars = arinyo_pars + igm_pars
    dict_save = {par+"_central": [] for par in all_pars} # save the central values of each parameter at each redshift
    # save the min-max range of each parameter at each redshift
    dict_save.update({par+"_min": [] for par in all_pars})
    dict_save.update({par+"_max": [] for par in all_pars})
    dict_save["z"] = []
    for sim_z in sim_dict_central:
        dict_save["z"].append(sim_z['z']) # save the redshifts
        for par in all_pars:
            if par in sim_z['Arinyo']:
                dict_save[par+"_central"].append(sim_z['Arinyo'][par])
            elif par in sim_z:
                dict_save[par+"_central"].append(sim_z[par])
            else:
                print("Parameter", par, "not found in central simulation at z=", sim_z['z'])
            # now find min, max values from training data
            all_training_values_iz = []
            for i in range(len(training_data)):
                if abs(training_data[i]['z']-sim_z['z'])<0.01:
                    if par in training_data[i]['Arinyo']:
                        all_training_values_iz.append(training_data[i]['Arinyo'][par])
                    elif par in training_data[i]:
                        all_training_values_iz.append(training_data[i][par])
            dict_save[par+"_min"].append(np.amin(np.asarray(all_training_values_iz)))
            dict_save[par+"_max"].append(np.amax(np.asarray(all_training_values_iz)))
    train_test_info = pd.DataFrame(dict_save)
    train_test_info.to_csv(gadget_short_info_file, index=False)
else:
    print("Gadget simulation info file already exists at", gadget_short_info_file, "\nLoading it.")
    train_test_info = pd.read_csv(gadget_short_info_file)

# %%
# choose some redshifts from the training data
zs = []

for sim_z in train_test_info['z']:
    if (sim_z < 2.7) & (sim_z > 2.2):
        zs.append(sim_z)
zs = np.sort(np.array(zs))
Nz = len(zs)
print("Will generate forecast at redshifts:", zs)
# manually overwrite data zs with ForestFlow zs
data.z = zs

# %% [markdown]
# Load the theory

# %%
# Load theory
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
cosmo = {
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

theory = set_theory(zs, bkgd_cosmo=cosmo, default_theory=f'best_fit_{param_mode}_from_p1d', p3d_label='arinyo', emulator_label='forestflow_emu', k_unit='iAA', verbose=True)

# %% [markdown]
# ## Generate it at the center of the original training simulation

# %%
# prepare to generate and write fake data
filepath = cupixpath + "/data/px_measurements/forecast/"

if add_noise:
    noise_str = 'noisy'
else:
    noise_str = 'noiseless'
if forecast=="random":
    rng = np.random.default_rng()
    forecast_str = f"_{rng.choice(1000):03d}"
else:
    forecast_str = ""

savestr = f"{filepath}/forecast_ff{forecast}{forecast_str}_{param_mode}_{data_label}_{Path(data_file).stem}_{noise_str}.hdf5"
if os.path.exists(savestr):
    print("File already exists at", savestr)
else:
    print("Will generate forecast and save to", savestr)

# %%
like_params = []
like_params_dict = {par:np.zeros(len(zs)) for par in pars}

for iz,zz in enumerate(zs):
    iz_traintest = np.where(train_test_info['z']==zz)[0][0]
    
    if forecast=="central":
        for par in pars:
            if par+"_central" in train_test_info.columns:
                like_params_dict[par][iz] = train_test_info[par+"_central"][iz_traintest]
            else:
                print("Parameter", par, "not found in training info file for redshift", zz)
                like_params_dict[par][iz] = 0

    elif forecast=="random":
        # randomly select a value within the min and max range of each parameter at this redshift
        for par in pars:
            if par+"_min" not in train_test_info.columns or par+"_max" not in train_test_info.columns:
                print("Min/max of parameter", par, "not found in training info file for redshift", zz)
            else:
                like_params_dict[par][iz] = rng.uniform(train_test_info[par+"_min"][iz_traintest], train_test_info[par+"_max"][iz_traintest])
                print("new parameter value for", par, like_params_dict[par][iz])


# %%

# create LikelihoodParameter objects
for par in like_params_dict:
    like_params.append(LikelihoodParameter(name=par, min_value=-1000, max_value=1000, value=like_params_dict[par])) # min/max values don't matter here


# %%
# initialize Likelihood
like = Likelihood(data, theory, z=2.25, verbose=True)


# %%
# initialize fake data
fakedata = FakeData(like)

# %%
print("Saving to...", savestr)
# use default theory
fakedata.write_to_file(savestr, add_noise=False)

# %%
# just copying and pasting from the previous output
# arinyo_dict = {'bias': np.array([0.11810589, 0.14949374]), 'beta': np.array([1.52642012, 1.4411819 ]), 'q1': np.array([0.29296646, 0.31081784]), 'kvav': np.array([0.51745933, 0.54350346]), 'av': np.array([0.32345456, 0.37763417]), 'bv': np.array([1.6457237 , 1.69155955]), 'kp': np.array([12.90727806, 13.73200226]), 'q2': np.array([0.26116902, 0.28188545])}
# arinyo_dict = {'bias': np.array([0.11719166, 0.14775302, 0.18601523, 0.23090219]), 'beta': np.array([1.52999781, 1.44227235, 1.3310534 , 1.2097371 ]), 'q1': np.array([0.30040638, 0.32946218, 0.3630141 , 0.40578129]), 'kvav': np.array([0.51962305, 0.55578851, 0.60605053, 0.67135953]), 'av': np.array([0.33154755, 0.39208764, 0.46333845, 0.53290917]), 'bv': np.array([1.65384236, 1.6902841 , 1.73748284, 1.78617228]), 'kp': np.array([13.15922816, 13.99669913, 14.89153095, 15.82119602]), 'q2': np.array([0.25414535, 0.26888122, 0.26672887, 0.25766121])}
# arinyo_dict = {'bias': np.array([0.11719166, 0.14775302, 0.18601523, 0.23090219]), 'beta': np.array([1.52999781, 1.44227235, 1.3310534 , 1.2097371 ]), 'q1': np.array([0.30040638, 0.32946218, 0.3630141 , 0.40578129]), 'kvav': np.array([0.51962305, 0.55578851, 0.60605053, 0.67135953]), 'av': np.array([0.33154755, 0.39208764, 0.46333845, 0.53290917]), 'bv': np.array([1.65384236, 1.6902841 , 1.73748284, 1.78617228]), 'kp': np.array([13.15922816, 13.99669913, 14.89153095, 15.82119602]), 'q2': np.array([0.25414535, 0.26888122, 0.26672887, 0.25766121])}
arinyo_dict = {'bias': np.array([0.11735305]), 'beta': np.array([1.4121992]), 'q1': np.array([0.25009289]), 'kvav': np.array([0.53466132]), 'av': np.array([0.42833239]), 'bv': np.array([1.67831029]), 'kp': np.array([10.24335181]), 'q2': np.array([0.27303594])}


# %%
# add the Arinyo parameters to the attributes
loaded_file = h5py.File(savestr, 'a')
print(loaded_file.keys()) 
for par in arinyo_dict:
    loaded_file['arinyo_pars'].attrs[par] = arinyo_dict[par]
loaded_file.close()

# %% [markdown]
# ## Test the forecast output

# %%
# make sure it worked
forecast = DESI_DR2(savestr)

# %%
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = forecast.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_M={Nk_M}, Nk_m={Nk_m}")
# rebinned values
Nz, Nt_A, Nk_M = forecast.Px_ZAM.shape
print(f"rebinned values: Nz={Nz}, Nt_A={Nt_A}, Nk_M={Nk_M}")

# %%
# get the central value of each redshift bin, of length Nz
zs = forecast.z
print(zs)
# get a 1D array of central values of the measured k bins, of length Nk_M
k_M = data.k_M_centers_AA
# get two 1D arrays with the edges of each theta bin, of length Nt_A each
theta_A_min = data.theta_min_A_arcmin
theta_A_max = data.theta_max_A_arcmin


# %%
# make a plot for a couple of theta bins, and one redshift bin
def plot_theta_bins(data, k_M, iz, it_M):
    label = '{:.2f} < theta < {:.2f}'.format(theta_A_min[it_M], theta_A_max[it_M])
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it_M]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_M]))
    plt.errorbar(k_M, Px, sig_Px, label=label)
    


# %%
plot_theta_bins(forecast, k_M, iz=0, it_M=0)
plot_theta_bins(data, k_M, iz=0, it_M=0)
plt.legend()

# %%
for i in range(8):
    plot_theta_bins(forecast, k_M, iz=0, it_M=i)
plt.legend()

# %%
