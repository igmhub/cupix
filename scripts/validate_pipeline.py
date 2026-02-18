# This script is based off the notebooks generate_forecast_data.ipynb and pipeline_validation_using_forecast.ipynb

import numpy as np
from cupix.likelihood.generate_fake_data import FakeData
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from lace.cosmo import camb_cosmo, fit_linP
import matplotlib.pyplot as plt
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import scipy
import os
from forestflow.archive import GadgetArchive3D
import forestflow
from lace.cosmo.thermal_broadening import thermal_broadening_kms
import h5py as h5
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer, save_analysis_npz
import cupix
from pathlib import Path
import time
import pandas as pd

# --------- settings ---------
forecast = "central" # "central" or "random"
param_mode = "cosmo_igm" # "cosmo_igm" or "arinyo"
add_noise = False
Nfree = 2 # number of free parameters to fit in validation
run_minimizer = True # if false, will only generate the forecast
# "central" uses the central simulation in the training set
# "random" randomly selects points within the minimum and maximum range of each training parameter
# --------------------------

# define paths
# cupixpath = cupix.__path__[0].rsplit('/', 1)[0] # use this line if trying to write to the cupix package
cupixpath = "/pscratch/sd/m/mlokken/desi-lya/px/"
filepath = cupixpath + "/data/px_measurements/forecast/"

# if it does not yet exist, start by saving a file with all the necessary information about the Gadget simulations:
# z
# per z, the min and max value of each parameter in the training set
# per z, the value of each parameter in the central simulation
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
    print("Gadget simulation info file already exists at", gadget_short_info_file, "Loading it.")
    train_test_info = pd.read_csv(gadget_short_info_file)

# set cosmology
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

# set cosmo+IGM pars and Arinyo pars
if param_mode == "cosmo_igm":
    pars = igm_pars
elif param_mode == "arinyo":
    pars = arinyo_pars


# Part 1: generate the forecast
# We will pull the theta bins, k bins, and covariance and window matrices from an existing mock data file
mockdata_file = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-0/tru_cont/binned_out_px-zbins_2-thetabins_18.hdf5"

# prepare to generate and write fake data

if add_noise:
    noise_str = 'noisy'
else:
    noise_str = 'noiseless'
if forecast=="random":
    rng = np.random.default_rng()
    forecast_str = f"_{rng.choice(1000):03d}"
else:
    forecast_str = ""

savestr = f"{filepath}/forecast_ff{forecast}{forecast_str}_{param_mode}_{Path(mockdata_file).stem}_{noise_str}.hdf5"

mockdata = DESI_DR2(mockdata_file)

# choose some redshifts from the training data
zs = []

for sim_z in train_test_info['z']:
    if (sim_z < 2.7) & (sim_z > 2.2):
        zs.append(sim_z)
zs = np.sort(np.array(zs))
Nz = len(zs)
print("Will generate forecast at redshifts:", zs)
# manually overwrite MockData zs with ForestFlow zs
mockdata.z = zs

# Load emulator
print("Loading emulator.")
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=zs, camb_kmax_Mpc=1000)
ffemu = FF_emulator(zs, fid_cosmo, cc)
ffemu.kp_Mpc = 1 # set pivot point
theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(zs)
theory_AA.emulator = ffemu


    
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

# create LikelihoodParameter objects
for par in like_params_dict:
    like_params.append(LikelihoodParameter(name=par, min_value=-1000, max_value=1000, value=like_params_dict[par])) # min/max values don't matter here

if os.path.exists(savestr):
    print("Forecast file already exists:", savestr, "Moving on to minimization.")
else:
    # initialize Likelihood
    like = Likelihood(mockdata, theory_AA, free_param_names=[], iz_choice=[0,1], like_params=like_params, verbose=False)
    # initialize fake data
    fakedata = FakeData(like)
    print("Saving to...", savestr)
    fakedata.write_to_file(savestr, add_noise=False)

if run_minimizer:
    # part 2: validate the pipeline by fitting to the fake data
    # read in the fake data
    mockdata_validation = DESI_DR2(savestr, kmax_cut_AA=1)
    # get the free parameters
    if param_mode == "arinyo":
        free_param_names = ['bias', 'beta'] # for Arinyo, we only fit bias and beta in this validation test
    elif param_mode == "cosmo_igm":
        free_param_names = ['mF', 'sigT_Mpc']
    print("Free parameters for validation:", free_param_names)
    # choose one redshift
    zs = mockdata_validation.z
    Nz = len(zs)
    for iz_choice in [0]:
    # np.arange(Nz):
        iz_traintest = np.where(train_test_info['z']==zs[iz_choice])[0][0]

        # read the likelihood params from earlier, adding more detail this time
        like_params_validation = []
        for par in like_params:
            min_training = train_test_info[par.name+"_min"][iz_traintest]
            max_training = train_test_info[par.name+"_max"][iz_traintest]
            like_params_validation.append(LikelihoodParameter(
                name=par.name,
                value=par.value[iz_choice],
                ini_value=np.random.uniform(min_training, max_training), # random initial value within training range
                min_value=min_training,
                max_value=max_training
            ))
        for par in like_params_validation:
            print("Likelihood parameters:", par.name, par.value, "min", par.min_value, "max", par.max_value)

        
        # make iz_choice into an array to be compatible with likelihood object
        iz_choice_arr = np.array([iz_choice])
        like_validation = Likelihood(mockdata_validation, theory_AA, free_param_names=free_param_names, iz_choice=iz_choice_arr, like_params=like_params_validation, verbose=False)
        like_validation.plot_px(iz_choice_arr, like_params_validation, multiply_by_k=False, ylim2=[-.05,.05], ylim=[-.005,0.25], every_other_theta=True, show=False, title=f"Redshift {zs[iz_choice]}", theorylabel=f'Theory: {forecast} sim', datalabel=f'Forecast at {forecast} sim', xlim=[0,.7], residual_to_theory=True)
        plt.savefig(cupixpath + f"/plots/validation_forecast_pipeline_{forecast}{forecast_str}_z{int(zs[iz_choice]*100):03d}_{noise_str}.png", bbox_inches='tight')
        plt.close()
        # minimize
        mini = IminuitMinimizer(like_validation, verbose=False)
        start = time.time()
        mini.minimize()
        end = time.time()
        print("Minimization took", end-start, "seconds")
        # plot the results
        true_vals={par.name:par.value for par in like_params_validation}
        # flip sign of bias if needed, for plotting
        if param_mode=='arinyo' and true_vals['bias']>0:
            true_vals['bias'] *= -1
        
        # save the fit results
        fit_results = mini.results_dict_2par()
        fit_results['minimization_time_seconds'] = end-start
        save_analysis_npz(fit_results, cupixpath + f"/data/fitter_results/validation_forecast_results_{free_param_names[0]}_{free_param_names[1]}_{forecast}{forecast_str}_{noise_str}_z{int(zs[iz_choice]*100):03d}.npz")

        # check the parameters
        print("Validation parameters were:")
        for p in like_params_validation:
            if p.name in free_param_names:
                print(p.name, "true value", p.value, "allowed to span", p.min_value, p.max_value, "and starting at", p.ini_value)
        
        index_par1 = [i for i, p in enumerate(like_params_validation) if p.name==free_param_names[0]][0]
        index_par2 = [i for i, p in enumerate(like_params_validation) if p.name==free_param_names[1]][0]

        mini.plot_ellipses(free_param_names[0], free_param_names[1], nsig=2, cube_values=False, true_vals=true_vals)
        
        plt.axvline(x=like_params_validation[index_par1].ini_value, color='orange', linestyle=':')
        plt.axhline(y=like_params_validation[index_par2].ini_value, color='orange', linestyle=':')
        plt.scatter(fit_results[free_param_names[0]], fit_results[free_param_names[1]], color='red', label='Fitted Value')
        print("Setting x limits for ", free_param_names[0], " to ", 0.9*train_test_info[free_param_names[0]+"_min"][iz_choice], 1.1*train_test_info[free_param_names[0]+"_max"][iz_choice])
        plt.xlim([0.9*train_test_info[free_param_names[0]+"_min"][iz_traintest], 1.1*train_test_info[free_param_names[0]+"_max"][iz_traintest]])
        plt.ylim([0.9*train_test_info[free_param_names[1]+"_min"][iz_traintest], 1.1*train_test_info[free_param_names[1]+"_max"][iz_traintest]])
        plt.text(like_params_validation[index_par1].ini_value, like_params_validation[index_par2].ini_value, 'Initial Value', color='orange')
        plt.savefig(cupixpath+f"/plots/validation_forecast_{free_param_names[0]}_{free_param_names[1]}_{forecast}{forecast_str}_{noise_str}_z{int(zs[iz_choice]*100):03d}.png", bbox_inches='tight')
        plt.close()