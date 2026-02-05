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

# --------- settings ---------
forecast = "random" # "central" or "random"
param_mode = "arinyo" # "cosmo_igm" or "arinyo"
add_noise = False
Nfree = 2 # number of free parameters to fit in validation
# "central" uses the central simulation in the training set
# "random" randomly selects points within the minimum and maximum range of each training parameter
# --------------------------

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
    pars = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
elif param_mode == "arinyo":
    pars = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']

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


# Part 1: generate the forecast
# We will pull the theta bins, k bins, and covariance and window matrices from an existing mock data file
mockdata_file = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-0/tru_cont/binned_out_px-zbins_2-thetabins_18.hdf5"

# prepare to generate and write fake data
# cupixpath = cupix.__path__[0].rsplit('/', 1)[0] # use this line if trying to write to the cupix package
cupixpath = "/pscratch/sd/m/mlokken/desi-lya/px/"
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

savestr = f"{filepath}/forecast_ff{forecast}{forecast_str}_{Path(mockdata_file).stem}_{noise_str}.hdf5"


mockdata = DESI_DR2(mockdata_file)

# choose some redshifts from the training data
zs = []
sim_dict_central =  Archive3D.get_testing_data("mpg_central")
for sim_z in sim_dict_central:
    if (sim_z['z'] < 2.7) & (sim_z['z'] > 2.2):
        zs.append(sim_z['z'])
zs = np.sort(np.array(zs))
Nz = len(zs)
# manually overwrite MockData zs with ForestFlow zs
mockdata.z = zs
training_data = Archive3D.training_data

# Load emulator
print("Loading emulator.")
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=zs, camb_kmax_Mpc=1000)
ffemu = FF_emulator(zs, fid_cosmo, cc)
ffemu.kp_Mpc = 1 # set pivot point
theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(zs)
theory_AA.emulator = ffemu

# find the min, max of all parameters
par_minmax    = {par:np.zeros((Nz, 2)) for par in pars} # [min, max] for each par at each z

for iz,zz in enumerate(zs):
    # collect all the training values at this redshift
    training_vals_iz = {par:[] for par in pars}
    for i in range(len(training_data)):
        if param_mode=="cosmo_igm":
            parameter_data = training_data[i]
        elif param_mode=="arinyo":
            parameter_data = training_data[i]['Arinyo']
        if abs(training_data[i]['z']-zz)<0.01: # find the matching redshift
            for par in parameter_data.keys():
                if par in pars:
                    training_vals_iz[par].append(parameter_data[par])
    print(len(training_vals_iz[pars[0]]), "training samples found at z=", zz)
    for par in pars:
        par_minmax[par][iz][0] = np.amin(np.asarray(training_vals_iz[par]))
        par_minmax[par][iz][1] = np.amax(np.asarray(training_vals_iz[par]))
        print(f"Parameter {par} at z={zz}: min={par_minmax[par][iz][0]}, max={par_minmax[par][iz][1]}")
print("par_minmax is", par_minmax)
like_params = []
like_params_dict = {par:np.zeros(len(zs)) for par in pars}

for iz,zz in enumerate(zs):
    if forecast=="central":
        for sim_z in sim_dict_central:
            if abs(sim_z['z']-zz)<0.01: # find the matching redshift
                if param_mode=="cosmo_igm":
                    parameter_data = sim_z
                elif param_mode=="arinyo":
                    parameter_data = sim_z['Arinyo']
                for par in parameter_data.keys():
                    if par in pars:
                        like_params_dict[par][iz] = parameter_data[par]
    elif forecast=="random":
        # randomly select a value within the min and max range of each parameter at this redshift
        for par in pars:
            like_params_dict[par][iz] = rng.uniform(par_minmax[par][iz][0], par_minmax[par][iz][1])
            # print(np.random.uniform(par_minmax[par][iz][0], par_minmax[par][iz][1]))
            # print(np.random.uniform(par_minmax[par][iz][0], par_minmax[par][iz][1]))
            print("new parameter value for", par, like_params_dict[par][iz])

# create LikelihoodParameter objects
for par in like_params_dict:
    like_params.append(LikelihoodParameter(name=par, min_value=-1000, max_value=1000, value=like_params_dict[par])) # min/max values don't matter here
for par in like_params:
    print("Likelihood parameters:", par.name, par.value, par.min_value, par.max_value)

if os.path.exists(savestr):
    print("Forecast file already exists:", savestr, "Moving on to minimization.")
else:
    # initialize Likelihood
    like = Likelihood(mockdata, theory_AA, free_param_names=[], iz_choice=[0,1], like_params=like_params, verbose=False)
    # initialize fake data
    fakedata = FakeData(like)
    print("Saving to...", savestr)
    fakedata.write_to_file(savestr, add_noise=False)


# part 2: validate the pipeline by fitting to the fake data
# read in the fake data
mockdata_validation = DESI_DR2(savestr, kmax_cut_AA=1)
# get the free parameters
free_param_names = pars[:Nfree]
print("Free parameters for validation:", free_param_names)
# choose one redshift
zs = mockdata_validation.z
Nz = len(zs)
for iz_choice in [0]:
# np.arange(Nz):
    
    # read the likelihood params from earlier, adding more detail this time
    like_params_validation = []
    for par in like_params:
        print(iz_choice, par_minmax[par.name][iz_choice])
        min_training = par_minmax[par.name][iz_choice,0]
        max_training = par_minmax[par.name][iz_choice,1]
        like_params_validation.append(LikelihoodParameter(
            name=par.name,
            value=par.value[iz_choice],
            ini_value=np.random.uniform(min_training, max_training), # random initial value within training range
            min_value=min_training,
            max_value=max_training
        ))
    # check the parameters
    print("Validation parameters are:")
    for p in like_params_validation:
        print(p.name, "true value", p.value, "allowed to span", p.min_value, p.max_value, "and starting at", p.ini_value)
    
    # make iz_choice into an array to be compatible with likelihood object
    iz_choice = np.array([iz_choice])
    like_validation = Likelihood(mockdata_validation, theory_AA, free_param_names=free_param_names, iz_choice=iz_choice, like_params=like_params_validation, verbose=False)
    like_validation.plot_px(iz_choice, like_params_validation, multiply_by_k=False, ylim2=[-.05,.05], ylim=[-.005,0.25], every_other_theta=True, show=False, title=f"Redshift {zs[iz_choice][0]}", theorylabel=f'Theory: {forecast} sim', datalabel=f'Forecast at {forecast} sim', xlim=[0,.7], residual_to_theory=True)
    plt.savefig(cupixpath + f"/plots/validation_forecast_pipeline_{forecast}{forecast_str}_z{int(zs[iz_choice][0]*100):03d}_{noise_str}.png", bbox_inches='tight')
    plt.close()
    # minimize
    mini = IminuitMinimizer(like_validation, verbose=False)
    start = time.time()
    mini.minimize()
    end = time.time()
    print("Minimization took", end-start, "seconds")
    # plot the results
    true_vals={par.name:par.value[iz_choice] for par in like_params}
    # flip sign of bias if needed, for plotting
    if param_mode=='arinyo' and true_vals['bias']>0:
        true_vals['bias'] *= -1
    


    # save the fit results
    fit_results = mini.results_dict_2par()
    fit_results['minimization_time_seconds'] = end-start
    save_analysis_npz(fit_results, cupixpath + f"/data/fitter_results/validation_forecast_results_{forecast}{forecast_str}_{noise_str}_z{int(zs[iz_choice][0]*100):03d}.npz")

    index_par1 = [i for i, p in enumerate(like_params_validation) if p.name==free_param_names[0]][0]
    index_par2 = [i for i, p in enumerate(like_params_validation) if p.name==free_param_names[1]][0]
    mini.plot_ellipses(free_param_names[0], free_param_names[1], nsig=2, cube_values=False, true_vals=true_vals)
    plt.axvline(x=like_params_validation[index_par1].ini_value, color='orange', linestyle=':')
    plt.axhline(y=like_params_validation[index_par2].ini_value, color='orange', linestyle=':')
    plt.scatter(fit_results[free_param_names[0]], fit_results[free_param_names[1]], color='red', label='Fitted Value')
    plt.xlim([par_minmax[free_param_names[0]][iz_choice,0]-0.02, par_minmax[free_param_names[0]][iz_choice,1]+0.02])
    plt.ylim([par_minmax[free_param_names[1]][iz_choice,0]-0.02, par_minmax[free_param_names[1]][iz_choice,1]+0.02])
    plt.text(like_params_validation[index_par1].ini_value, like_params_validation[index_par2].ini_value, 'Initial Value', color='orange')
    plt.savefig(cupixpath+f"/plots/validation_forecast_{free_param_names[0]}_{free_param_names[1]}_{forecast}{forecast_str}_{noise_str}_z{int(zs[iz_choice][0]*100):03d}.png", bbox_inches='tight')
    plt.close()