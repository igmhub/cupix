# This script generates a small file with only the gadget minimum and maximum values for all possible parameters to vary,
# for easy access later instead of needing to load the full GadgetArchive3D every time


import numpy as np
import os
from forestflow.archive import GadgetArchive3D
import forestflow
import cupix
import pandas as pd

# define paths
cupixpath = cupix.__path__[0].rsplit('/', 1)[0] # use this line if trying to write to the cupix package
# cupixpath = "/pscratch/sd/m/mlokken/desi-lya/px/"
filepath = cupixpath + "/data/px_measurements/forecast/"

# if it does not yet exist, start by saving a file with all the necessary information about the Gadget simulations:
# z
# per z, the min and max value of each parameter in the training set
# per z, the value of each parameter in the central simulation
igm_pars = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc', 'T0']
arinyo_pars = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']
gadget_short_info_file = cupixpath + '/data/emulator/ff_training_info.csv'
if not os.path.exists(gadget_short_info_file):
    # Figure out the ForestFlow training central simulation
    path_program = os.path.dirname(forestflow.__path__[0]) + '/'
    path_program
    folder_lya_data = path_program + "/data/best_arinyo/"
    print("Loading archive.")
    Archive3D = GadgetArchive3D(
        base_folder=path_program[:-1]
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
            if par in sim_z['Arinyo_min']:
                dict_save[par+"_central"].append(sim_z['Arinyo_min'][par])
            elif par in sim_z:
                dict_save[par+"_central"].append(sim_z[par])
            else:
                print("Parameter", par, "not found in central simulation at z=", sim_z['z'])
            # now find min, max values from training data
            all_training_values_iz = []
            for i in range(len(training_data)):
                if abs(training_data[i]['z']-sim_z['z'])<0.01:
                    if par in training_data[i]['Arinyo_min']:
                        all_training_values_iz.append(training_data[i]['Arinyo_min'][par])
                    elif par in training_data[i]:
                        all_training_values_iz.append(training_data[i][par])
            dict_save[par+"_min"].append(np.amin(np.asarray(all_training_values_iz)))
            dict_save[par+"_max"].append(np.amax(np.asarray(all_training_values_iz)))
    train_test_info = pd.DataFrame(dict_save)
    train_test_info.to_csv(gadget_short_info_file, index=False)
else:
    print("Gadget simulation info file already exists at", gadget_short_info_file)
    train_test_info = pd.read_csv(gadget_short_info_file)