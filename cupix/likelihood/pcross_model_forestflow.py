""" This module provides a set of functions to get a prediction of Px using ForestFlow"""

import numpy as np
import os, sys
sys.path.insert(0, os.environ['HOME']+'/Software/LyaP3D')
# from tools import SPEED_LIGHT, LAMBDA_LYA

# For forestflow
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
# convert angular coordinates to physical
from astropy.cosmology import FlatLambdaCDM
from forestflow import pcross
import forestflow
from lace.cosmo.thermal_broadening import thermal_broadening_kms
from lace.cosmo import camb_cosmo


def load_emulator(Nrealizations):
    """ This function loads the emulator and doesn't require any input """

    path_program = forestflow.__path__[0][:-10]

    # LOAD P3D ARCHIVE
    folder_lya_data = path_program + "/data/best_arinyo/"

    Archive3D = GadgetArchive3D(
        base_folder=path_program[:-1],
        folder_data=folder_lya_data,
        force_recompute_plin=False,
        average="both",
    )

    # Load emulator
    training_type = "Arinyo_min"
    model_path=path_program+"/data/emulator_models/mpg_hypercube.pt"

    emulator = P3DEmulator(
        Archive3D.training_data,
        Archive3D.emu_params,
        nepochs=300,
        lr=0.001,  # 0.005
        batch_size=20,
        step_size=200,
        gamma=0.1,
        weight_decay=0,
        adamw=True,
        nLayers_inn=12,  # 15
        Archive=Archive3D,
        training_type=training_type,
        model_path=model_path,
        Nrealizations=Nrealizations,
    )

    return emulator


def load_arinyo(z, cosmo_param_dict):
    """ This function reads redshift z and cosmo paremeters dictionary and loads the Arinyo model
    Arguments:
    ----------
    z: Float or array of floats
    Redshift.

    cosmo_param_dict: Dictionary
    It should include 'H0', 'omch2', 'ombh2', 'mnu', 'omk', 'As', 'ns', 'nrun', 'w'.

    Return:
    -------
    arinyo: instance of the ArinyoModel class
    """

    # Getting sim_cosmo from camb_cosmo based on cosmo input
    sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo_param_dict)

    # Initializing the Arinyo model with the emulated values of bias and beta heyyyy
    camb_results = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000) # set default cosmo
    arinyo = ArinyoModel(cosmo=sim_cosmo, camb_results=camb_results, zs=z, camb_kmax_Mpc=1000) # set model

    return arinyo


def get_forestflow_params(z, igm_param_dict, dkms_dMpc_zs, cosmo_param_dict=None, delta_np_dict=None):
    """ This function: 
        - Reads the IGM parameters from igm_param_dict
        - Reads the cosmo parameters from cosmo_param_dict or amplitude and slope from delta_np_dict
        - Converts IGM parameters T0 and lambda_pressure to emulator parameters sigT_Mpc and kF_Mpc respectively
        - Defines emualator parameters dictionary emu_params and linear power spectrum information dictionary
        - Returns them

    Arguments:
    ----------
    z: Float or array of floats
    Redshift.

    igm_param_dict: Dictionary 
    It should include 'T0' [K], 'gamma', 'mF', 'lambda_pressure' [kpc].

    cosmo_param_dict: Dictionary
    It should include 'H0', 'omch2', 'ombh2', 'mnu', 'omk', 'As', 'ns', 'nrun', 'w'.

    delta_np_dict: Dictionary
    It should include 'Delta2_p' and 'n_p'.

    Return:
    -------
    emu_params: Dictionary
    It includes the emulator parameters: 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc'.

    info_power: Dictionary
    If delta_np_dict is given: it includes redshift 'z', 'Delta2_p' and 'n_p', otherwise: it includes redshift 'z' and cosmo parameters 'cosmo'.
    """

    # Converting IGM parameters T0 and lambda_pressure to emulator parameters sigT_Mpc and kF_Mpc respectively
    sigT_kms = thermal_broadening_kms(igm_param_dict['T0'])
    sigT_Mpc = sigT_kms / dkms_dMpc_zs[0]
    kF_Mpc = 1 / (igm_param_dict['lambda_pressure'] / 1000)

    # Defining dictionaries
    emu_params = {
        "mF": igm_param_dict['mF'],
        "gamma": igm_param_dict['gamma'],
        "sigT_Mpc":sigT_Mpc,
        "kF_Mpc":kF_Mpc,
    }
    # check which type of param dict is given
    if cosmo_param_dict is not None and delta_np_dict is None:
        info_power = {
            "cosmo": cosmo_param_dict,
            "z": z,
        }
    elif cosmo_param_dict is None and delta_np_dict is not None:
        info_power = {
            "Delta2_p": delta_np_dict['Delta2_p'],
            "n_p": delta_np_dict['n_p'],
            "z": z,
        }
    else: # This means both dicts are given and it should not happen in general since this is fixed in the get_pcross_forestflow for now
        print('Warning: both delta_np_dict and cosmo_param_dict arguments are given, therefore Delta2_p and n_p will be fixed to the values given and will not be recomputed based on new cosmo')
        info_power = {
            "Delta2_p": delta_np_dict['Delta2_p'],
            "n_p": delta_np_dict['n_p'],
            "z": z,
        }
    return emu_params, info_power


def convert_kpar_to_forestflow_units(kpar, inout_unit, dAA_dMpc_zs=None, dkms_dMpc_zs=None):
    """ Converts kpar to Mpc, the only input unit accepted by forestflow """
    kpar_iMpc = []
    if inout_unit == 'AA':
        if dAA_dMpc_zs is not None:
            for dAA_dMpc in dAA_dMpc_zs:
                kpar_iMpc.append(kpar * dAA_dMpc)
        else:
            sys.exit('Must input dAA_dMpc_zs')
    elif inout_unit == 'kmps':
        if dkms_dMpc_zs is not None:
            for dkms_dMpc in dAA_dMpc_zs:
                kpar_iMpc.append(kpar * dkms_dMpc)
        else:
            sys.exit('Must input dkms_dMpc_zs') 
    else:
        sys.exit('Must input inout_unit: AA or kmps')
    
    return np.asarray(kpar_iMpc)


def convert_pcross_to_output_units(Px_pred_Mpc, inout_unit, dAA_dMpc_zs=None, dkms_dMpc_zs=None):
    # ML note: still need to update this to work with len(dAA_dMpc_zs)>1 
    """ Converts the output of forestflow from Mpc to the desired output unit """

    if inout_unit == 'AA':
        if dAA_dMpc_zs is not None:
            Px_pred_outunits = Px_pred_Mpc / dAA_dMpc_zs
        else:
            sys.exit('Must input dAA_dMpc_zs')
    elif inout_unit == 'kmps':
        if dkms_dMpc_zs is not None:
            Px_pred_outunits = Px_pred_Mpc / dkms_dMpc_zs
        else:
            sys.exit('Must input dkms_dMpc_zs') 
    else:
        sys.exit('Must input inout_unit: AA or kmps')

    return Px_pred_outunits


def convert_sepbins_to_foresflow_units(z, cosmo_param_dict, sepbins, sepbins_unit):
    """ Converts the transverse separation values from their input unit to Mpc as accepted by forestflow """

    H0 = cosmo_param_dict['H0']
    omch2 = cosmo_param_dict['omch2']
    ombh2 = cosmo_param_dict['ombh2']
    mnu = cosmo_param_dict['mnu']

    if sepbins_unit == 'deg':
        sepbins_Mpc = np.zeros(sepbins.shape)
        h = H0 / 100
        Om0 = (omch2 + ombh2) / (h**2)
        astropy_cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, m_nu=[mnu, 0, 0])
        for i, z_i in enumerate(z):
            sepbins_Mpc[i] = astropy_cosmo.comoving_transverse_distance(z_i) * np.deg2rad(sepbins[i])
    elif sepbins_unit == 'Mpc':
        sepbins_Mpc = sepbins
    else:
        sys.exit('Must input sepbins_unit: Mpc or deg')
    return sepbins_Mpc


def get_pcross_forestflow(kpar, sepbins, z, cosmo_param_dict, dAA_dMpc_zs, dkms_dMpc_zs, 
                          emulator, arinyo, inout_unit, sepbins_unit, delta_np_dict, mF, T0, gamma, lambda_pressure):
    """ This function predicts Px from forestflow given the IGM and cosmo parameters of the input.
    PS: the function is not yet adapted to vary cosmology.

    Arguments:
    ----------
    kpar: Array of floats
    Array of k_parallel at which we want to get a prediction.

    sepbins: Array of floats
    Array of sepbins at which we want to get a prediction.

    z: Float or array of floats
    Redshift.

    cosmo_param_dict: Dictionary
    Dictionary of cosmo parameters. It should include 'H0', 'omch2', 'ombh2', 'mnu', 'omk', 'As', 'ns', 'nrun', 'w'. 
    PS: they vary as function of redshift z, but are given as input to this function since cosmo is not to be varied for the moment.

    dAA_dMpc_zs, dkms_dMpc_zs: Float or array of floats
    Conversion factors.

    emulator: Emulator already loaded using load_emulator() function.

    arinyo: Loaded using load_arinyo() function. It must be given as input as long as the cosmo is not to be varied for now.

    inout_unit: String, default: 'AA', options: 'kmps'
    Units of input kpar that must be given in terms of the output units we want, and the output will be given in that same unit.

    sepbins_unit: Sting, default: 'deg', options: 'Mpc'
    Units of separation values at which we want to get the prediction.

    Delta2_p, n_p: Floats
    Amplitude and slope of the linear matter power spectrum precomputed from fixed cosmo for now.

    mF: Float or array of floats []
    Mean transmitted flux fraction. It is just an array if z is an array.

    T0: Float or array of floats
    Amplitude of the temperature density relation T = T0 * delta_b**(gamma - 1). It is just an array if z is an array.

    gamma: Float or array of floats
    Slope of the temperature density relation T = T0 * delta_b**(gamma - 1). It is just an array if z is an array.

    lambda_pressure: Float or array of floats
    Pressure smoothing scale (Jeans smoothing): The scale where pressure overcomes gravity at small scales -> smoothing of fluctuations.

    Return:
    -------
    Px_pred_output_units: 
    
    """

    # Code won't work if kpar has a zero
    if kpar[0] == 0:
        sys.exit('kpar array must not have a zero')

    # Conversions made at the beginning so that if the units are wrong, the code exits from here
    kpar_iMpc = convert_kpar_to_forestflow_units(kpar, inout_unit, dAA_dMpc_zs=dAA_dMpc_zs, dkms_dMpc_zs=dkms_dMpc_zs)
    sepbins_Mpc = convert_sepbins_to_foresflow_units(z, cosmo_param_dict, sepbins, sepbins_unit)

    # Creating dictionary of IGM parameters that will then be transformed into emu_params
    # PS: cosmo parameters are not given explicitely since they're not varied for now
    
    igm_param_dict = {
        "mF": mF,
        "gamma": gamma,
        "lambda_pressure": lambda_pressure,
        "T0": T0,
    }

    # Getting forestflow parameters
    emu_params, info_power = get_forestflow_params(z=z, igm_param_dict=igm_param_dict, dkms_dMpc_zs=dkms_dMpc_zs, cosmo_param_dict=None, delta_np_dict=delta_np_dict)
    # print('Input parameters given to the emulator are:', emu_params, info_power)
    
    # merge the two
    emu_params.update(info_power)

    # prepare an Arinyo dictionary
    arinyo_coeffs = []

    # Evaluating emulator at the input parameters values
    for i, z_i in enumerate(z):
        emu_params_i = {}
        for key in emu_params.keys():
            emu_params_i[key] = emu_params[key][i]
        arinyo_coeffs_i = emulator.predict_Arinyos(
        emu_params=emu_params_i)    
        # turn into a dictionary
        arinyo_coeffs_i = {"bias": arinyo_coeffs_i[0], "beta": arinyo_coeffs_i[1], "q1": arinyo_coeffs_i[2],
                        "kvav": arinyo_coeffs_i[3], "av": arinyo_coeffs_i[4], "bv": arinyo_coeffs_i[5],
                        "kp": arinyo_coeffs_i[6], "q2": arinyo_coeffs_i[7]}
        arinyo_coeffs.append(arinyo_coeffs_i)
    # Predict Px
    try:
        # use the commented lines if you want to do a detailed version that changes settings
        # rperp_pred, Px_pred_Mpc = pcross.Px_Mpc_detailed(kpar_iMpc,
        # arinyo.P3D_Mpc,
        # info_power['z'],
        # rperp_choice=sepbins_Mpc,
        # P3D_mode='pol',
        # min_kperp=10**-3,
        # max_kperp=10**2.9,
        # nkperp=2**12,
        # **{"pp":arinyo_coeffs})
        
        Px_pred = []
        for i, z_i in enumerate(z):
            rperp_pred_i, Px_pred_Mpc_i = pcross.Px_Mpc(kpar_iMpc[i], arinyo.P3D_Mpc, z_i, rperp_choice = sepbins_Mpc[i], **{"pp":arinyo_coeffs[i]})
            # Convert Px_pred_Mpc to Px_pred_output that has inout_units
            Px_pred_output_i = convert_pcross_to_output_units(Px_pred_Mpc_i, inout_unit, dAA_dMpc_zs=dAA_dMpc_zs[i], dkms_dMpc_zs=dkms_dMpc_zs[i])
            
            # Return transpose to match Px_data shapes
            Px_pred_output_transpose = Px_pred_output_i.T

            if np.any(np.isnan(Px_pred_output_transpose)):
                print("NaN encountered in Px prediction!")
                print(Px_pred_output_transpose)
            Px_pred.append(Px_pred_output_transpose)
        return np.asarray(Px_pred)
    
    except:
        print('Input parameters from minuit variation:', igm_param_dict, cosmo_param_dict, delta_np_dict)
        print('Input parameters given to the emulator are:', emu_params, info_power)
        print('Input parameters given to the arinyo model are:', arinyo_coeffs)
        print('Problematic model so None is returned for Px prediction')
        return None