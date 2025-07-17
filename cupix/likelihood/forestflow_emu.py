""" This module provides a set of functions to get a prediction of Px using ForestFlow"""

import numpy as np
import sys

# For forestflow
import forestflow
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
from forestflow import pcross

class FF_emulator():
    def __init__(
            self,
            z,
            cosmo_param_dict,
            camb_cosmo_results,
            kp_Mpc=0.7,
            match_lyacolore=False
        ):

        self.emu_params = [
            "Delta2_p",
            "n_p",
            "mF",
            "sigT_Mpc",
            "gamma",
            "kF_Mpc",
        ]
        self.z = z
        self.cosmo_param_dict = cosmo_param_dict
        self.camb_cosmo_results = camb_cosmo_results
        self.emulator_label = "forestflow_emu"
        self.kp_Mpc = kp_Mpc
        self.kmax_Mpc = 5 # from Forestflow paper plots, could revisit
        self._load_emu()
        self._load_arinyo()
        if match_lyacolore:
            self.match_lyacolore = True
        else:
            self.match_lyacolore = False

    def _load_emu(self, Nrealizations=1000):
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

        self.emu = emulator


    def _load_arinyo(self):
        """ This function reads redshift z and cosmo paremeters dictionary and loads the Arinyo model
        Arguments:
        ----------
        z: Float or array of floats
        Redshift.

        input_cosmo: Cosmology dictionary (like what is returned by lac.camb_cosmo.get_cosmology_from_dictionary(cosmo_param_dict))

        camb_cosmo_results: Camb Results object, returned by lace.camb_cosmo.get_camb_results
        Return:
        -------
        arinyo: instance of the ArinyoModel class
        """

        arinyo = ArinyoModel(cosmo=self.cosmo_param_dict, camb_results=self.camb_cosmo_results, zs=self.z, camb_kmax_Mpc=1000) # set model

        self.arinyo = arinyo


    def emulate_px_Mpc(self, emu_call, k_Mpc, theta_Mpc):
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

        # need to either load emulator or have it pre-loaded (e.g. as a class)

        # Code won't work if kpar has a zero
        if 0 in  k_Mpc:
            sys.exit('kpar array must not have a zero')
        # won't work if kpar is an array of single element
        if len(k_Mpc)==1:
            k_Mpc = k_Mpc[0]
        
        # prepare an Arinyo dictionary
        arinyo_coeffs = []

        Nz = len(self.z)
        if Nz>1:
            for i in range(Nz):
                
                emu_params_i = {}
                for key in emu_call.keys():
                    emu_params_i[key] = emu_call[key][i]
                arinyo_coeffs_i = self.emu.predict_Arinyos(
                emu_params=emu_params_i)    
                # turn into a dictionary
                arinyo_coeffs_i = {"bias": arinyo_coeffs_i[0], "beta": arinyo_coeffs_i[1], "q1": arinyo_coeffs_i[2],
                                "kvav": arinyo_coeffs_i[3], "av": arinyo_coeffs_i[4], "bv": arinyo_coeffs_i[5],
                                "kp": arinyo_coeffs_i[6], "q2": arinyo_coeffs_i[7]}
                arinyo_coeffs.append(arinyo_coeffs_i)
        else:
            for key in emu_call.keys():
                if type(emu_call[key]) is np.ndarray:
                    emu_call[key] = emu_call[key][0] # turn arrays into floats
            arinyo_coeffs = self.emu.predict_Arinyos(
                emu_params=emu_call)
            if self.match_lyacolore:
                print("Enforcing Lyacolore values for arinyo parameters")
                arinyo_coeffs[0] = -0.115 # 0.127 * ((1+self.z)/(1+2.3))**2.9 # bias
                arinyo_coeffs[1] = 1.55 # beta
                arinyo_coeffs[2] = 0.1112 # q1
                arinyo_coeffs[3] = 0.0001**0.2694 # kvav
                arinyo_coeffs[4] = 0.2694 # av
                arinyo_coeffs[5] = .0002 # bv
                arinyo_coeffs[6] = .5740 # kp # previously 0.25
                # ### values to 4 Mpc/h
                # arinyo_coeffs[2] = 0.0 # q1
                # arinyo_coeffs[3] = 0.3**0.3 # kvav
                # arinyo_coeffs[4] = 0.3 # av
                # arinyo_coeffs[5] = 1.0 # bv
                # arinyo_coeffs[6] = 472e-3 # kp # previously 0.25
            # turn into a dictionary
            arinyo_coeffs = {"bias": arinyo_coeffs[0], "beta": arinyo_coeffs[1], "q1": arinyo_coeffs[2],
                            "kvav": arinyo_coeffs[3], "av": arinyo_coeffs[4], "bv": arinyo_coeffs[5],
                            "kp": arinyo_coeffs[6], "q2": arinyo_coeffs[7]}
        # Predict Px
        # try:
            # print('Input parameters were:', emu_call)
            # print('Input parameters given to the arinyo model are:', arinyo_coeffs)
        try:
            if Nz>1:
                Px_pred = []
                for i, z_i in enumerate(self.z):
                    # _, Px_pred_Mpc_i = self.arinyo.Px_Mpc(z_i, k_Mpc[i], arinyo_coeffs[i], **{'rperp_choice':theta_Mpc[i]})
                    _, Px_pred_Mpc_i = pcross.Px_Mpc(k_Mpc[i], self.arinyo.P3D_Mpc, self.z[i], rperp_choice=theta_Mpc[i],**{"pp":arinyo_coeffs[i]})
                    # Return transpose to match Px_data shapes
                    Px_pred_output_transpose = Px_pred_Mpc_i.T
                    if np.any(np.isnan(Px_pred_output_transpose)):
                        print("NaN encountered in Px prediction!")
                    Px_pred.append(Px_pred_output_transpose)
                return np.asarray(Px_pred)
            else:
                # _, Px_pred_Mpc = self.arinyo.Px_Mpc(self.z, k_Mpc, arinyo_coeffs, **{'rperp_choice':theta_Mpc})
                _, Px_pred_Mpc = pcross.Px_Mpc(k_Mpc, self.arinyo.P3D_Mpc, self.z, rperp_choice=theta_Mpc,**{"pp":arinyo_coeffs})
                return np.transpose(Px_pred_Mpc,(1, 2, 0))
        except:
            print('Problematic model so None is returned for Px prediction')
            print('Input parameters were:', emu_call)
            print('Input parameters given to the arinyo model are:', arinyo_coeffs)
            return None
        