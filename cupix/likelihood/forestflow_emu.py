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
        ):

        self.emu_params = [
            "Delta2_p",
            "n_p",
            "mF",
            "sigT_Mpc",
            "gamma",
            "kF_Mpc"
        ]
        self.z = z
        self.cosmo_param_dict = cosmo_param_dict
        self.camb_cosmo_results = camb_cosmo_results
        self.emulator_label = "forestflow_emu"
        self.kp_Mpc = kp_Mpc
        self.kmax_Mpc = 5 # from Forestflow paper plots, could revisit
        
        self._load_emu()
        self._load_arinyo()

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


    def emulate_P3D_params(self, emu_call):
        """ This function predicts P3D parameters (the 'arinyo coefficients') from forestflow given the IGM and cosmo parameters of the input.
        Arguments:
        ----------
        emu_call: Dictionary with keys as the names of the parameters and values as the values of the parameters.

        Return:
        -------
        arinyo_coeffs: Dictionary with keys as the names of the Arinyo coefficients and values as the values of the coefficients.
        
        """
        
        # make sure that emu_call has a value for every z
        Nz = len(self.z)
        
        for key in emu_call.keys():
            assert len(emu_call[key]) == Nz, f"Parameter {key} has {len(emu_call[key])} values but should have {Nz} values for each redshift z."
        
        # prepare an Arinyo dictionary
        Nz = len(self.z)
        arinyo_coeffs = {}
        for key in ["bias",
            "beta",
            "q1",
            "kvav",
            "av",
            "bv",
            "kp",
            "q2"
        ]:
            # special case for optional parameter "q2"
            if key == "q2" and "q2" not in emu_call.keys():
                continue
            arinyo_coeffs[key] = np.zeros(Nz)
        for iz in range(Nz):
            emu_call_iz = {} # make a dictionary just for this z
            for key in emu_call.keys():
                if key in self.emu_params:
                    emu_call_iz[key] = emu_call[key][iz]
                else:
                    print(f"Warning: {key} is not a valid emu parameter. It will not be used in the emulation.")
            # make sure emu_call contains all the required parameters
            for key in self.emu_params:
                if key not in emu_call_iz:
                    raise ValueError(f"Parameter {key} is missing from emu_call. It is required for the emulation.")
            print("Trying to predict arinyo params with emu_call", emu_call_iz)
            arinyo_coeffs_iz = self.emu.predict_Arinyos(
            emu_params=emu_call_iz)
            
            # turn into a dictionary
            ia = 0
            for key in arinyo_coeffs.keys():
                arinyo_coeffs[key][iz] = arinyo_coeffs_iz[ia]
                ia += 1
        print("got arinyo coeffs", arinyo_coeffs)

        for key in emu_call.keys():
            if key in ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]:
                # If any of the keys are in the emu_call, overwrite the emulated values
                arinyo_coeffs[key] = emu_call[key]
        
        return arinyo_coeffs
        