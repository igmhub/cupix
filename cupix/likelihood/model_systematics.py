import numpy as np

from cupix.nuisance.resolution_model import Resolution_Model
from cupix.nuisance.resolution_model_chunks import Resolution_Model_Chunks


class Systematics(object):
    """Contains all IGM models"""

    def __init__(
        self, free_param_names=None, resolution_model=None, pars_syst=None
    ):
        if "Gauss_priors" in pars_syst:
            Gauss_priors = pars_syst["Gauss_priors"]
        else:
            Gauss_priors = None
        # setup Resolution model
        if resolution_model:
            self.resolution_model = resolution_model
        else:
            if pars_syst["res_model_type"] == "pivot":
                self.resolution_model = Resolution_Model(
                    free_param_names=free_param_names,
                    fid_R_coeff=pars_syst["R_coeff"],
                    Gauss_priors=Gauss_priors,
                )
            elif pars_syst["res_model_type"] == "chunks":
                self.resolution_model = Resolution_Model_Chunks(
                    free_param_names=free_param_names,
                    Gauss_priors=Gauss_priors,
                )
            else:
                raise ValueError(
                    "resolution_model_type must be 'pivot' or 'chunks'"
                )

    # def get_dict_cont(self):
    #     dict_out = {}

    #     if self.args.fid_syst["res_model_type"] == "pivot":
    #         for ii in range(len(self.args.fid_syst["R_coeff"])):
    #             dict_out["R_coeff_" + str(ii)] = self.args.fid_syst["R_coeff"][
    #                 -1 - ii
    #             ]
    #     else:
    #         for ii in range(self.resolution_model.get_Nparam()):
    #             dict_out["R_coeff_" + str(ii)] = 0

    #     return dict_out

    def get_contamination(self, z, k_kms, like_params=[]):
        # include multiplicative resolution correction
        cont_resolution = self.resolution_model.get_contamination(
            z=z, k_kms=k_kms, like_params=like_params
        )

        # print("cont_resolution", cont_resolution)

        return cont_resolution
