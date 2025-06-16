import numpy as np

from cup1d.nuisance import (
    metal_model,
    hcd_model_McDonald2005,
    hcd_model_Rogers2017,
    hcd_model_new,
    SN_model,
    AGN_model,
)


class Contaminants(object):
    """Contains all IGM models"""

    def __init__(
        self,
        free_param_names=None,
        dict_models=None,
        hcd_model=None,
        sn_model=None,
        agn_model=None,
        metal_lines=None,
        fid_metals=None,
        fid_A_damp=None,
        fid_A_scale=None,
        fid_SN=None,
        fid_AGN=None,
        hcd_model_type=None,
        ic_correction=None,
        Gauss_priors=None,
    ):
        self.metal_lines = metal_lines
        self.fid_metals = fid_metals
        self.fid_A_damp = fid_A_damp
        self.fid_A_scale = fid_A_scale
        self.fid_SN = fid_SN
        self.fid_AGN = fid_AGN
        self.hcd_model_type = hcd_model_type
        self.ic_correction = ic_correction

        # setup metal models
        self.metal_models = {}

        for metal_line in self.metal_lines:
            create_model = True
            if dict_models is not None:
                if metal_line in dict_models:
                    self.metal_models[metal_line] = dict_models[metal_line]
                    create_model = False
            if create_model:
                self.metal_models[metal_line] = metal_model.MetalModel(
                    metal_label=metal_line,
                    free_param_names=free_param_names,
                    X_fid_value=self.fid_metals[metal_line + "_X"],
                    D_fid_value=self.fid_metals[metal_line + "_D"],
                    L_fid_value=self.fid_metals[metal_line + "_L"],
                    A_fid_value=self.fid_metals[metal_line + "_A"],
                    Gauss_priors=Gauss_priors,
                )

        # setup HCD model
        if hcd_model:
            self.hcd_model = hcd_model
        else:
            if self.hcd_model_type == "Rogers2017":
                self.hcd_model = hcd_model_Rogers2017.HCD_Model_Rogers2017(
                    free_param_names=free_param_names,
                    fid_A_damp=self.fid_A_damp,
                    fid_A_scale=self.fid_A_scale,
                )
            elif self.hcd_model_type == "McDonald2005":
                self.hcd_model = hcd_model_McDonald2005.HCD_Model_McDonald2005(
                    free_param_names=free_param_names,
                    fid_A_damp=self.fid_A_damp,
                )
            elif self.hcd_model_type == "new":
                self.hcd_model = hcd_model_new.HCD_Model_new(
                    free_param_names=free_param_names,
                    fid_A_damp=self.fid_A_damp,
                    fid_A_scale=self.fid_A_scale,
                    Gauss_priors=Gauss_priors,
                )
            else:
                raise ValueError(
                    "hcd_model_type must be one of 'Rogers2017', 'McDonald2005', or 'new'"
                )

        # setup SN model
        if sn_model:
            self.sn_model = sn_model
        else:
            self.sn_model = SN_model.SN_Model(
                free_param_names=free_param_names,
                fid_value=self.fid_SN,
            )

        # setup AGN model
        if agn_model:
            self.agn_model = sn_model
        else:
            self.agn_model = AGN_model.AGN_Model(
                free_param_names=free_param_names,
                fid_value=self.fid_AGN,
            )

    def get_dict_cont(self):
        dict_out = {}

        # maximum number of parameters
        for ii in range(2):
            for metal_line in self.metal_lines:
                flag = "ln_x_" + metal_line + "_" + str(ii)
                dict_out[flag] = self.fid_metals[metal_line + "_X"][-1 - ii]
                flag = "d_" + metal_line + "_" + str(ii)
                dict_out[flag] = self.fid_metals[metal_line + "_D"][-1 - ii]
                flag = "l_" + metal_line + "_" + str(ii)
                dict_out[flag] = self.fid_metals[metal_line + "_L"][-1 - ii]
                flag = "a_" + metal_line + "_" + str(ii)
                dict_out[flag] = self.fid_metals[metal_line + "_A"][-1 - ii]
            dict_out["ln_A_damp_" + str(ii)] = self.fid_A_damp[-1 - ii]
            dict_out["ln_A_scale_" + str(ii)] = self.fid_A_scale[-1 - ii]
            dict_out["ln_SN_" + str(ii)] = self.fid_SN[-1 - ii]
            dict_out["ln_AGN_" + str(ii)] = self.fid_AGN[-1 - ii]
        dict_out["ic_correction"] = self.ic_correction

        return dict_out

    def get_contamination(self, z, k_kms, mF, M_of_z, like_params=[]):
        # include multiplicative metal contamination
        if len(z) == 1:
            cont_metals = np.ones_like(k_kms)
        else:
            cont_metals = []
            for iz in range(len(z)):
                cont_metals.append(np.ones_like(k_kms[iz]))

        for model_name in self.metal_models:
            cont = self.metal_models[model_name].get_contamination(
                z=z,
                k_kms=k_kms,
                mF=mF,
                like_params=like_params,
            )
            if len(z) == 1:
                cont_metals *= cont
            else:
                for iz in range(len(z)):
                    if type(cont) != int:
                        cont_metals[iz] *= cont[iz]

        # include HCD contamination
        cont_HCD = self.hcd_model.get_contamination(
            z=z,
            k_kms=k_kms,
            like_params=like_params,
        )

        # include SN contamination
        if len(z) != 1:
            k_Mpc = []
            for iz in range(len(z)):
                k_Mpc.append(k_kms[iz] * M_of_z[iz])
        else:
            k_Mpc = [k_kms[0] * M_of_z[0]]
        cont_SN = self.sn_model.get_contamination(
            z=z,
            k_Mpc=k_Mpc,
            like_params=like_params,
        )

        # include AGN contamination
        cont_AGN = self.agn_model.get_contamination(
            z=z,
            k_kms=k_kms,
            like_params=like_params,
        )
        if np.any(cont_AGN < 0):
            return None

        if self.ic_correction:
            IC_corr = ref_nyx_ic_correction(k_kms, z)
        else:
            IC_corr = 1

        if len(z) == 1:
            cont_total = cont_metals * cont_HCD * cont_SN * cont_AGN * IC_corr
        else:
            cont_total = []
            if type(cont_metals) == int:
                _cont_metals = np.ones_like(z)
            else:
                _cont_metals = cont_metals

            if type(cont_HCD) == int:
                _cont_HCD = np.ones_like(z)
            else:
                _cont_HCD = cont_HCD

            if type(cont_SN) == int:
                _cont_SN = np.ones_like(z)
            else:
                _cont_SN = cont_SN

            if type(cont_AGN) == int:
                _cont_AGN = np.ones_like(z)
            else:
                _cont_AGN = cont_AGN

            if type(IC_corr) == int:
                _IC_corr = np.ones_like(z)
            else:
                _IC_corr = IC_corr

            for iz in range(len(z)):
                cont_total.append(
                    _cont_metals[iz]
                    * _cont_HCD[iz]
                    * _cont_SN[iz]
                    * _cont_AGN[iz]
                    * _IC_corr[iz]
                )

        return cont_total


def ref_nyx_ic_correction(k_kms, z):
    # This is the function fitted from the comparison of two Nyx runs,
    # one with 2lpt (single fluid) IC and the other one with monofonic (2 fluid)
    # - The high k points and z evolution are well determined
    # - Low k term: quite uncertain, due to cosmic variance
    ic_corr_z = np.array([0.15261529, -2.30600644, 2.61877894])
    ic_corr_k = 0.003669741766936781
    if len(z) == 1:
        ancorIC = (ic_corr_z[0] * z**2 + ic_corr_z[1] * z + ic_corr_z[2]) * (
            1 - np.exp(-k_kms / ic_corr_k)
        )
        corICs = 1 / (1 - ancorIC / 100)
    else:
        corICs = []
        for iz in range(len(z)):
            ancorIC = (
                ic_corr_z[0] * z[iz] ** 2 + ic_corr_z[1] * z[iz] + ic_corr_z[2]
            ) * (1 - np.exp(-k_kms[iz] / ic_corr_k))
            corICs.append(1 / (1 - ancorIC / 100))

    return corICs


# def nyx_ic_correction(k_kms, z):
#     # This is the function fitted from the comparison of two Nyx runs,
#     # one with 2lpt (single fluid) IC and the other one with monofonic (2 fluid)
#     # - The high k points and z evolution are well determined
#     # - Low k term: quite uncertain, due to cosmic variance
#     coeff0 = np.poly1d(np.array([0.00067032, -0.00626953, 0.0073908]))
#     coeff1 = np.poly1d(np.array([0.00767315, -0.04693207, 0.07151469]))
#     cfit = np.zeros(2)
#     cfit[0] = coeff0(z)
#     cfit[1] = coeff1(z)

#     rfit = np.poly1d(cfit)
#     # multiplicative correction
#     ic_corr = 10 ** rfit(np.log10(k_kms))

#     return ic_corr


# def nuisance_nyx_ic_correction(P0, k, z, Aic, Bic):
#     ic_corr_k = 0.003669741766936781
#     correction = (Aic + Bic * (z - 3)) * (1 - np.exp(-k / ic_corr_k))  # in %
#     return P0 / (1 - 0.01 * correction)


# def prior_nyx_ic_correction():
#     # The central values are the result of a 1st order polynomial fit from the
#     # 2nd order function given in _ref_nyx_ic_correction(P0, k, z)
#     return {"Aic": (-2.9, 1.0), "Bic": (-1.4, 0.5)}
