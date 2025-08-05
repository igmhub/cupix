import numpy as np

from cupix.nuisance import (
    metal_model_class,
    metal_metal_model_class,
    hcd_model_McDonald2005,
    hcd_model_Rogers2017,
    hcd_model_class,
    hcd_model_rogers_class,
    si_mult,
    si_add,
    SN_model,
    AGN_model,
)


class Contaminants(object):
    """Contains all IGM models"""

    def __init__(
        self,
        free_param_names=None,
        metal_models=None,
        hcd_model=None,
        sn_model=None,
        agn_model=None,
        pars_cont=None,
        ic_correction=None,
    ):
        self.pars_cont = pars_cont
        self.ic_correction = ic_correction
        # setup metal models
        self.metal_models = {}
        if "flat_priors" in pars_cont:
            flat_priors = pars_cont["flat_priors"]
        else:
            flat_priors = None
        if "Gauss_priors" in pars_cont:
            Gauss_priors = pars_cont["Gauss_priors"]
        else:
            Gauss_priors = None

        if "z_max" in pars_cont:
            z_max = pars_cont["z_max"]
        else:
            z_max = None

        prop_coeffs = {}
        fid_vals = {}
        for key in pars_cont:
            fid_vals[key] = pars_cont[key]
            for key2 in ["otype", "ztype", "znodes"]:
                key3 = key + "_" + key2
                if key3 in pars_cont:
                    prop_coeffs[key3] = pars_cont[key3]
                else:
                    if key3.endswith("otype"):
                        prop_coeffs[key3] = "exp"
                    elif key3.endswith("ztype"):
                        prop_coeffs[key3] = "pivot"

        # if joint_model:
        self.metal_add = ["Si_add"] #"CIVa_CIVb", "MgIIa_MgIIb", 

        # setup metal models
        key = "Si_mult"
        try:
            print("Models already set up")
            self.metal_models[key] = metal_models[key]
        except:
            print("Setting up Si_mult model")
            self.metal_models[key] = si_mult.SiMult(
                free_param_names=free_param_names,
                fid_vals=fid_vals,
                prop_coeffs=prop_coeffs,
                z_max=z_max,
                flat_priors=flat_priors,
                Gauss_priors=Gauss_priors,
            )

        key = "Si_add"
        try:
            self.metal_models[key] = metal_models[key]
        except:
            self.metal_models[key] = si_add.SiAdd(
                free_param_names=free_param_names,
                fid_vals=fid_vals,
                prop_coeffs=prop_coeffs,
                z_max=z_max,
                flat_priors=flat_priors,
                Gauss_priors=Gauss_priors,
            )

        # setup HCD model
        if hcd_model:
            self.hcd_model = hcd_model
        else:
            if pars_cont["hcd_model_type"] == "Rogers2017":
                self.hcd_model = hcd_model_Rogers2017.HCD_Model_Rogers2017(
                    free_param_names=free_param_names,
                    fid_A_damp=pars_cont["A_damp1"],
                    fid_A_scale=pars_cont["A_scale1"],
                )
            elif pars_cont["hcd_model_type"] == "McDonald2005":
                self.hcd_model = hcd_model_McDonald2005.HCD_Model_McDonald2005(
                    free_param_names=free_param_names,
                    fid_A_damp=pars_cont["A_damp1"],
                )
            elif pars_cont["hcd_model_type"] == "new":
                self.hcd_model = hcd_model_class.HCD_Model(
                    free_param_names=free_param_names,
                    fid_vals=pars_cont,
                    flat_priors=flat_priors,
                    Gauss_priors=Gauss_priors,
                )
            elif pars_cont["hcd_model_type"] == "new_rogers":
                self.hcd_model = hcd_model_rogers_class.HCD_Model_Rogers(
                    free_param_names=free_param_names,
                    fid_vals=fid_vals,
                    prop_coeffs=prop_coeffs,
                    flat_priors=flat_priors,
                    Gauss_priors=Gauss_priors,
                )
            else:
                raise ValueError(
                    "hcd_model_type must be one of 'Rogers2017', 'McDonald2005', 'new', or 'new2'"
                )

        # setup SN model
        if sn_model:
            self.sn_model = sn_model
        else:
            self.sn_model = SN_model.SN_Model(
                free_param_names=free_param_names,
                fid_value=pars_cont["SN"],
            )

        # setup AGN model
        if agn_model:
            self.agn_model = sn_model
        else:
            self.agn_model = AGN_model.AGN_Model(
                free_param_names=free_param_names,
                fid_value=pars_cont["AGN"],
            )

    # def get_dict_cont(self):
    #     dict_out = {}

    #     # maximum number of parameters
    #     for ii in range(2):
    #         for metal_line in self.args.metal_lines:
    #             flag = "f_" + metal_line + "_" + str(ii)
    #             dict_out[flag] = self.fid_metals[flag][-1 - ii]
    #             flag = "s_" + metal_line + "_" + str(ii)
    #             dict_out[flag] = self.fid_metals[flag][-1 - ii]
    #         dict_out["ln_A_damp_" + str(ii)] = self.fid_A_damp[-1 - ii]
    #         dict_out["ln_A_scale_" + str(ii)] = self.fid_A_scale[-1 - ii]
    #         dict_out["ln_SN_" + str(ii)] = self.fid_SN[-1 - ii]
    #         dict_out["ln_AGN_" + str(ii)] = self.fid_AGN[-1 - ii]
    #     dict_out["ic_correction"] = self.args.ic_correction

    #     return dict_out

    def get_contamination(
        self, z, k_kms, mF, M_of_z, like_params=[], remove=None
    ):
        # include multiplicative metal contamination
        
        if len(z) == 1:
            cont_mul_metals = np.ones_like(k_kms)
            cont_add_metals = np.zeros_like(k_kms)
        else:
            cont_mul_metals = []
            cont_add_metals = []
            for iz in range(len(z)):
                cont_mul_metals.append(np.ones_like(k_kms[iz]))
                cont_add_metals.append(np.zeros_like(k_kms[iz]))
        print("midway model_contaminants.py")
        print(self.metal_models)
        for model_name in self.metal_models:
            cont = self.metal_models[model_name].get_contamination(
                z=z,
                k_kms=k_kms,
                mF=mF,
                like_params=like_params,
                remove=remove,
            )
            print("further model_contaminants.py")
            if len(z) == 1:
                if model_name in self.metal_add:
                    cont_add_metals += cont
                else:
                    cont_mul_metals *= cont
            else:
                for iz in range(len(z)):
                    if model_name in self.metal_add:
                        if type(cont) != int:
                            cont_add_metals[iz] += cont[iz]
                        else:
                            cont_add_metals[iz] += cont
                    else:
                        if type(cont) != int:
                            cont_mul_metals[iz] *= cont[iz]
                        else:
                            cont_mul_metals[iz] *= cont
        print("late model_contaminants.py")
        # include HCD contamination
        cont_HCD = self.hcd_model.get_contamination(
            z=z,
            k_kms=k_kms,
            like_params=like_params,
        )

        # include SN contamination
        # if len(z) != 1:
        #     k_Mpc = []
        #     for iz in range(len(z)):
        #         k_Mpc.append(k_kms[iz] * M_of_z[iz])
        # else:
        #     k_Mpc = [k_kms[0] * M_of_z[0]]
        # cont_SN = self.sn_model.get_contamination(
        #     z=z,
        #     k_Mpc=k_Mpc,
        #     like_params=like_params,
        # )
        cont_SN = 1

        # include AGN contamination
        # cont_AGN = self.agn_model.get_contamination(
        #     z=z,
        #     k_kms=k_kms,
        #     like_params=like_params,
        # )
        # if np.any(cont_AGN < 0):
        #     cont_AGN = 1
        cont_AGN = 1

        if self.ic_correction:
            IC_corr = ref_nyx_ic_correction(k_kms, z)
        else:
            IC_corr = 1

        if len(z) == 1:
            mult_cont_total = (
                cont_mul_metals * cont_HCD * cont_SN * cont_AGN * IC_corr
            )
            add_cont_total = cont_add_metals
        else:
            mult_cont_total = []
            add_cont_total = []
            if type(cont_mul_metals) == int:
                _cont_mul_metals = np.ones_like(z)
            else:
                _cont_mul_metals = cont_mul_metals

            if type(cont_add_metals) == int:
                _cont_add_metals = np.zeros_like(z)
            else:
                _cont_add_metals = cont_add_metals

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
                mult_cont_total.append(
                    _cont_mul_metals[iz]
                    * _cont_HCD[iz]
                    * _cont_SN[iz]
                    * _cont_AGN[iz]
                    * _IC_corr[iz]
                )
                add_cont_total.append(_cont_add_metals[iz])
        print("end model_contaminants.py") 
        return mult_cont_total, add_cont_total


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
