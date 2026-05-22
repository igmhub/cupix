import numpy as np
import emcee
from forestflow import priors
from cupix.likelihood.free_parameter import FreeParameter
from cupix.likelihood.model_lya import get_priors_gadget

def get_latex_label(parname):
    if parname == 'bias':
        return r'b_\alpha'
    if parname == 'beta':
        return r'\beta_\alpha'
    if parname == 'q1':
        return r'q_1'
    if parname == 'bv':
        return r'b_v'
    if parname == 'av':
        return r'a_v'
    if parname == 'kv_Mpc':
        return r'k_v [Mpc^{-1}]'
    if parname == 'kp_Mpc':
        return r'k_p [Mpc^{-1}]'
    # need to add IGM parameters

    
def prepare_free_parameters(free_param_names, theory, config, shift_ini = .05):
    """
    Prepare list of FreeParameter objects for the sampler, based on the free_param_names and the theory. Inputs:
    - free_param_names: list of strings with the names of the free parameters, e.g. ['bias', 'beta']
    - theory: Theory object, used to get the redshift and other info
    - config: dictionary with different settings, including default Lya model
    - shift_ini: the amount by which to shift the initial value of the parameter from the true value, as a fraction (e.g. 0.05 means 5% shift)
        The shift will be randomly positive or negative.
    """
    free_params_list = [] # list of FreeParameter objects
    z = theory.z
    default_lya_model = config.get('default_lya_model', None)
    if 'p1d' in default_lya_model.lower():
        if 'igm' in default_lya_model.lower():
            prior_info = priors.get_IGM_priors(z=z, tag='DESI_DR1_P1D')
        elif 'arinyo' in default_lya_model.lower():
            prior_info = priors.get_arinyo_priors(z=z, tag='DESI_DR1_P1D')
        for parname in free_param_names:
            this_param = FreeParameter(
                name=parname,
                min_value=prior_info["percen_5"][parname], # note that this is only used in minimizer
                max_value=prior_info["percen_95"][parname], # note that this is only used in minimizer
                ini_value=prior_info["mean"][parname] + shift_ini * prior_info["mean"][parname] * np.random.choice([-1, 1]),
                true_value=prior_info["mean"][parname],
                gauss_prior_mean=prior_info["mean"][parname], # note that this is only used in sampler
                gauss_prior_width=prior_info["std"][parname], # note that this is only used in sampler
                delta=0.1*prior_info["std"][parname], # Will set steps of minimizer
                latex_label=get_latex_label(parname)
            )
            free_params_list.append(this_param)

    elif 'gadget' in default_lya_model.lower():
        if 'igm' in default_lya_model.lower():
            prior_info = get_priors_gadget(z=z, model='igm')
        elif 'arinyo' in default_lya_model.lower():
            prior_info = get_priors_gadget(z=z, model='arinyo')
        for parname in free_param_names:
            this_param = FreeParameter(
                name=parname,
                min_value=prior_info[parname]["min"], # note that this is only used in minimizer
                max_value=prior_info[parname]["max"], # note that this is only used in minimizer
                ini_value=prior_info[parname]["mean"] + shift_ini * prior_info[parname]["mean"] * np.random.choice([-1, 1]),
                true_value=prior_info[parname]["mean"],
                gauss_prior_mean=prior_info[parname]["mean"], # note that this is only used in sampler
                gauss_prior_width=prior_info[parname]["std"], # note that this is only used in sampler
                delta=0.1*prior_info[parname]["std"], # Will set steps of minimizer
                latex_label=get_latex_label(parname)
            )
            free_params_list.append(this_param)
    return free_params_list
