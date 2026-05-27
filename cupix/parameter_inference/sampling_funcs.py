import numpy as np
import emcee
from forestflow import priors
from cupix.likelihood.free_parameter import FreeParameter
from cupix.likelihood.model_lya import get_priors_gadget, get_priors_colore

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

    if parname == 'Delta2_p':
        return r'\Delta^2_p'
    if parname == 'n_p':
        return r'n_p'
    if parname == 'mF':
        return r'm_F'
    if parname == 'gamma':
        return r'\gamma'
    if parname == 'sigT_Mpc':
        return r'\sigma_T [Mpc]'
    if parname == 'kF_Mpc':
        return r'k_F [Mpc^{-1}]'
    
def prepare_free_parameters(free_param_names, theory, theory_config, shift_ini = .05, params_config={}):
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
    default_lya_model = theory_config.get('default_lya_model', '')
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

    elif 'colore' in default_lya_model.lower():
        prior_info = get_priors_colore(z)
        for parname in free_param_names:
            this_param = FreeParameter(
                name=parname,
                min_value=prior_info[parname]["min"],
                max_value=prior_info[parname]["max"],
                ini_value=prior_info[parname]["mean"] + shift_ini * prior_info[parname]["mean"] * np.random.choice([-1, 1]),
                true_value=prior_info[parname]["mean"],
                gauss_prior_mean=prior_info[parname]["mean"], # note that this is only used in sampler
                gauss_prior_width=prior_info[parname]["std"], # note that this is only used in sampler
                delta=0.1*prior_info[parname]["std"], # Will set steps of minimizer
                latex_label=get_latex_label(parname)
            )
            free_params_list.append(this_param)
    # if params_config is not empty, replace with whatever is in params_config
    for par in free_params_list:
        if par.name in params_config:
            par.min_value = params_config[par.name].get('min_value', par.min_value)
            par.max_value = params_config[par.name].get('max_value', par.max_value)
            par.ini_value = params_config[par.name].get('ini_value', par.ini_value)
            par.true_value = params_config[par.name].get('true_value', par.true_value)
            par.gauss_prior_mean = params_config[par.name].get('gauss_prior_mean', par.gauss_prior_mean)
            par.gauss_prior_width = params_config[par.name].get('gauss_prior_width', par.gauss_prior_width)
            par.delta = params_config[par.name].get('delta', par.delta)
    for parname in params_config: # create the FreeParam object for any missing ones. This is the case when default_lya_model is None.
        if (parname not in [par.name for par in free_params_list]) and (parname in free_param_names):
            this_param = FreeParameter(
                name=parname,
                min_value=params_config[parname].get('min_value', None),
                max_value=params_config[parname].get('max_value', None),
                ini_value=params_config[parname].get('ini_value', None),
                true_value=params_config[parname].get('true_value', None),
                gauss_prior_mean=params_config[parname].get('gauss_prior_mean', None), # note that this is only used in sampler
                gauss_prior_width=params_config[parname].get('gauss_prior_width', None), # note that this is only used in sampler
                delta=params_config[parname].get('delta', None), # Will set steps of minimizer
                latex_label=get_latex_label(parname)
            )
            free_params_list.append(this_param)
    return free_params_list
