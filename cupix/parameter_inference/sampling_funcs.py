import numpy as np
import emcee
import matplotlib.pyplot as plt
import yaml
import os
import h5py as h5
from getdist import MCSamples, plots
# ours
from forestflow import priors
from cupix.likelihood.free_parameter import FreeParameter
from cupix.likelihood.model_lya import get_priors_gadget, get_priors_colore, allowed_igm_params, allowed_lya_params
from cupix.likelihood.model_contaminants import allowed_continuum_params, allowed_hcd_params, allowed_metal_params, allowed_sky_params

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

    # first, split the free param names into lya/igm or contaminants
    free_param_names_lyaigm = [par for par in free_param_names if par in allowed_lya_params() or par in allowed_igm_params()]
    free_param_names_cont   = [par for par in free_param_names if par in allowed_continuum_params() or par in allowed_hcd_params() or par in allowed_metal_params() or par in allowed_sky_params()]

    if 'p1d' in default_lya_model.lower():
        if 'igm' in default_lya_model.lower():
            prior_info = priors.get_IGM_priors(z=z, tag='DESI_DR1_P1D')
        elif 'arinyo' in default_lya_model.lower():
            prior_info = priors.get_arinyo_priors(z=z, tag='DESI_DR1_P1D')
        for parname in free_param_names_lyaigm:
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
        for parname in free_param_names_lyaigm:
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
        for parname in free_param_names_lyaigm:
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
    for parname in params_config: # create the FreeParam object for any missing ones. This is the case when default_lya_model is None and for any contaminant params.
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

def plot_tau_estimates(tau_estimates, fname):
    plt.plot(np.arange(len(tau_estimates))*100, tau_estimates)
    plt.xlabel("step")
    plt.ylabel("estimated autocorrelation time")
    plt.savefig(fname)
    plt.clf()

def plot_chains(outdir, emcee_sampler, free_params, param_idx):
    # first get the full chain and plot one, to understand burnin
    param_name = free_params[param_idx].name
    chain_full = emcee_sampler.get_chain(flat=False)
    plt.plot(chain_full[:, :, param_idx], alpha=0.5)
    plt.ylabel(free_params[param_idx].latex_label)
    plt.xlabel("step")
    plt.title("Full chain for parameter %s" % free_params[0].name)
    plt.savefig(os.path.join(outdir, f"chain_full_{param_name}.png"))
    plt.clf()
    
def plot_contours(outdir, chain, free_params, title=None):
    gdnames  = [par.name for par in free_params]
    gdlabels = [par.latex_label for par in free_params]
    gdsamples = MCSamples(samples=chain, names=gdnames, labels=gdlabels)
    g = plots.get_subplot_plotter()
    g.triangle_plot([gdsamples], filled=True)
    # add truth values
    for i, par in enumerate(free_params):
        if par.true_value is not None:
            g.add_param_markers({par.name:par.true_value})
    if title is not None:
        g.fig.suptitle(title)
    g.finish_plot()
    plt.savefig(os.path.join(outdir, "contours.png"))

def save_chain(outdir, chain, free_params):
    # save the chain
    chain_fname = os.path.join(outdir, "chain.h5")
    gdnames  = [par.name for par in free_params]
    gdlabels = [par.latex_label for par in free_params]
    with h5.File(chain_fname, 'w') as f:
        f.create_dataset('chain', data=chain)
        f.attrs['gdnames'] = gdnames
        f.attrs['gdlabels'] = gdlabels
        for par in free_params:
            if par.true_value is not None:
                f.attrs[f'{par.name}_true_value'] = par.true_value
            f.attrs[f'{par.name}_ini_value'] = par.ini_value
    
def record_mcmc_settings(outdir, nwalkers, max_nsteps, nburnin):
    # these settings might be changed from the inference config based on how many cores were available to the script
    # we record the final settings used for the sampler in a yaml file in the output directory for reproducability
    settings = {
        'nwalkers': nwalkers,
        'max_nsteps': max_nsteps,
        'nburnin': nburnin
    }
    with open(os.path.join(outdir, 'mcmc_settings.yaml'), 'w') as f:
        yaml.dump(settings, f)

def create_output_directory(inf_config, runname):
    basedir = inf_config.samp_config.get("outputs_dir", "/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/")
    output_dir = os.path.join(basedir,runname)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def get_initial_walkers(free_params, nwalkers):
    """Setup initial states of walkers in sensible points """

    ndim = len(free_params)

    # Random values between [0, 1) --> [-0.5, 0.5)
    shifts = -0.5 + np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    ini_walkers = np.empty_like(shifts)
    for ip, par in enumerate(free_params):
        ini_value = par.gauss_prior_mean
        rms = par.gauss_prior_width
        val = ini_value + shifts[:, ip] * rms
        # check that you don't end up outside the bounds
        min_val = par.min_value
        max_val = par.max_value
        _ = val < min_val
        val[_] = min_val + 0.01 * rms
        _ = val > max_val
        val[_] = max_val - 0.01 * rms

        # store into ndarray 
        ini_walkers[:, ip] = val

    return ini_walkers
