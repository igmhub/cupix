import os
import h5py as h5
import numpy as np

from cupix.likelihood.config import Config
from cupix.inference.inference_config import InferenceConfig
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.inference.sampling_funcs import prepare_free_parameters
from lace.cosmo import cosmology

def load_mini_results(results_directory):
    """ Load the chain, and all setup configs, from directory to be able to re-create or continue working
    with analysis."""
    # Martine note: I might change this to a class later, since there are a lot of items to be returned
    # load the chain
    minires_fname = os.path.join(results_directory, "iminuit_results.npz")
    
    # Load results file
    outfile = np.load(minires_fname)
    # turn outfile into a dictionary
    results_dict = {key: outfile[key] for key in outfile.files}

    # recreate the theory, posterior, and likelihood objects
    setup_config = Config(os.path.join(results_directory, 'setup_config_mini.yaml'))
    inf_config = InferenceConfig(os.path.join(results_directory, 'inference_config_mini.yaml'))
    data = DESI_DR2(setup_config.data_config)
    iz = setup_config.theory_config['iz']
    z = data.z[iz]
    cosmo = cosmology.Cosmology(cosmo_params_dict=setup_config.cosmo_config)
    theory = Theory(z=z, fid_cosmo=cosmo, config=setup_config.theory_config)
    like = Likelihood(data=data, theory=theory, iz=iz, 
                config=setup_config.like_config)
    free_param_names = list(inf_config.params_config.keys())
    free_params = prepare_free_parameters(free_param_names, theory, setup_config.theory_config, params_config=inf_config.params_config)

    return results_dict, free_params, data, cosmo, theory, like, setup_config, inf_config