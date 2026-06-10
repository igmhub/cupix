import sys
import time
from cupix.inference.posterior import Posterior
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from datetime import datetime
import emcee
import shutil


from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood

from cupix.inference.sampling_funcs import prepare_free_parameters
from cupix.likelihood.config import Config
from cupix.inference.inference_config import InferenceConfig
from cupix.inference.sampling_funcs import plot_tau_estimates, plot_chains, plot_contours, create_output_directory, save_chain
from cupix.inference.minimize_posterior import Minimizer

# set the directory name
# YYYYMMDD_shorttag/
if len(sys.argv) < 4:
    sys.exit("Usage: python minimizer_pipeline.py [short tag name] [setup_config.yaml] [inference_config.yaml].")
# example setup config is in: cupixpath + "/example_configs/fcast_best_fit_arinyo_from_p1d_theory_config.yaml"
# example inference config is in: cupixpath + "/example_configs/inference_config_ex.yaml"
yyyymmdd = datetime.now().strftime("%Y%m%d")
shorttag = sys.argv[1]
runname = f'{yyyymmdd}_{shorttag}'

setup_config_path = sys.argv[2]
inf_config_path = sys.argv[3]
setup_config = Config(setup_config_path)
inf_config = InferenceConfig(inf_config_path)

setup_config.print_all()
inf_config.print_all()

outdir = create_output_directory(inf_config, runname)
print("Outputs will be saved to", outdir)
# copy the run config into the outdir
shutil.copy(setup_config_path, os.path.join(outdir, 'setup_config_mini.yaml'))
shutil.copy(inf_config_path, os.path.join(outdir, 'inference_config_mini.yaml'))

data = DESI_DR2(setup_config.data_config)
iz = setup_config.theory_config['iz']
z = data.z[iz]

# update config class with a use_truth option that could replace the cosmo and theory params with forecast values if it is a forecast and use_truth is True
# is_forecast = config.data_config['is_forecast']
# use_truth = config.post_config.get('use_truth', False)
# if is_forecast and use_truth:
# config.update_with_truth()

cosmo = cosmology.Cosmology(cosmo_params_dict=setup_config.cosmo_config)

theory = Theory(z=z, fid_cosmo=cosmo, config=setup_config.theory_config)
like = Likelihood(data=data, theory=theory, iz=iz, 
                config=setup_config.like_config)

free_param_names = list(inf_config.params_config.keys())
free_params = prepare_free_parameters(free_param_names, theory, setup_config.theory_config, params_config=inf_config.params_config)

for par in free_params:
    print("Free parameters are: (name, ini_value, true_value, gauss_prior_mean, gauss_prior_width)", par.name, par.ini_value, par.true_value, par.gauss_prior_mean, par.gauss_prior_width)

post = Posterior(like, free_params, config=inf_config.post_config)

minimizer_start = time.time()

mini = Minimizer(post, config=inf_config.mini_config)
mini.silence()
mini.minimize()
minimizer_end = time.time()
print("Time to run sampler: %.2f seconds" % (minimizer_end - minimizer_start))
mini.save_results(outdir=outdir) # will save to file called iminuit_results.npz
mini.plot_ellipses('bias', 'kp_Mpc')
mini.print_results()
mini.plot_best_fit(outdir=outdir)