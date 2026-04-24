# Run emcee on a set of parameters, using the forecast data.
import numpy as np
import time

from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.likelihood.free_parameter import FreeParameter
from cupix.likelihood.posterior import Posterior
from cupix.likelihood.minimize_posterior import Minimizer
from cupix.likelihood.sampler import Sampler
from cupix.utils.utils import get_path_repo
cupixpath = get_path_repo('cupix')

print('cupix path', cupixpath)

# for each redshift, we will fit bias / beta / kp_Mpc using theta > 5 arcmin to get a prior for these
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
kM_max_cut_AA = 0.5
km_max_cut_AA = 1.1*kM_max_cut_AA
theta_min_cut_arcmin = 5.0
data = DESI_DR2(fname, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA, theta_min_cut_arcmin=theta_min_cut_arcmin)

# setup cosmology (should check this is the right cosmology in the mocks)
cosmo = cosmology.Cosmology()
# starting point for Lya bias parameters in mocks
#default_lya_model = 'best_fit_arinyo_from_colore'
default_lya_model = 'pressure_only_fits_from_colore'
theory_config = {'verbose': False, 'default_lya_model': default_lya_model, 'include_continuum': False}

# set free parameters (ini values depend on z)
par_bias = FreeParameter(
    name='bias',
    min_value=-0.5,
    max_value=-0.01,
    ini_value=None,
    delta=0.01,
)
par_beta = FreeParameter(
    name='beta',
    min_value=0.1,
    max_value=5.0,
    ini_value=None,
    delta=0.1,
)
par_kp = FreeParameter(
    name='kp_Mpc',
    min_value=0.0,
    max_value=5.0,
    ini_value=None,
    delta=0.01
)
free_params = [par_bias, par_beta, par_kp]

def run_z_bin(iz):
    z = data.z[iz]
    print('analyze zbin {}, z = {}'.format(iz, z))
    run = {'z': z}

    # setup theory
    run['theory'] = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
    ini_bias = run['theory'].lya_model.default_lya_params['bias']
    free_params[0].ini_value = ini_bias
    ini_beta = run['theory'].lya_model.default_lya_params['beta']
    free_params[1].ini_value = ini_beta
    ini_kp = run['theory'].lya_model.default_lya_params['kp_Mpc']
    free_params[2].ini_value = ini_kp
    for par in free_params:
        print(par.name, par.ini_value)

    # set likelihood / posterior / minimizer
    run['like'] = Likelihood(data, run['theory'], iz=iz, config={'verbose':False})
    run['post'] = Posterior(run['like'], free_params, config={'verbose':False})
    run['mini'] = Minimizer(run['post'], config={'verbose':False})

    # run minimizer
    start = time.time()
    run['mini'].silence()
    run['mini'].minimize()
    end = time.time()
    print(f"Time taken for minimization: {end - start:.2f} seconds")

    # print results to screen
    run['mini'].print_results()

    return run


# run multiple analyses for each z bin
runs = []
for iz in range(4):
    run = run_z_bin(iz)
    runs.append(run)

