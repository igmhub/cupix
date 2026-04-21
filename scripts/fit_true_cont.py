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

# first we will fit bias / beta to theta > 10 arcmin to get a prior for these
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
kM_max_cut_AA = 1.0
km_max_cut_AA = 1.1*kM_max_cut_AA
theta_min_cut_arcmin = 10.0
data = DESI_DR2(fname, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA, theta_min_cut_arcmin=theta_min_cut_arcmin)
iz = 1
z = data.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))

# setup cosmology (should check this is the right cosmology in the mocks)
cosmo = cosmology.Cosmology()
# starting point for Lya bias parameters in mocks
default_lya_model = 'best_fit_arinyo_from_colore'
theory_config = {'verbose': False, 'default_lya_model': default_lya_model, 'include_continuum': False}
theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)

# set initial value for bias / beta based on best-fit values from Laura
ini_bias = theory.lya_model.default_lya_params['bias']
ini_beta = theory.lya_model.default_lya_params['beta']
par_bias = FreeParameter(
    name='bias',
    min_value=-0.5,
    max_value=-0.01,
    ini_value=ini_bias,
    delta=0.01,
)
par_beta = FreeParameter(
    name='beta',
    min_value=0.1,
    max_value=5.0,
    ini_value=ini_beta,
    delta=0.1,
)
free_params = [par_bias, par_beta]
for par in free_params:
    print(par.name, par.ini_value)

# set likelihood / posterior / minimizer
like = Likelihood(data, theory, iz=iz, config={'verbose':False})
post = Posterior(like, free_params, config={'verbose':False})
mini = Minimizer(post, config={'verbose':False})

# run minimizer
start = time.time()
mini.silence()
mini.minimize()
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")

# print results to screen
mini.print_results()


# set priors to bias / beta
for par in free_params:
    pname = par.name
    val, err = mini.get_best_fit_value(pname, return_hesse=True)
    print(pname, val, err)
    par.gauss_prior_mean = val
    par.gauss_prior_width = err

# add other free parameters (without prior values for now)
ini_q1 = theory.lya_model.default_lya_params['q1']
par_q1 = FreeParameter(
    name='q1',
    min_value=0.0,
    max_value=5.0,
    ini_value=ini_q1,
    delta=0.01
)
ini_av = theory.lya_model.default_lya_params['av']
par_av = FreeParameter(
    name='av',
    min_value=0.0,
    max_value=5.0,
    ini_value=ini_av,
    delta=0.01
)
ini_bv = theory.lya_model.default_lya_params['bv']
par_bv = FreeParameter(
    name='bv',
    min_value=0.0,
    max_value=5.0,
    ini_value=ini_bv,
    delta=0.01
)
ini_kp = theory.lya_model.default_lya_params['kp_Mpc']
par_kp = FreeParameter(
    name='kp_Mpc',
    min_value=0.0,
    max_value=5.0,
    ini_value=ini_kp,
    delta=0.01
)
ini_kv = theory.lya_model.default_lya_params['kv_Mpc']
par_kv = FreeParameter(
    name='kv_Mpc',
    min_value=0.0,
    max_value=5.0,
    ini_value=ini_kv,
    delta=0.01
)

free_q1 = True
free_av = True
free_bv = False
free_kp = True
free_kv = True
if free_q1: free_params.append(par_q1)
if free_av: free_params.append(par_av)
if free_bv: free_params.append(par_bv)
if free_kp: free_params.append(par_kp)
if free_kv: free_params.append(par_kv)
for par in free_params:
    print(par.name, par.ini_value)


# run multiple analyses with different theta cuts
runs = []
for theta in [0.5, 1.0, 2.0, 3.0]:
    run = {}
    run['data'] = DESI_DR2(fname, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA, theta_min_cut_arcmin=theta)
    run['theta_min'] = run['data'].theta_min_a_arcmin[0]
    run['theory'] = theory
    run['like'] = Likelihood(data=run['data'], theory=run['theory'], iz=iz, config={'verbose':False})
    run['post'] = Posterior(run['like'], free_params, config={'verbose':False})
    run['mini'] = Minimizer(run['post'], config={'verbose':False})
    runs.append(run)

for ii, run in enumerate(runs):
    theta_min = run['theta_min']
    print('-----------------------------')
    print('minimize theta_min = {:.3f} arcmin'.format(theta_min))
    run['mini'].silence()
    start = time.time()
    run['mini'].minimize()
    end = time.time()
    print(f"Time taken for minimization: {end - start:.2f} seconds")
    run['mini'].print_results()

