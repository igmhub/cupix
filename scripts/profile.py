import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"

import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from lace.cosmo import camb_cosmo
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import h5py as h5
import time
import pandas as pd
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

rng = np.random.default_rng()
outfile = "/pscratch/sd/m/mlokken/desi-lya/px/data/profiling/profile_results.csv"
timing_results = {}

forecast_file = "/global/common/software/desi/users/mlokken/cupix/data/px_measurements/forecast//forecast_ffcentral_cosmo_igm_binned_out_px-zbins_2-thetabins_18_noiseless.hdf5"
forecast = DESI_DR2(forecast_file, kmax_cut_AA=1)


iz_choice = np.array([0])
param_names = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
z = forecast.z[iz_choice]

omnuh2 = 0.0006
mnu = omnuh2 * 93.14
H0 = 67.36
omch2 = 0.12
ombh2 = 0.02237
As = 2.1e-9
ns = 0.9649
nrun = 0.0
w = -1.0
omk = 0
fid_cosmo = {
    'H0': H0,
    'omch2': omch2,
    'ombh2': ombh2,
    'mnu': mnu,
    'omk': omk,
    'As': As,
    'ns': ns,
    'nrun': nrun,
    'w': w
}
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)

ffemu = FF_emulator(z, fid_cosmo, cc, Nrealizations=3000)
ffemu.kp_Mpc = 1 # set pivot point

theory_AA = set_theory(ffemu, k_unit='iAA', verbose=True)
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu
# dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array(z), camb_results=cc)

# read the likelihood params from forecast file
like_params_dict = {}
like_params = []
with h5.File(forecast_file, "r") as f:
    params = f['like_params']
    attrs = params.attrs
    for key in param_names: # important to get the sorting right
        if key in attrs:
            val = attrs[key][:]
            like_params_dict[key] = val[iz_choice][0] + rng.uniform(-0.05,0.05)
            like_params.append(LikelihoodParameter(
                name=key,
                value=val[iz_choice] + rng.uniform(-0.05,0.05), # add some noise to the initial value
                ini_value= val[iz_choice] + rng.uniform(-0.05,0.05), # not used for now
                min_value=-1000,
                max_value=1000, # not used for now
            ))
# check the parameters
for p in like_params_dict:
    print(p, like_params_dict[p])


# start = time.time()
# ffemu.emu.predict_Arinyos(emu_params=like_params_dict, Nrealizations=3000, return_all_realizations=False)
# end = time.time()
# print(f"Time taken for prediction: {end - start:.2f} seconds")
# timing_results['predict_Arinyos'] = end - start

like = Likelihood(forecast, theory_AA, free_param_names=["mF", "gamma"], iz_choice=iz_choice, like_params=like_params, verbose=True)

# start = time.time()
# like.get_log_like(np.ones(len(like_params))*0.5)
# end = time.time()
# print(f"Time taken for likelihood evaluation: {end - start:.2f} seconds")
# timing_results['get_log_like'] = end - start

# start = time.time()
# like.minus_log_prob(np.ones(len(like_params))*0.5)
# end = time.time()
# print(f"Time taken for minus_log_prob: {end - start:.2f} seconds")
# timing_results['minus_log_prob'] = end - start


# start = time.time()
# like.get_convolved_Px_AA(iz_choice, np.arange(len(forecast.theta_max_A_arcmin)), like_params)
# end = time.time()
# print(f"Time taken for get_convolved_Px_AA: {end - start:.2f} seconds")
# timing_results['get_convolved_Px_AA'] = end - start



from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
mini = IminuitMinimizer(like, verbose=False)
start = time.time()
mini.minimize(compute_hesse=True)
end = time.time()
print(f"Time taken for minimization: {end - start:.2f} seconds")
timing_results['minimization'] = end - start

# # save the timing results to a csv file
# df = pd.DataFrame.from_dict(timing_results, orient='index', columns=['time_seconds'])
# df.to_csv(outfile)