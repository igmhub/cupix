import cupix
import numpy as np
import numpy as np
import os
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from lace.cosmo import camb_cosmo, fit_linP
import matplotlib.pyplot as plt
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import scipy
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
from astropy.io import fits

def save_analysis_npz(results, filename="analysis_results.npz"):
    """
    results: list or dict of per-analysis dictionaries
    """
    out = {}

    if isinstance(results, list):
        for i, r in enumerate(results):
            out[f'analysis-{i}'] = r
    else:  # dict
        for k, r in results.items():
            out[str(k)] = r

    # Save each dict as an object
    np.savez(filename, **out, allow_pickle=True)

###################
####  CHOICES #####
# mock_type = 'raw'
# mock_type = 'uncontaminated'
mock_type = 'tru_cont'
analysis_type = 'stack'

# bin_type = 'unbinned'
bin_type = 'binned'
if bin_type == 'unbinned':
    ntheta = 20
else:
    ntheta = 5

nmocks=50
iz_choice = 0

##################
# repo = os.path.dirname(cupix.__path__[0])
# savedir = os.path.join(repo, "data", "fitter_results")
savedir = "/pscratch/sd/m/mlokken/desi-lya/px/" # to scratch for NERSC write-out

# Load emulator

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
# read redshifts from file
MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-0/{mock_type}/{bin_type}_out_px-zbins_2-thetabins_{ntheta}_w_res.hdf5", theta_min_cut_arcmin=14, kmax_cut_AA=1)

z = MockData.z

sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)
ffemu = FF_emulator(z, fid_cosmo, cc)
ffemu.kp_Mpc = 1 # set pivot point

theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu


# original Laura fits
like_params = []
like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-.5,
    max_value=0.0,
    value=-0.115,
    ini_value=-0.117,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.0,
    max_value=3.0,    
    value = 1.55,
    ini_value = 1.4,
    Gauss_priors_width=1
    ))
like_params.append(LikelihoodParameter(
    name='q1',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.1112,
    value = 0.1112,
    Gauss_priors_width=0.5
    ))
like_params.append(LikelihoodParameter(
    name='kvav',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.0001**0.2694,
    value = 0.0001**0.2694
    ))
like_params.append(LikelihoodParameter(
    name='av',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.2694,
    value = 0.2694
    ))
like_params.append(LikelihoodParameter(
    name='bv',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.0002,
    value = 0.0002
    ))
like_params.append(LikelihoodParameter(
    name='kp',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.5740,
    value = 0.5740
    ))



for n in range(nmocks):
    savefile = f"iminuit_mock{n}_{mock_type}_{analysis_type}_{bin_type}_bias_beta.npz"
    savepath = os.path.join(savedir,savefile)
    
    if not os.path.exists(savepath):
        print(f"Analyzing mock {n}")
        MockData = DESI_DR2(f"/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-{n}/{mock_type}/{bin_type}_out_px-zbins_2-thetabins_{ntheta}_w_res.hdf5", theta_min_cut_arcmin=14, kmax_cut_AA=1)
        like = Likelihood(MockData, theory_AA, free_param_names=["bias", "beta"], iz_choice=iz_choice, like_params=like_params)
        mini = IminuitMinimizer(like, verbose=True)
        mini.minimize()
        print("Done minimizing.")
        results_dict = {}
        for parname in like.free_param_names:
            bestfit, err = mini.best_fit_value(parname, return_hesse=True)
            results_dict[parname] = bestfit
            results_dict[parname+'_err'] = err
        prob = like.fit_probability(mini.minimizer.values)
        results_dict['prob'] = prob
        chi2 = like.get_chi2(mini.minimizer.values)
        results_dict['chi2'] = chi2
        save_analysis_npz(results_dict, savepath)