# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from cupix.likelihood.generate_fake_data import FakeData
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
from lace.cosmo.thermal_broadening import thermal_broadening_kms
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
from cupix.likelihood.lya_theory import Theory
import forestflow
from forestflow.archive import GadgetArchive3D
import os
from pathlib import Path
import h5py 
# %load_ext autoreload
# %autoreload 2


# %%
# realdata = DESI_DR2("/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/binned_out_px-zbins_4-thetabins_20_w_res.hdf5")
mockdata_file = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/analysis-0/tru_cont/binned_out_px-zbins_2-thetabins_18.hdf5"
mockdata = DESI_DR2(mockdata_file)

# %%
mockdata.k_M_edges[0][1]-mockdata.k_M_edges[0][0]

# %%
# Figure out the ForestFlow training central simulation
path_program = os.path.dirname(forestflow.__path__[0]) + '/'
path_program
folder_lya_data = path_program + "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    average="both",
)
training_data = Archive3D.training_data

# %%
zs = []
sim_dict =  Archive3D.get_testing_data("mpg_central")
for sim_z in sim_dict:
    if (sim_z['z'] < 2.7) & (sim_z['z'] > 2.2):
        zs.append(sim_z['z'])
zs = np.sort(np.array(zs))

# %%
# manually overwrite MockData zs with ForestFlow zs
mockdata.z = zs
print(mockdata.z)

# %%
# Load emulator
# z = MockData.z[0:4]

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
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=zs, camb_kmax_Mpc=1000)

ffemu = FF_emulator(zs, fid_cosmo, cc)
ffemu.kp_Mpc = 1 # set pivot point

theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(zs)
theory_AA.emulator = ffemu
dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array(zs), camb_results=cc)

# %% [markdown]
# ## Generate it at the center of the original training simulation

# %%
like_params = []

sim_dict =  Archive3D.get_testing_data("mpg_central")
for par in ['sigT_Mpc', 'gamma', 'mF', 'Delta2_p', 'n_p', 'kF_Mpc']:
    thispar_values = []
    for iz,zz in enumerate(zs):
        for sim_z in sim_dict:
            if abs(sim_z['z']-zz)<0.01:
                print(sim_z['z'])
                for par2 in sim_z.keys():
                    if par2 == par:
                        thispar_values.append(sim_z[par])
    thispar_values = np.asarray(thispar_values)
    like_params.append(LikelihoodParameter(name=par, min_value=-20, max_value=20, value=thispar_values))

# %%
for par in like_params:
    print(par.name, par.value)

# %%
like = Likelihood(mockdata, theory_AA, free_param_names=["bias"], iz_choice=np.asarray([0,1]), like_params=like_params, verbose=True)

# %%
fakedata = FakeData(like)

# %%
savestr

# %%
add_noise = False
if add_noise:
    noise_str = 'noisy'
else:
    noise_str = 'noiseless'
savestr = f"../../data/px_measurements/forecast/forecast_ffcentral_{Path(mockdata_file).stem}_{noise_str}.hdf5"
print("Save to...", savestr)
fakedata.write_to_file(savestr, add_noise=False)

# %%
# just copying and pasting from the previous output
arinyo_dict = {'bias': np.array([0.11810589, 0.14949374]), 'beta': np.array([1.52642012, 1.4411819 ]), 'q1': np.array([0.29296646, 0.31081784]), 'kvav': np.array([0.51745933, 0.54350346]), 'av': np.array([0.32345456, 0.37763417]), 'bv': np.array([1.6457237 , 1.69155955]), 'kp': np.array([12.90727806, 13.73200226]), 'q2': np.array([0.26116902, 0.28188545])}


# %%
# add the Arinyo parameters to the attributes
loaded_file = h5py.File(savestr, 'a')
print(loaded_file.keys()) 
for par in arinyo_dict:
    loaded_file['like_params'].attrs[par] = arinyo_dict[par]
loaded_file.close()

# %%
# Walther+ constriaints
T0_z2p2 = 1.014*1e4 # Kelvin
gamma_z2p2 = 1.74
mF_z2p2 = 0.825
lambdap_z2p2 = 79.4 # [kpc]

T0_z2p4 = 1.165*1e4
gamma_z2p4 = 1.63
mF_z2p4 = 0.799
lambdap_z2p4 = 81.1 # [kpc]

sigma_T_kms_z2p2 = thermal_broadening_kms(T0_z2p2)
sigT_Mpc_z2p2 = sigma_T_kms_z2p2 / dkms_dMpc_zs[0]
kF_Mpc_z2p2 = 1/(lambdap_z2p2/1000)

sigma_T_kms_z2p4 = thermal_broadening_kms(T0_z2p4)
sigT_Mpc_z2p4 = sigma_T_kms_z2p4 / dkms_dMpc_zs[0]
kF_Mpc_z2p4 = 1/(lambdap_z2p4/1000)

# %%
# compute linear power parameters at each z (in Mpc units)
linP_zs = fit_linP.get_linP_Mpc_zs(
    sim_cosmo, z, 0.7
)

# %%
like_params = []
like_params.append(LikelihoodParameter(
    name='sigT_Mpc',
    min_value=-1.0,
    max_value=1.0,
    value=np.array([sigT_Mpc_z2p2, sigT_Mpc_z2p4]),
    ini_value=sigT_Mpc_z2p2,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='gamma',
    min_value=0.0,
    max_value=3.0,
    value = np.array([gamma_z2p2, gamma_z2p4]),
    ini_value = gamma_z2p4,
    Gauss_priors_width=1
    ))
like_params.append(LikelihoodParameter(
    name='mF',
    min_value=0.0,
    max_value=1.0,
    ini_value = mF_z2p2,
    value = np.array([mF_z2p2, mF_z2p4]),
    Gauss_priors_width=0.5
    ))
like_params.append(LikelihoodParameter(
    name='Delta2_p',
    min_value=0.0,
    max_value=1.0,
    ini_value = linP_zs[0]['Delta2_p'],
    value = np.array([linP_zs[0]['Delta2_p'], linP_zs[1]['Delta2_p']])
    ))
like_params.append(LikelihoodParameter(
    name='n_p',
    min_value=0.0,
    max_value=1.0,
    ini_value = linP_zs[0]['n_p'],
    value = np.array([linP_zs[0]['n_p'], linP_zs[1]['n_p']])
    ))
like_params.append(LikelihoodParameter(
    name='kF_Mpc',
    min_value=0.0,
    max_value=1.0,
    ini_value = kF_Mpc_z2p2,
    value = np.array([kF_Mpc_z2p2, kF_Mpc_z2p4])
    ))
like_params.append(LikelihoodParameter(
    name='lambda_P',
    min_value=0.0,
    max_value=1.0,
    ini_value = lambdap_z2p2,
    value = np.array([lambdap_z2p2, lambdap_z2p4])
    ))

# %%
# # set the likelihood parameters as the Arinyo params with some fiducial values

# like_params = []
# like_params.append(LikelihoodParameter(
#     name='bias',
#     min_value=-1.0,
#     max_value=0,
#     ini_value=-0.05,
#     value =-0.115,
#     Gauss_priors_width=.05
#     ))
# like_params.append(LikelihoodParameter(
#     name='beta',
#     min_value=0.0,
#     max_value=2.0,    
#     ini_value = 1.2,
#     value=1.55,
#     Gauss_priors_width=.5
#     ))
# like_params.append(LikelihoodParameter(
#     name='q1',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.04,
#     value=0.1112,
#     Gauss_priors_width=0.111
#     ))
# like_params.append(LikelihoodParameter(
#     name='kvav',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.0003**0.2694,
#     value=0.0001**0.2694,
#     Gauss_priors_width=0.0003**0.2694,
#     ))
# like_params.append(LikelihoodParameter(
#     name='av',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.7,
#     value=0.2694,
#     Gauss_priors_width=0.27
#     ))
# like_params.append(LikelihoodParameter(
#     name='bv',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.0004,
#     value=0.0002,
#     Gauss_priors_width=0.0002
#     ))
# like_params.append(LikelihoodParameter(
#     name='kp',
#     min_value=0.0,
#     max_value=1.0,
#     ini_value = 0.33,
#     value=0.5740,
#     Gauss_priors_width=0.5
#     ))


# # likelihood_params = []
# # likelihood_params.append(LikelihoodParameter(
# #     name='Delta2_p',
# #     min_value=-1.0,
# #     max_value=1.0,
# #     ))
# # likelihood_params.append(LikelihoodParameter(
# #     name='n_p',
# #     min_value=-1.0,
# #     max_value=1.0,
# #     ))
# # likelihood_params.append(LikelihoodParameter(
# #     name='mF',
# #     min_value=-1.0,
# #     max_value=1.0,
# #     ))
# # likelihood_params.append(LikelihoodParameter(
# #     name='gamma',
# #     min_value=-1.0,
# #     max_value=1.0,
# #     ))
# # likelihood_params.append(LikelihoodParameter(
# #     name='kF_Mpc',
# #     min_value=-1.0,
# #     max_value=1.0,
# #     ))
# # likelihood_params.append(LikelihoodParameter(
# #     name='sigT_Mpc',
# #     min_value=-1.0,
# #     max_value=1.0,
# #     ))

# %%
like = Likelihood(MockData, theory_AA, free_param_names=["bias"], iz_choice=[0,1], like_params=like_params, verbose=True)

# %%
fakedata = FakeData(like)

# %%
# save the fake data in the same format as the Colore mock measurements
fakedata.write_to_file("../../data/px_measurements/forecast/forecast_binned_out_px-zbins_4-thetabins_20_noiseless.hdf5", add_noise=False)

# %%
import h5py as h5

# %%
dat = h5.File("../../data/px_measurements/forecast/forecast_binned_out_px-zbins_4-thetabins_20_noiseless.hdf5")

# %%
dat.keys()

# %%
dat['P_Z_AM']['z_1']['theta_rebin_0'][:]

# %%
dat.close()

# %%
# read in the fake data to verify
forecast = DESI_DR2("../../data/px_measurements/forecast/forecast_binned_out_px-zbins_4-thetabins_20_noiseless.hdf5", theta_min_cut_arcmin=0, kmax_cut_AA=1)

# %%
forecast

# %%
forecast.cov_ZAM[0].shape

# %%
np.sqrt(np.diag(forecast.cov_ZAM[z][iA])).shape

# %%
forecast.Px_ZAM[z][iA,:].shape

# %%

for z in range(len(forecast.z)):
    print("z bin ", z)
    k_forecast = (forecast.k_M_edges[z][:-1]+forecast.k_M_edges[z][1:])/2.
    k_data = (MockData.k_M_edges[z][:-1]+MockData.k_M_edges[z][1:])/2.
    for iA in range(0,forecast.Px_ZAM[z].shape[0], 3):    
        plt.errorbar(k_forecast, forecast.Px_ZAM[z][iA,:], yerr=np.sqrt(np.diag(forecast.cov_ZAM[z][iA])), fmt='o', label=f'Theta={like.data.theta_centers_arcmin[iA]:.2f}\', fake data')
        # compare with the real data
        plt.errorbar(k_data, MockData.Px_ZAM[z][iA,:], yerr=np.sqrt(np.diag(MockData.cov_ZAM[z][iA])), fmt='*', color='k', label='Lyacolore Mock Data')
        plt.legend()
        plt.show()
        plt.clf()

# %%
# noisy_Px = fakedata.generate_px(iz_choice=0, theta_A_ind=np.arange(len(MockData.theta_centers_arcmin)), like_params=like_params, add_noise=False)

# %%
k = (like.data.k_M_edges[:-1]+like.data.k_M_edges[1:])/2.
for iA in range(0,noisy_Px.shape[0], 1):
    plt.errorbar(k, noisy_Px[iA,:], yerr=np.sqrt(np.diag(like.data.cov_ZAM[0][iA])), fmt='o', label=f'Theta={like.data.theta_centers_arcmin[iA]:.2f}\', fake data')
    # compare with the real data
    plt.errorbar(k, like.data.Px_ZAM[0][iA,:], yerr=np.sqrt(np.diag(like.data.cov_ZAM[0][iA])), fmt='*', color='k', label='Lyacolore Mock Data')
    plt.legend()
    plt.show()
    plt.clf()

# %%
