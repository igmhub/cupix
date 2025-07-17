# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: pcross
#     language: python
#     name: python3
# ---

import sys
import numpy as np
sys.path.append("/Users/mlokken/research/lyman_alpha/software/cupix")
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
# %load_ext autoreload
# %autoreload 2
from lace.cosmo import camb_cosmo
import matplotlib.pyplot as plt


# Load the emulator

# Load emulator
z = np.array([2.2,2.4])
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
cosmo = {
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
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)
ffemu = FF_emulator(z, cosmo, cc)

ffemu.emu_params

ffemu.kp_Mpc = 1 # pivot point

# Set the theory

# +
emu_params = Args()
emu_params.set_baseline()
# initial_gamma = 1.74
# initial_lambda_pressure = 79.4

theory_kms = set_theory(emu_params, ffemu, k_unit='ikms')

theory_kms.set_fid_cosmo(z)
# -

ffemu.emu_params

# theta should be of shape: # (N_z, N_theta, 2)
k_kms = np.array([np.linspace(0.0001,.1,100), np.linspace(0.0001,.1,100)])
theta_bin_deg = np.asarray([[[0,0.01],[0.01,0.1],[.1,.5],[.5,1]], [[0,0.05],[0.05,0.07],[0.07,0.1],[0.1,0.4]]])
theta_bin_deg.shape

out_kms = theory_kms.get_px_kms(
        zs = z,
        k_kms=k_kms,
        theta_bin_deg=theta_bin_deg,
        return_blob=False
    )

out_kms.shape

# +
for iz, zbin in enumerate(out_kms):
    print("z=", z[iz])
    if iz==0:
        linestyle='dashed'
    else:
        linestyle='solid'
    
    for itheta, theta in enumerate(zbin):
        # print(out[iz][itheta])
        plt.plot(k_kms[iz], out_kms[iz][itheta], label='theta={}, z={}'.format(theta_bin_deg[iz][itheta], z[iz]), linestyle=linestyle)
    
plt.legend()
# -

# Current working version includes option to have different k for different redshift bins, but not for different theta (might want to make this fully generic)
#

# Try in Angstroms

# Load emulator
z = np.array([2.2, 2.4])
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
cosmo = {
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
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)
ffemu = FF_emulator(z, cosmo, cc)
ffemu.kp_Mpc = 1 # pivot point

emu_params = Args()
emu_params.set_baseline()
theory_AA = set_theory(emu_params, ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)

theta_bin_deg = np.asarray([[[0,0.02],[0.02,0.1],[.1,.5],[.5,1]], [[0,0.02],[0.02,0.1],[.1,.5],[.5,1]]])
k_AA = np.array([np.linspace(0.01,.8,100), np.linspace(0.01,.8,100)])
k_AA.shape, z.shape, theta_bin_deg.shape

out_AA = theory_AA.get_px_AA(
        zs = z,
        k_AA=k_AA,
        theta_bin_deg=theta_bin_deg,
        return_blob=False
    )

# +
for iz, zbin in enumerate(out_AA):
    print("z=", z[iz])
    if iz==0:
        linestyle='dashed'
    else:
        linestyle='solid'
    
    for itheta, theta in enumerate(zbin):
        # print(out[iz][itheta])
        plt.plot(k_AA[iz], out_AA[iz][itheta], label='theta={}, z={}'.format(theta_bin_deg[iz][itheta], z[iz]), linestyle=linestyle)
    
plt.legend()

# -

# Show for a case of only 1 z

# Load emulator
z = np.array([2.2])
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
cosmo = {
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
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)
ffemu = FF_emulator(z, cosmo, cc)
ffemu.kp_Mpc = 1 # pivot point

emu_params = Args()
emu_params.set_baseline()
theory_AA = set_theory(emu_params, ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)

theta_bin_deg = np.asarray([[[0,0.02],[0.02,0.1],[.1,.5],[.5,1]]])
k_AA = np.array([np.linspace(0.01,.8,100)])
k_AA.shape, z.shape, theta_bin_deg.shape

out_AA = theory_AA.get_px_AA(
        zs = z,
        k_AA=k_AA,
        theta_bin_deg=theta_bin_deg,
        return_blob=False
    )

# +
for iz, zbin in enumerate(out_AA):
    print("z=", z[iz])
    if iz==0:
        linestyle='dashed'
    else:
        linestyle='solid'
    
    for itheta, theta in enumerate(zbin):
        # print(out[iz][itheta])
        plt.plot(k_AA[iz], out_AA[iz][itheta], label=r'$\theta$={}, z={}'.format(theta_bin_deg[iz][itheta], z[iz]), linestyle=linestyle)
    
plt.legend()
plt.ylim([0,0.18])
# -


