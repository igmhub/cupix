# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import sys
import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from lace.cosmo import camb_cosmo
import matplotlib.pyplot as plt
from cupix.likelihood.pipeline import set_Px
# set text size for plot
plt.rcParams.update({'font.size': 14})

# %%
args = Args(data_label='lyacolore')
args.set_baseline()
args.emu_cov_factor = None
args.use_pk_smooth = False
args.rebin_k = 1
args.n_steps = 3

args.n_tau = 1
args.n_sigT = 0
args.n_gamma = 0
args.n_kF = 0
args.n_d_dla = 0
args.n_s_dla = 0
args.n_sn = 0
args.n_agn = 0
args.n_res = 0
args.nwalkers = 25

# %%
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
args.emulator = ffemu

# %%
theory_AA = set_theory(args, ffemu, free_parameters=['ln_tau_0'], k_unit='iAA')
theory_AA.set_fid_cosmo(z)

# %%
k_AA = np.array([np.linspace(0.01, 0.8, 100)])
thetabin_deg = np.array([[0.,         0.01666667],
[0.01666667, 0.08333333],
[0.08333333, 0.16666667],
[0.16666667, 0.25      ],
[0.25,       0.33333333],
[0.33333333, 0.83333334]])

# %%
out_AA_windowless = theory_AA.get_px_AA(
        zs = z,
        k_AA=k_AA,
        theta_bin_deg=thetabin_deg,
        window_function=None,
        return_blob=False
    )

# %%
color_set = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
for iz, zbin in enumerate(out_AA_windowless):

    print("z=", z[iz])
    if iz==0:
        linestyle='solid'
    else:
        linestyle='dashed'
    linestyle_windowless = 'dotted'
    
    for itheta, theta in enumerate(zbin):
        if itheta>0:
            
            plt.plot(k_AA[iz], k_AA[iz]*out_AA_windowless[iz][itheta], linestyle=linestyle_windowless, color=color_set[itheta % len(color_set)])


plt.xlim([0,0.8])
plt.ylim([-0.001,.0125])
plt.ylabel('$k~P(k)~[\AA]$')
plt.xlabel(r'$k~[\AA^{-1}]$')

# custom legend
#get legend handles, labels
handles, labels = plt.gca().get_legend_handles_labels()

# add dotted black line for windowless theory
handles.append(plt.Line2D([], [], color='black', linestyle='dotted', label='windowless theory'))
# add dot markers for data

plt.legend(handles=handles, loc='upper right', fontsize=12)




# %%
