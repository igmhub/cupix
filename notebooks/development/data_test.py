# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: pcross
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

# %% [markdown]
# Set basic args

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

# %% [markdown]
# Load the emulator

# %%
match_lyacolore = True

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
ffemu = FF_emulator(z, cosmo, cc, match_lyacolore=match_lyacolore)
args.emulator = ffemu

# %% [markdown]
# Set more specific arguments

# %% [markdown]
# ## Read in the data from Sindhu

# %%
Px_data = set_Px(args)
for i in range(len(Px_data.thetabin_deg[0])):
    errs = np.sqrt(np.diag(Px_data.cov_Pk_AA[0,i]))
    print(f"theta bin {Px_data.thetabin_deg[0][i]}")
    plt.errorbar(Px_data.k_AA[0], Px_data.Pk_AA[0,i], errs/1147, label='data')
plt.xlim([0,0.8])

# %%
theory_AA = set_theory(args, ffemu, free_parameters=['ln_tau_0'], k_unit='iAA')
theory_AA.set_fid_cosmo(z)

# %%
dir(theory_AA.model_igm)

# %%
out_AA_windowed = theory_AA.get_px_AA(
        zs = z,
        k_AA=Px_data.k_AA,
        theta_bin_deg=Px_data.thetabin_deg,
        window_function=Px_data.window,
        return_blob=False
    )

out_AA_windowless = theory_AA.get_px_AA(
        zs = z,
        k_AA=Px_data.k_AA,
        theta_bin_deg=Px_data.thetabin_deg,
        window_function=None,
        return_blob=False
    )

# %%
color_set = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
for iz, zbin in enumerate(out_AA_windowed):

    print("z=", z[iz])
    if iz==0:
        linestyle='solid'
    else:
        linestyle='dashed'
    linestyle_windowless = 'dotted'
    
    for itheta, theta in enumerate(zbin):
        if itheta>0:
            plt.plot(Px_data.k_AA[iz], Px_data.k_AA[iz]*out_AA_windowed[iz][itheta], label=rf'$\theta=$[{(Px_data.thetabin_deg[iz][itheta]*60)[0]:.0f}, {(Px_data.thetabin_deg[iz][itheta]*60)[1]:.0f}]$\prime$', linestyle=linestyle, color=color_set[itheta % len(color_set)])
            # plt.plot(Px_data.k_AA[iz], Px_data.k_AA[iz]*out_AA_windowless[iz][itheta], linestyle=linestyle_windowless, color=color_set[itheta % len(color_set)])
            # plot the data
            plt.plot(Px_data.k_AA[iz], Px_data.k_AA[iz]*Px_data.Pk_AA[iz, itheta], markersize=3, color=color_set[itheta % len(color_set)], marker='o', linestyle='None')

plt.xlim([0,0.8])
plt.ylim([-0.001,.0125])
plt.ylabel('$k~P(k)~[\AA]$')
plt.xlabel(r'$k~[\AA^{-1}]$')
if match_lyacolore:
    title = 'Lyacolore data compared to small-scale-fitted theory'
else:
    title = 'Lyacolore data compared to fiducial theory'

# custom legend
#get legend handles, labels
handles, labels = plt.gca().get_legend_handles_labels()
# add solid black line for windowed theory
handles.append(plt.Line2D([], [], color='black', linestyle='solid', label='windowed theory'))
# add dotted black line for windowless theory
# handles.append(plt.Line2D([], [], color='black', linestyle='dotted', label='windowless theory'))
# add dot markers for data
handles.append(plt.Line2D([], [], color='black', marker='o', linestyle='None', label='LyaCoLoRe mock data'))
plt.legend(handles=handles, loc='upper right', fontsize=12)

plt.title(title)


# %% [markdown]
# Version with residuals

# %%
color_set = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
fig, ax = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6), sharex=True)
min_itheta = 2
for iz, zbin in enumerate(out_AA_windowed):

    print("z=", z[iz])
    if iz==0:
        linestyle='solid'
    else:
        linestyle='dashed'
    linestyle_windowless = 'dotted'
    
    for itheta, theta in enumerate(zbin):
        if itheta>min_itheta:
            is_nonzero = np.abs(out_AA_windowed[iz][itheta]) > 1e-3
            # plot the theory
            ax[0].plot(Px_data.k_AA[iz], Px_data.k_AA[iz]*out_AA_windowed[iz][itheta], label=rf'$\theta=$[{(Px_data.thetabin_deg[iz][itheta]*60)[0]:.0f}, {(Px_data.thetabin_deg[iz][itheta]*60)[1]:.0f}]$\prime$', linestyle=linestyle, color=color_set[itheta % len(color_set)])
            ax[0].plot(Px_data.k_AA[iz], Px_data.k_AA[iz]*out_AA_windowless[iz][itheta], linestyle=linestyle_windowless, color=color_set[itheta % len(color_set)])
            # plot the data
            errs = np.sqrt(np.diag(Px_data.cov_Pk_AA[iz,itheta]))/1147
            ax[0].errorbar(Px_data.k_AA[iz], Px_data.k_AA[iz]*Px_data.Pk_AA[iz, itheta], yerr=Px_data.k_AA[iz]*errs, markersize=3, color=color_set[itheta % len(color_set)], marker='o', linestyle='None')
            # plot the residuals
            ax[1].plot(Px_data.k_AA[iz], 
                       (Px_data.Pk_AA[iz, itheta] - out_AA_windowed[iz][itheta])/errs,
                       linestyle=linestyle, color=color_set[itheta % len(color_set)])
            ax[1].plot(Px_data.k_AA[iz], 
                       (Px_data.Pk_AA[iz, itheta] - out_AA_windowless[iz][itheta])/errs,
                       linestyle=linestyle_windowless, color=color_set[itheta % len(color_set)])
if min_itheta==1:
    ax[0].set_ylim([-0.001,.005])
    ax[1].set_ylim([-3.5, 3.5])
    ax[1].set_xlim([0.01,0.6])
    ax[0].legend(handles=handles, loc='upper right', fontsize=12, ncol=1)

elif min_itheta==2:
    ax[0].set_ylim([-0.0005,.0032])
    ax[1].set_ylim([-4, 4])
    ax[1].set_xlim([0.01,0.6])
    ax[0].legend(handles=handles, loc='upper right', fontsize=12, ncol=1)
else:
    ax[0].set_ylim([-0.001,.0125])
    ax[1].set_ylim([-10, 6])
    ax[1].set_xlim([0.01,1.0])
    ax[0].legend(handles=handles, loc='upper left', fontsize=10, ncol=1)

ax[0].set_ylabel(r'$k_\parallel P(k_\parallel)~[\AA]$')

if match_lyacolore:
    title = 'Mock measurements compared to small-scale-fitted theory (z=2)'
else:
    title = 'Mock measurements compared to fiducial theory (z=2)'
ax[1].set_ylabel(r'$\dfrac{(\rm{Data} - \rm{theory})}{ 1\sigma~\rm{error}}$')
ax[1].set_xlabel(r'$k_\parallel~[\AA^{-1}]$')


# custom legend
#get legend handles, labels
handles, labels = ax[0].get_legend_handles_labels()
# add solid black line for windowed theory
handles.append(plt.Line2D([], [], color='black', linestyle='solid', label='ForestFlow prediction, window'))
# add dotted black line for windowless theory
handles.append(plt.Line2D([], [], color='black', linestyle='dotted', label='ForestFlow prediction, no window'))
# add dot markers for data
handles.append(plt.Line2D([], [], color='black', marker='o', linestyle='None', label='IFAE-QL mock data'))


plt.suptitle(title)

# %%
