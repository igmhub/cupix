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

# %% [markdown]
# ## Likelihood example notebook

# %%
import sys
import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
# %load_ext autoreload
# %autoreload 2
from lace.cosmo import camb_cosmo, fit_linP
import matplotlib.pyplot as plt
import h5py
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta
from cupix.px_data.data_lyacolore import Px_Lyacolore
import scipy

# %% [markdown]
# Set the redshifts and fiducial cosmology

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

# %% [markdown]
# Set up the emulator

# %%
ffemu = FF_emulator(z, fid_cosmo, cc)
ffemu.kp_Mpc = 1 # set pivot point

# %% [markdown]
# Set up the theory with some default parameters (this part inherits some old behavior from cup1d that we may change later)

# %%
emu_params = Args()
emu_params.set_baseline()
print(emu_params)

# %%
theory_AA = set_theory(emu_params, ffemu, free_parameters=['ln_tau_0'], k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu

# %% [markdown]
# Set the data

# %%
MockData = Px_Lyacolore("binned_out_truecont_px-zbins_2-thetabins_9.hdf5")

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values

likelihood_params = []
likelihood_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=1.0,
    value=[-0.115]
    ))
likelihood_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.0,
    max_value=2.0,    
    value = [1.55]
    ))
likelihood_params.append(LikelihoodParameter(
    name='q1',
    min_value=0.0,
    max_value=1.0,
    value = [0.1112]
    ))
likelihood_params.append(LikelihoodParameter(
    name='kvav',
    min_value=0.0,
    max_value=1.0,
    value = [0.0001**0.2694]
    ))
likelihood_params.append(LikelihoodParameter(
    name='av',
    min_value=0.0,
    max_value=1.0,
    value = [0.2694]
    ))
likelihood_params.append(LikelihoodParameter(
    name='bv',
    min_value=0.0,
    max_value=1.0,
    value = [0.0002]
    ))
likelihood_params.append(LikelihoodParameter(
    name='kp',
    min_value=0.0,
    max_value=1.0,
    value = [0.5740]
    ))


# likelihood_params = []
# likelihood_params.append(LikelihoodParameter(
#     name='Delta2_p',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='n_p',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='mF',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='gamma',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='kF_Mpc',
#     min_value=-1.0,
#     max_value=1.0,
#     ))
# likelihood_params.append(LikelihoodParameter(
#     name='sigT_Mpc',
#     min_value=-1.0,
#     max_value=1.0,
#     ))

# %%
Px_theory_bestfit = []
bias_vals = np.linspace(-0.1135,-0.118,20)
chi2_per_param = []
redchi2_per_param = []
z_ind = 0 # do this independently per redshift bin
for i in range(len(bias_vals)):
    for j, param in enumerate(likelihood_params):
        if param.name=='bias':
            # replace the first element of the likelihood_params list
            likelihood_params[j] = LikelihoodParameter(
                name='bias',
                min_value=-2.0,
                max_value=1.0,
                value=[bias_vals[i]]
                )
    # make sure it worked
    for param in likelihood_params:
        print(param.name, param.value)
    
    # save results
    chi2_list = []
    Px_theory = []
    Px_data_toplot = []
    Px_data_errs = []
    for theta_A_ind in range(len(MockData.theta_min_A))[6:]:
        
        ind_in_theta = MockData.B_A_a.astype(bool)[theta_A_ind,:]
        theta_a_inds = np.where(ind_in_theta)[0]
        # collect the Px for all small theta bins that correspond to this large theta bin
        Px_aMZ_list = []
        V_aMZ_list = []
        for t_a in theta_a_inds:
            print("small theta bin", MockData.theta_min_a[t_a], MockData.theta_max_a[t_a])
            z    = MockData.z[z_ind]
            k_AA = MockData.k_m[1:]
            theta_a_arcmin = [(MockData.theta_min_a[t_a] + MockData.theta_max_a[t_a])/2.]
            print(f"Finding raw theory prediction for z={z:.2f}, theta_a={theta_a_arcmin[0]:.2f} arcmin")
            # first, get the theory prediction
            Px_amZ = theory_AA.get_px_AA(
                zs = [z],
                k_AA=[k_AA],
                theta_arcmin=[theta_a_arcmin],
                like_params=likelihood_params,
                return_blob=False
            )
            # next, apply the window matrix
            U_aMnZ = MockData.U_ZaMn[z_ind,t_a]
            Px_aMZ = convolve_window(U_aMnZ[:,1:],Px_amZ[0].T)
            Px_aMZ_list.append(Px_aMZ.flatten())
            V_aMZ_list.append(MockData.V_ZaM[z_ind,t_a])

        Px_AMZ = rebin_theta(np.asarray(V_aMZ_list), np.array(Px_aMZ_list))
        Px_theory.append(Px_AMZ.flatten())
        Px_data_errs.append(np.sqrt(np.diag(MockData.cov_ZAM[z_ind, theta_A_ind, :, :])))
        

        diff   = np.reshape(MockData.Px_ZAM[z_ind,theta_A_ind, :] - Px_AMZ.flatten(), (-1,1))
        icov_Px = np.linalg.inv(MockData.cov_ZAM[z_ind, theta_A_ind, :, :])
        chi2_z = diff.T @ icov_Px @ diff
        dof = len(diff) - 1
        chi2_red = chi2_z/dof
        chi2_list.append(chi2_z[0][0])
    
    # combined chi2
    chi2_list = np.array(chi2_list)
    chi2_total = np.sum(chi2_list)
    ndof_total = np.sum([len(MockData.Px_ZAM[z_ind,theta_A_ind, :]) - 1 for theta_A_ind in range(len(MockData.theta_min_A))])
    chi2_per_param.append(chi2_total)
    redchi2_per_param.append(chi2_total/ndof_total)

    # if this is the best-fit so far, save the theory prediction
    if i==0 or chi2_total < np.min(chi2_per_param[:-1]):
        Px_theory_bestfit = Px_theory
        best_bias = bias_vals[i]
        print(f"New best fit found with chi2={chi2_total:.2f} at bias={best_bias:.4f}")


# %%
delta_chi2 = np.array(chi2_per_param)-min(chi2_per_param)
plt.plot(bias_vals, delta_chi2, marker='o')
plt.xlabel("bias")
plt.ylabel(r"$\Delta \chi^2$")
plt.title("Profile likelihood for bias")

# find the region corresponding to 5 sigma
sigma=5
cdf_5sig = scipy.stats.chi2.cdf(sigma**2,1)
chi_squared_5sig = scipy.stats.chi2.ppf( cdf_5sig, 1)
# delta_chi2_5sig = chi_squared_5sig - min(chi2_per_param)
print(f"5 sigma corresponds to delta chi2 = {chi_squared_5sig:.2f}")
where_5sig = np.where(delta_chi2 - chi_squared_5sig < 0.5)[0]
plt.axhline(chi_squared_5sig, color='red', linestyle='--', label='5 sigma threshold')
plt.show()
plt.clf()

# %%
# plot the best-fit value

plt.rcParams.update({'font.size': 14})
# plot all theta on one, easily distinguishable colors
colors = plt.cm.tab10(np.linspace(0,1,len(MockData.theta_min_A)))
k = MockData.k_M_edges[:-1]

for A in range(len(MockData.theta_min_A))[6:]:
    theory_index = A-6
    errors = np.diag(MockData.cov_ZAM[z_ind, A, :, :])**0.5
    plt.errorbar(k, MockData.Px_ZAM[z_ind, A, :]*k, errors*k, label=r'$\theta_A={:.2f}^\prime$'.format(0.5*(MockData.theta_min_A[A]+MockData.theta_max_A[A])), color=colors[A], linestyle='none', marker='o')
    plt.plot(k, np.array(Px_theory_bestfit[theory_index])*k, color=colors[A], linestyle='solid')
    plt.legend()
    plt.xlabel(r'$k [\AA^{-1}]$')
    plt.ylabel(r'$k P_\times$')
plt.xlim([-0.001,0.6])
plt.ylim([-.0005, 0.003])
# custom legend
#get legend handles, labels
handles, labels = plt.gca().get_legend_handles_labels()
# add solid black line for windowed theory
handles.append(plt.Line2D([], [], color='black', linestyle='solid', label='ForestFlow prediction, windowed'))
# add dash markers for data
handles.append(plt.Line2D([], [], color='black', marker='o', linestyle='none', label='IFAE-QL mock data, true-continuum'))

plt.legend(handles=handles, loc='upper right', fontsize='small')

# %%
best_bias

# %%
# plot the best-fit value

plt.rcParams.update({'font.size': 14})
# plot all theta on one, easily distinguishable colors
colors = plt.cm.tab10(np.linspace(0,1,len(MockData.theta_min_A)))
k = MockData.k_M_edges[:-1]

for A in range(len(MockData.theta_min_A))[8:]:
    theory_index = A-6
    errors = np.diag(MockData.cov_ZAM[z_ind, A, :, :])**0.5
    plt.errorbar(k, MockData.Px_ZAM[z_ind, A, :]*k, errors*k, label=r'$\theta_A={:.2f}^\prime$'.format(0.5*(MockData.theta_min_A[A]+MockData.theta_max_A[A])), color=colors[A], linestyle='none', marker='o')
    plt.plot(k, np.array(Px_theory_bestfit[theory_index])*k, color=colors[A], linestyle='solid')
    plt.legend()
    plt.xlabel(r'$k [\AA^{-1}]$')
    plt.ylabel(r'$k P_\times$')
plt.xlim([-0.001,0.6])
plt.ylim([-.00005, 0.0002])
# custom legend
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(plt.Line2D([], [], color='#17becf', linestyle='solid', label='ForestFlow prediction, windowed'))
plt.legend(handles=handles, loc='upper right', fontsize='small')

# %%
import scipy.stats
import math

#stand deviations to calculate
sigma = [   1.0,
            math.sqrt(scipy.stats.chi2.ppf(0.8,1)),
            math.sqrt(scipy.stats.chi2.ppf(0.9,1)),
            math.sqrt(scipy.stats.chi2.ppf(0.95,1)),
            2.0,
            math.sqrt(scipy.stats.chi2.ppf(0.99,1)),
            3.0,
            math.sqrt(scipy.stats.chi2.ppf(0.999,1)),
            4.0,
            5.0   ]

#confidence intervals these sigmas represent:
conf_int = [ scipy.stats.chi2.cdf( s**2,1) for s in sigma ]

#degrees of freedom to calculate
dof = range(1,5)

print("sigma     \t" + "\t".join(["%1.2f"%(s) for s in sigma]))
print("conf_int  \t" + "\t".join(["%1.5f%%"%(100*ci) for ci in conf_int]))
print("p-value   \t" + "\t".join(["%1.5f"%(1-ci) for ci in conf_int]))

for d in dof:
    chi_squared = [ scipy.stats.chi2.ppf( ci, d) for ci in conf_int ]
    print("chi2(k=%d)\t"%d + "\t" .join(["%1.2f" % c for c in chi_squared]))

# %% [markdown]
# Do a Silicon test
#

# %%
likelihood_params.append(LikelihoodParameter(
    name='bias_SiIII',
    min_value=-1.0,
    max_value=1.0,
    value=[-9.79e-3]
    ))
likelihood_params.append(LikelihoodParameter(
    name='beta_SiIII',
    min_value=0.0,
    max_value=2.0,    
    value = [1.55]
    ))
likelihood_params.append(LikelihoodParameter(
    name='k_p_SiIII',
    min_value=0.0,
    max_value=1.0,
    value = [0.5740]
    ))

# %%
Px_amZ_with_silicon = theory_AA.get_px_AA(
    zs = [z],
    k_AA=[k_AA],
    theta_arcmin=[theta_a_arcmin],
    like_params=likelihood_params,
    return_blob=False,
    add_silicon=True,
    
)
Px_amZ_without_silicon = theory_AA.get_px_AA(
    zs = [z],
    k_AA=[k_AA],
    theta_arcmin=[theta_a_arcmin],
    like_params=likelihood_params,
    return_blob=False,
    add_silicon=False,
    
)

# %%
plt.plot(k_AA, Px_amZ_without_silicon.flatten(), label='without SiIII')
plt.plot(k_AA, Px_amZ_with_silicon.flatten(), label='with SiIII')
plt.xlim([0,0.3])
plt.legend()

# %%
# check comoving size at this redshift
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
(cosmo.kpc_comoving_per_arcmin(2.4)*32.5*u.arcmin).to(u.Mpc)

# %% [markdown]
#
