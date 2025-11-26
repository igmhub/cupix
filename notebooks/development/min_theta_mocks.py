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
# ## Finding the minimum $\theta$ value to apply for mock tests

# %%
import sys
import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
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
# emu_params = Args()
# emu_params.set_baseline()
# print(emu_params)

# %%
theory_AA = set_theory(ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu

# %% [markdown]
# Set the data

# %%
MockData_full = Px_Lyacolore("binned_out_trucont_px-zbins_2-thetabins_10_res.hdf5", kmax_cut_AA=1)

# %% [markdown]
# Set the Likelihood

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

# %% [markdown]
# Set the bias values to test

# %%
bias_vals = np.linspace(-0.114,-0.123,10)

# %%
bias_vals

# %%
MockData_full.theta_min_A_arcmin[:]

# %%
# find the deltachi2 corresponding to 1 sigma
sigma=1
cdf_1sig = scipy.stats.chi2.cdf(sigma**2,1)
chi_squared_1sig = scipy.stats.chi2.ppf( cdf_1sig, 1)
chi_squared_1sig

# %%
best_chi2s = []
best_probs = []
best_biases= []
ndof = []
thetamin = []
all_chi2s_per_thetamin = []
error_estimates = []
for min_cut in MockData_full.theta_min_A_arcmin[2:]:
    print("Testing a minimum cut of ", min_cut, " arcmin")
    thetamin.append(min_cut)
    MockData = Px_Lyacolore("binned_out_trucont_px-zbins_2-thetabins_10_res.hdf5", theta_min_cut_arcmin=min_cut, kmax_cut_AA=1)
    Like = Likelihood(MockData, theory_AA, iz_choice=0, like_params = likelihood_params)
    chi2_per_param = []
    bestfit_this_min = None
    bestprob_this_min = None
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
            if param.name=='bias':
                print("Testing bias of", param.value)

        chi2_i = Like.get_chi2([lp.value for lp in likelihood_params])
        chi2_per_param.append(chi2_i)
        prob = Like.fit_probability([lp.value for lp in likelihood_params], n_free_p=1)
        if i==0 or chi2_i < np.min(chi2_per_param[:-1]):
            best_bias = bias_vals[i]
            print(f"New best fit found with chi2={chi2_i:.2f} at bias={best_bias:.4f}")
            bestfit_this_min = chi2_i
            bestprob_this_min = prob # save the best probability, overwriting if the chi2 is lower than before
        if i==0:
            ndof.append(Like.ndeg(0))
    best_chi2s.append(bestfit_this_min)
    best_probs.append(bestprob_this_min)
    best_biases.append(best_bias)
    print("End of loop for this min cut, best_chi2s is now ", best_chi2s)
    all_chi2s_per_thetamin.append(chi2_per_param)
        # if this is the best-fit so far, save the theory prediction
    # estimate error on bias by finding delta chi2 = 1
    chi2s = np.array(chi2_per_param)
    min_chi2 = np.min(chi2s)
    delta_chi2s = chi2s - min_chi2
    # find where delta chi2 crosses chi_squared_1sig
    try:
        lower_idx = np.where(delta_chi2s < chi_squared_1sig)[0][0]
        upper_idx = np.where(delta_chi2s < chi_squared_1sig)[0][-1]
        bias_lower = bias_vals[lower_idx]
        bias_upper = bias_vals[upper_idx]
        error_estimate = (bias_upper - bias_lower) / 2
    except:
        error_estimate = np.nan
    print("Estimated error on bias: ", error_estimate)

# %%
plt.plot(thetamin, best_chi2s, marker='o')
plt.ylabel(r"Best $\chi^2$")
plt.xlabel(r"$\theta_{\min}$ [arcmin]")

# %%
plt.plot(thetamin, best_probs, marker='o')
plt.ylabel("Pprobability of fit for best-fit model")
plt.xlabel(r"$\theta_{\min}$ [arcmin]")

# %%
plt.plot(thetamin, best_biases, marker='o')
plt.ylabel("Best bias")
plt.axhline(-0.115, label=r'Best-fit bias from $\xi$ fits', color='gray', linestyle='--')

plt.xlabel(r"$\theta_{\min}$ [arcmin]")
plt.legend()

# %%
colors = plt.cm.tab10(np.linspace(0,1,len(all_chi2s_per_thetamin)))

for i, fits in enumerate(all_chi2s_per_thetamin):
    if i>1:
        plt.plot(bias_vals, fits-np.amin(fits), marker='o', label=rf"$\theta_{{min}}$={thetamin[i]:.1f} arcmin", color=colors[i])
        plt.axvline(best_biases[i]+.0005*1/(i+1), color=colors[i], linestyle='--')
plt.axvline(-0.115, label=r'Best-fit bias from $\xi$ fits', color='gray', linestyle='--')
plt.ylabel(r"$\Delta \chi^2$")
plt.xlabel("Bias")
plt.legend()


# %%
likelihood_params_best = likelihood_params
for j, param in enumerate(likelihood_params):
        if param.name=='bias':
            # replace the first element of the likelihood_params list
            likelihood_params_best[j] = LikelihoodParameter(
                name='bias',
                min_value=-2.0,
                max_value=1.0,
                value=[best_biases[-1]]
                )

# %%
Like.plot_px(0, likelihood_params_best, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None)

# %%
# set the likelihood parameters as the Arinyo params with some fiducial values

likelihood_params = []
likelihood_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=1.0,
    value=[-0.117]
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


# %%
beta_vals = np.linspace(1.4, 1.6, 10) # Laura's fit 1.55,-0.120,8

best_chi2s_vb = []
best_probs_vb = []
best_betas = []
ndof_vb = []
thetamin_vb = []
all_chi2s_per_thetamin_vb = []
for min_cut in MockData_full.theta_min_A_arcmin[12:]:
    print("Testing a minimum cut of ", min_cut, " arcmin")
    thetamin_vb.append(min_cut)
    MockData = Px_Lyacolore("binned_out_trucont_px-zbins_2-thetabins_18.hdf5", theta_min_cut_arcmin=min_cut, kmax_cut_AA=1)
    Like = Likelihood(MockData, theory_AA, iz_choice=0, like_params = likelihood_params)
    chi2_per_param = []
    bestfit_this_min = None
    bestprob_this_min = None
    for i in range(len(beta_vals)):
        for j, param in enumerate(likelihood_params):
            if param.name=='beta':
                # replace the first element of the likelihood_params list
                likelihood_params[j] = LikelihoodParameter(
                    name='beta',
                    min_value=-2.0,
                    max_value=1.0,
                    value=[beta_vals[i]]
                    )
        # make sure it worked
        for param in likelihood_params:
            if param.name=='beta':
                print("Testing beta of", param.value)

        chi2_i = Like.get_chi2([lp.value for lp in likelihood_params])
        chi2_per_param.append(chi2_i)
        prob = Like.fit_probability([lp.value for lp in likelihood_params], n_free_p=1)
        if i==0 or chi2_i < np.min(chi2_per_param[:-1]):
            best_beta = beta_vals[i]
            print(f"New best fit found with chi2={chi2_i:.2f} at beta={best_beta:.4f}")
            bestfit_this_min = chi2_i
            bestprob_this_min = prob # save the best probability, overwriting if the chi2 is lower than before
        if i==0:
            ndof.append(Like.ndeg(0))
    best_chi2s_vb.append(bestfit_this_min)
    best_probs_vb.append(bestprob_this_min)
    best_betas.append(best_beta)
    print("End of loop for this min cut, best_chi2s is now ", best_chi2s)
    all_chi2s_per_thetamin_vb.append(chi2_per_param)
        # if this is the best-fit so far, save the theory prediction


# %%
plt.plot(thetamin_vb, best_chi2s_vb, marker='o')
plt.ylabel(r"Best $\chi^2$")
plt.xlabel(r"$\theta_{\min}$ [arcmin]")

# %%
plt.plot(thetamin_vb, best_probs_vb, marker='o')
plt.ylabel("Best probability")
plt.xlabel(r"$\theta_{\min}$ [arcmin]")

# %%
plt.plot(thetamin_vb, best_betas, marker='o')
plt.ylabel("Best beta")
plt.axhline(1.55, label=r'Best-fit beta from $\xi$ fits', color='gray', linestyle='--')

plt.xlabel(r"$\theta_{\min}$ [arcmin]")
plt.legend()

# %%
colors = ['blue', 'orange', 'green', 'red', 'brown', 'gray', 'olive', 'cyan']

for i, fits in enumerate(all_chi2s_per_thetamin_vb):
    if i>1:
        plt.plot(beta_vals, fits-np.amin(fits), marker='o', label=rf"$\theta_{{min}}$={thetamin_vb[i]:.1f} arcmin", color=colors[i])
        plt.axvline(best_betas[i]+.0007*1/i, color=colors[i], linestyle='--')
plt.ylabel(r"$\Delta \chi^2$")
plt.xlabel("Beta")
plt.legend()


# %%
likelihood_params_best = likelihood_params
for j, param in enumerate(likelihood_params):
        if param.name=='bias':
            # replace the first element of the likelihood_params list
            likelihood_params_best[j] = LikelihoodParameter(
                name='bias',
                min_value=-2.0,
                max_value=1.0,
                value=[best_biases[-5]]
            )
        elif param.name=='beta':
            # replace the first element of the likelihood_params list
            likelihood_params_best[j] = LikelihoodParameter(
                name='beta',
                min_value=-2.0,
                max_value=1.0,
                value=[best_betas[-5]]
            )

# %%
thetamin_vb[-5]

# %%
MockData = Px_Lyacolore("binned_out_trucont_px-zbins_2-thetabins_18.hdf5", theta_min_cut_arcmin=thetamin_vb[-5], kmax_cut_AA=1)
Like = Likelihood(MockData, theory_AA, iz_choice=0, like_params = likelihood_params)

# %%
for value in [lp.value for lp in likelihood_params_best]:
    print(value)

# %%
Like.plot_px(0, likelihood_params_best, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None)

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
# Version with the minimizer

# %%

# %%
best_chi2s = []
best_probs = []
best_biases= []
best_betas= []
ndof = []
thetamin = []
all_chi2s_per_thetamin = []
error_estimates = []

for min_cut in MockData.theta_min_A_arcmin[2:]:
    print("Trying min cut of ", min_cut, " arcmin")
    MockData_cut = Px_Lyacolore("binned_out_trucont_px-zbins_2-thetabins_10_res.hdf5", kmax_cut_AA=1, theta_min_cut_arcmin=min_cut)
    like = Likelihood(MockData_cut, theory_AA, free_param_names=["bias", "beta", "q1", "kvav", "kp"], iz_choice=0, like_params=like_params, verbose=False)
    mini = IminuitMinimizer(like, like_params, verbose=False)
    mini.minimize()

    ini_values = []
    final_values = []
    hesse_errs = []
    priors = []
    for param in like.free_param_names:
        for param2 in like_params:
            if param == param2.name:
                ini_values.append(param2.ini_value)
                priors.append(param2.Gauss_priors_width)
                break

        for param3 in mini.minimizer.params:
            print(param3)
            if param == param3.name:
                final_values.append(mini.minimizer.params[param3.name].value)
                hesse_errs.append(mini.minimizer.params[param3.name].error)
                break

    all_values = []
    
    for i, lp in enumerate(like_params):
        if lp.name in like.free_param_names:
            index = like.free_param_names.index(lp.name)
            all_values.append(final_values[index])
        else:
            all_values.append(lp.ini_value)
    best_biases.append(all_values[0])
    best_betas.append(all_values[1])
    
    
    prob = like.fit_probability(all_values, n_free_p=5)
    best_probs.append(prob)
    
    

# %%
plt.plot(MockData.theta_min_A_arcmin[2:], best_probs, 'o-')
plt.xlabel("Theta min cut [arcmin]")
plt.ylabel("Best fit probability")


# %%
plt.plot(MockData.theta_min_A_arcmin[2:], best_biases, 'o-', label='bias')
# plt.plot(MockData.theta_min_A_arcmin[2:], best_betas, 'o-', label='beta')
plt.xlabel("Theta min cut [arcmin]")
plt.ylabel("Best fit bias")
plt.axhline(-0.115, label=r'Best-fit bias from $\xi$ fits', color='gray', linestyle='--')
plt.legend()

# %%
# plt.plot(MockData.theta_min_A_arcmin[2:], best_biases, 'o-', label='bias')
plt.plot(MockData.theta_min_A_arcmin[2:], best_betas, 'o-', label='beta')
plt.xlabel("Theta min cut [arcmin]")
plt.ylabel("Best fit beta")
plt.axhline(1.55, label=r'Best-fit beta from $\xi$ fits', color='gray', linestyle='--')
plt.legend()


# %%
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
# find transvers comoving Mpc for 20'
cosmo.kpc_comoving_per_arcmin(2.6).to(u.Mpc/u.arcmin) * 20 * u.arcmin

# %%
which_theta_min = 7
min_cut = MockData.theta_min_A_arcmin[which_theta_min]
print("Trying min cut of ", min_cut, " arcmin")
import copy
like_params_to_plot = copy.deepcopy(like_params)
for i, lp in enumerate(like_params_to_plot):
    if lp.name == 'bias':
        lp.value = best_biases[which_theta_min]
    if lp.name == 'beta':
        lp.value = best_betas[which_theta_min]
MockData_cut_plot = Px_Lyacolore("binned_out_trucont_px-zbins_2-thetabins_10_res.hdf5", kmax_cut_AA=1, theta_min_cut_arcmin=min_cut)
like_plot = Likelihood(MockData_cut_plot, theory_AA, free_param_names=["bias", "beta", "q1", "kvav", "kp"], iz_choice=0, like_params=like_params_to_plot, verbose=False)
like_plot.plot_px(0, like_params_to_plot, every_other_theta=False, show=True, theorylabel=None, datalabel=None, plot_fname=None, ylim=[-.0005, .0005], xlim=[-0.002, .5])



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
