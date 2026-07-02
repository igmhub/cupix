# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fitting continuum distortion from the stack of many mocks

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
# %load_ext autoreload
# %autoreload 2

# %%
from lace.cosmo import cosmology
from cupix.px_data.data_DESI_DR2 import DESI_DR2
from cupix.likelihood.theory import Theory
from cupix.likelihood.likelihood import Likelihood
from cupix.inference.free_parameter import FreeParameter
from cupix.inference.posterior import Posterior
from cupix.inference.minimize_posterior import Minimizer

# %% [markdown]
# ### Read the Px from the stack of 50 mocks

# %%
# ls /global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/uncontaminated/

# %%
# drop bals and dlas
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
# true continuum
true_fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
true_data = DESI_DR2(config={'data_file':true_fname, 'kM_max_cut_AA':0.3, 'km_max_cut_AA':0.35, 'theta_min_cut_arcmin':20.0})
# # uncontaminated
unco_fname = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
unco_data = DESI_DR2(config={'data_file':unco_fname, 'kM_max_cut_AA':0.3, 'km_max_cut_AA':0.35, 'theta_min_cut_arcmin':20.0})

# contaminated, drop BALs and DLAs
co_fname = mockdir + "contaminated/contaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
co_data = DESI_DR2(config={'data_file':co_fname, 'kM_max_cut_AA':0.3, 'km_max_cut_AA':0.35, 'theta_min_cut_arcmin':20.0})

# contaminated, drop BALs and DLAs
co_fname_nobal_nodla = mockdir + "contaminated/drop_BALs_and_DLAs/contaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
co_data_nobal_nodla = DESI_DR2(config={'data_file':co_fname_nobal_nodla, 'kM_max_cut_AA':0.3, 'km_max_cut_AA':0.35, 'theta_min_cut_arcmin':20.0})

# %%
from matplotlib.lines import Line2D

# check out the differences
# get the central value of each redshift bin, of length Nz
zs = co_data.z
# get a 1D array of central values of the measured k bins, of length Nk_M
k_M = co_data.k_M_centers_AA
# get two 1D arrays with the edges of each theta bin, of length Nt_A each
theta_A_min = co_data.theta_min_A_arcmin
theta_A_max = co_data.theta_max_A_arcmin
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = co_data.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_m={Nk_m}")
# rebinned values
Nz, Nt_A, Nk_M = co_data.Px_ZAM.shape
print(f"rebinned values: Nz={Nz}, Nt_A={Nt_A}, Nk_M={Nk_M}")

def plot_theta_bin(data, iz, it_M, ls = 'solid'):
    colors = plt.get_cmap('tab10')
    label = r"${:.2f}' < \theta < {:.2f}'$".format(theta_A_min[it_M], theta_A_max[it_M])
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it_M]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_M]))
#    print(len(k_M), len(Px), len(sig_Px))
    plt.errorbar(k_M, Px, sig_Px, label=label, ls=ls, color=colors(it_M))
def plot_z_bin(data, iz, its_M, ls= 'solid', show_legend=True):
    
    for it_M in its_M[::3]:
        plot_theta_bin(data=data, iz=iz, it_M=it_M, ls=ls)
    plt.title('DESI DR2 at z={:.1f}'.format(zs[iz]))
    if show_legend:
        plt.legend(fontsize=10)
    plt.xlabel(r'$k_\parallel$ [1/A]')
    plt.ylabel(r'$P_\times(\theta, k_\parallel)$ [A]')
#    print(len(k_M), len(Px), len(sig_Px))
    if show_legend:
                
        # Get existing legend entries
        handles, labels = plt.gca().get_legend_handles_labels()

        # Add line-style entries
        handles.extend([
            Line2D([0], [0], color='k', linestyle='-',  label='true-cont'),
            Line2D([0], [0], color='k', linestyle='--', label='uncontam'),
            Line2D([0], [0], color='k', linestyle=':',  label='contam'),
            Line2D([0], [0], color='k', linestyle='dashdot',  label='contam_drop'),
        ])

        plt.legend(handles=handles)
for iz in range(4):
    
    plt.figure(figsize=[8,3])
    plot_z_bin(true_data, iz=iz, its_M=range(Nt_A))
    plot_z_bin(unco_data, iz=iz, its_M = range(Nt_A), ls='dashed', show_legend=False)
    plot_z_bin(co_data, iz=iz, its_M = range(Nt_A), ls='dotted', show_legend=False)
    plot_z_bin(co_data_nobal_nodla, iz=iz, its_M = range(Nt_A), ls='dashdot', show_legend=False)
    

# %% [markdown]
# ### Start by fitting bias/beta from the stack of true-continuum mocks (one-z at a time)

# %%
# define fiducial cosmo
cosmo = cosmology.Cosmology()
# starting point for Lya bias parameters in mocks
default_lya_model = 'best_fit_arinyo_from_colore'

# %%
# set free parameters
bias = FreeParameter(
    name='bias',
    min_value=-0.5,
    max_value=-0.01,
    ini_value=-0.15,
    delta=0.01,   
)
beta = FreeParameter(
    name='beta',
    min_value=0.1,
    max_value=5.0,
    ini_value=1.5,
    delta=0.1,
)
free_params = [bias, beta]
for par in free_params:
    print(par.name, par.ini_value)

# %%
true_minis = []
for iz, z in enumerate(true_data.z): 
    theory = Theory(z=z, fid_cosmo=cosmo, 
                    config={'verbose': True, 'default_lya_model': default_lya_model, 'include_continuum': False})
    # reset bias/beta 
    assert free_params[0].name == 'bias'
    free_params[0].ini_value = theory.lya_model.default_lya_params['bias']
    assert free_params[1].name == 'beta'
    free_params[1].ini_value = theory.lya_model.default_lya_params['beta'] 
    like = Likelihood(data=true_data, theory=theory, iz=iz, config={'verbose':True})
    post = Posterior(like, free_params, config={'verbose': True})
    mini = Minimizer(post, config={'verbose':True})
    true_minis.append(mini)

# %%
for mini in true_minis:
    z = mini.post.like.theory.z
    print('--------- z = {:.2f} -------'.format(z))
    # number of data points (per z bin)
    Nz, Nt_A, Nk_M = mini.post.like.data.Px_ZAM.shape
    Ndp = Nt_A * Nk_M
    # silence and minimize
    mini.silence()
    mini.minimize(compute_hesse=True)
    chi2 = mini.get_best_fit_chi2()
    best_fit = mini.get_best_fit_params()
    print('best fit chi2 and params')
    print(z, Ndp, chi2, best_fit)
    mini.plot_ellipses(pname_x='bias', pname_y='beta', nsig=2)
    label=""
    for key, par in mini.get_best_fit_params().items():
        label += "{} = {:.3f}   ".format(key, par)
    mini.plot_best_fit(multiply_by_k=False, theorylabel=label, datalabel='Stack (true continuum)')

# %%
if True:
    z = [ mini.post.like.theory.z for mini in true_minis]
    val = [ mini.get_best_fit_value('bias', return_hesse=True)[0] for mini in true_minis]
    err = [ mini.get_best_fit_value('bias', return_hesse=True)[1] for mini in true_minis]
    plt.errorbar(z, val, err, label='Px fits')
    val = [ mini.post.like.theory.lya_model.default_lya_params['bias'] for mini in true_minis ]
    plt.plot(z, val, 'ro', label='Xi3D fits')
    plt.xlabel('z')
    plt.ylabel('bias')
    plt.legend()

# %%
if True:
    z = [ mini.post.like.theory.z for mini in true_minis]
    val = [ mini.get_best_fit_value('beta', return_hesse=True)[0] for mini in true_minis]
    err = [ mini.get_best_fit_value('beta', return_hesse=True)[1] for mini in true_minis]
    plt.errorbar(z, val, err, label='Px fits')
    val = [ mini.post.like.theory.lya_model.default_lya_params['beta'] for mini in true_minis ]
    plt.plot(z, val, 'ro', label='Xi3D fits')
    plt.xlabel('z')
    plt.ylabel('beta')
    plt.legend()

# %% [markdown]
# ### Now fit continuum-fitted mocks (fixed bias/beta)

# %%
free_params = []
free_params.append(FreeParameter(
    name='kC_Mpc',
    min_value=1e-4,
    max_value=1e-1,
    ini_value=0.01,
    delta=0.001
))
free_params.append(FreeParameter(
    name='pC',
    min_value=0.01,
    max_value=2.0,
    ini_value=1.0,
    delta=0.01
))
for par in free_params:
    print(par.name, par.ini_value)

# %%
unco_minis = []
for iz, z in enumerate(unco_data.z): 
    config={'verbose': True, 'default_lya_model': default_lya_model, 'include_continuum': True}
    config['bias'] = true_minis[iz].get_best_fit_value('bias') 
    config['beta'] = true_minis[iz].get_best_fit_value('beta') 
    theory = Theory(z=z, fid_cosmo=cosmo, config=config)
    like = Likelihood(data=unco_data, theory=theory, iz=iz, config={'verbose':True})
    post = Posterior(like, free_params, config={'verbose': True})
    mini = Minimizer(post, config={'verbose':True})
    print('initial chi2', like.get_chi2())
    unco_minis.append(mini)

# %%
for mini in unco_minis:
    z = mini.post.like.theory.z
    print('--------- z = {:.2f} -------'.format(z))
    # number of data points (per z bin)
    Nz, Nt_A, Nk_M = mini.post.like.data.Px_ZAM.shape
    Ndp = Nt_A * Nk_M
    # silence and minimize
    mini.silence()
    mini.minimize(compute_hesse=True)
    chi2 = mini.get_best_fit_chi2()
    best_fit = mini.get_best_fit_params()
    print('best fit chi2 and params')
    print(z, Ndp, chi2, best_fit)
    mini.plot_ellipses(pname_x='kC_Mpc', pname_y='pC', nsig=2)
    label=""
    for key, par in mini.get_best_fit_params().items():
        label += "{} = {:.3f}   ".format(key, par)
    mini.plot_best_fit(multiply_by_k=False, theorylabel=label, datalabel='Stack (uncontaminated)')

# %%
if True:
    z = [ mini.post.like.theory.z for mini in unco_minis]
    val = [ mini.get_best_fit_value('kC_Mpc', return_hesse=True)[0] for mini in unco_minis]
    err = [ mini.get_best_fit_value('kC_Mpc', return_hesse=True)[1] for mini in unco_minis]
    plt.errorbar(z, val, err, label='Px fits')
    plt.xlabel('z')
    plt.ylabel('kC_Mpc')
    plt.tight_layout()
    plt.savefig('kC_Mpc_z.png')

# %%
if True:
    z = [ mini.post.like.theory.z for mini in unco_minis]
    val = [ mini.get_best_fit_value('pC', return_hesse=True)[0] for mini in unco_minis]
    err = [ mini.get_best_fit_value('pC', return_hesse=True)[1] for mini in unco_minis]
    plt.errorbar(z, val, err, label='Px fits')
    plt.xlabel('z')
    plt.ylabel('pC')
    plt.tight_layout()
    plt.savefig('pC_z.png')

# %%
if True:
    z = [ mini.post.like.theory.z for mini in unco_minis]
    ini_chi2 = [ mini.post.like.get_chi2() for mini in unco_minis]
    best_fit_chi2 = [ mini.get_best_fit_chi2() for mini in unco_minis]
    plt.plot(z, ini_chi2, label=r'initial $\chi^2$')
    plt.plot(z, best_fit_chi2, label=r'best-fit $\chi^2$')
    plt.plot(z, 0.1*Ndp*np.ones_like(z), label='Number of data points / 10')
    plt.xlabel('z')
    plt.legend()

# %%
for mini in unco_minis:
    z = mini.post.like.theory.z
    print(z, mini.get_best_fit_params())

# %%

# %%
