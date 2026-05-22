# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: cupix
# ---

# %% [markdown]
# # Fit all scales of DESI DR2

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
from cupix.likelihood.free_parameter import FreeParameter
from cupix.likelihood.posterior import Posterior
from cupix.sampling.minimize_posterior import Minimizer

# %% [markdown]
# ### Read the data from DESI DR2

# %%
basedir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/"
#fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5"
fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_20_w_res.hdf5"

# speed-up code by only looking at low kpar
kM_max_cut_AA = 1.0
km_max_cut_AA = 1.1*kM_max_cut_AA
data = DESI_DR2(filepath=fname, kM_max_cut_AA=kM_max_cut_AA, km_max_cut_AA=km_max_cut_AA)

# %%
# get the central value of each redshift bin, of length Nz
zs = data.z
# get a 1D array of central values of the measured k bins, of length Nk_M
k_M = data.k_M_centers_AA
# get two 1D arrays with the edges of each theta bin, of length Nt_A each
theta_A_min = data.theta_min_A_arcmin
theta_A_max = data.theta_max_A_arcmin
print(theta_A_min)

# %%
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = data.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_m={Nk_m}")
# rebinned values
Nz, Nt_A, Nk_M = data.Px_ZAM.shape
print(f"rebinned values: Nz={Nz}, Nt_A={Nt_A}, Nk_M={Nk_M}")
# number of data points (per z bin)
Ndp = Nt_A * Nk_M

# %% [markdown]
# ### Setup (contaminated) theory and likelihood objects

# %%
# define fiducial cosmo
cosmo = cosmology.Cosmology()

# %%
config={'verbose': False, 'include_hcd': True, 'include_metal': True,
        'include_sky': True, 'include_continuum': True}
theories = []
for iz,z in enumerate(zs):
    theories.append(Theory(z=z, fid_cosmo=cosmo, config=config))

# %%
likes = []
for iz, z in enumerate(zs):
    likes.append(Likelihood(data=data, theory=theories[iz], iz=iz, config={'verbose':False}))

# %% [markdown]
# ### Setup minimizers and free parameters

# %%
iz=1
print('Lya params =', theories[iz].lya_model.default_lya_params)
print('HCD params =', theories[iz].cont_model.default_hcd_params)
print('Metal params =', theories[iz].cont_model.default_metal_params)
print('Sky params =', theories[iz].cont_model.default_sky_params)
print('Cont params =', theories[iz].cont_model.default_continuum_params)

# %%
par_bias = FreeParameter(
    name='bias',
    min_value=-0.5,
    max_value=-0.01,
    ini_value=None,
    delta=0.001,
    gauss_prior_mean=None,
    gauss_prior_width=0.02
)
par_beta = FreeParameter(
    name='beta',
    min_value=0.5,
    max_value=2.5,
    ini_value=None,
    delta=0.01,
    gauss_prior_mean=None,
    gauss_prior_width=0.2
)
par_q1 = FreeParameter(
    name='q1',
    min_value=0.0,
    max_value=2.0,
    ini_value=None,
    delta=0.01,
    gauss_prior_mean=None,
    gauss_prior_width=0.2
)
par_bX = FreeParameter(
    name='b_X',
    min_value=-1.0,
    max_value=0.0,
    ini_value=-0.008,
    delta=1e-4,
    gauss_prior_mean=-0.008,
    gauss_prior_width=0.002
)
par_bH = FreeParameter(
    name='b_H',
    min_value=-1.0,
    max_value=0.0,
    ini_value=-0.02,
    delta=1e-3,
    gauss_prior_mean=-0.03,
    gauss_prior_width=0.01
)
free_params = [par_bias, par_beta, par_q1, par_bX, par_bH]
for par in free_params:
    print(par.name, par.ini_value)

# %%
minis = []
for iz in range(Nz):
#for iz in [2]:
    # chose more realistic initial values for bias/beta
    assert free_params[0].name == 'bias'
    ini_bias = likes[iz].theory.lya_model.default_lya_params['bias']
    free_params[0].ini_value = ini_bias
    free_params[0].gauss_prior_mean = ini_bias
    assert free_params[1].name == 'beta'
    ini_beta = likes[iz].theory.lya_model.default_lya_params['beta']   
    free_params[1].ini_value = ini_beta
    free_params[1].gauss_prior_mean = ini_beta
    assert free_params[2].name == 'q1'
    ini_q1 = likes[iz].theory.lya_model.default_lya_params['q1']   
    free_params[2].ini_value = ini_q1
    free_params[2].gauss_prior_mean = ini_q1
    
    print('----------------------------')
    print(iz, 'z bin has updated free params')
    for par in free_params:
        print(par.name, par.ini_value)
    post = Posterior(likes[iz], free_params, config={'verbose': False})
        
    mini = Minimizer(post, config={'verbose':False})
    print('minimizing zbin {}, at z={}'.format(iz, theories[iz].z))
    mini.silence()
    mini.minimize(compute_hesse=True)
    mini.print_results()
    minis.append(mini)

# %%
for mini in minis:
    plt.figure()
    mini.plot_ellipses(pname_x='bias', pname_y='beta', nsig=2)

# %%
for mini in minis:
    z=mini.post.like.theory.z
    plot_fname='px_fit_z_{}'.format(z)
    mini.plot_best_fit(multiply_by_k=False, every_other_theta=True, 
                       datalabel="DR2 (z = {})".format(z), 
                       theorylabel="Best-fit model", 
                       plot_fname=plot_fname, show=True)

# %%
if True:
    z = [ mini.post.like.theory.z for mini in minis]
    chi2 = [ mini.get_best_fit_chi2() for mini in minis]
    plt.plot(z, chi2, label=r'$\chi^2$')
    plt.plot(z, Ndp*np.ones_like(z), label='Number of data points')
    plt.xlabel('z')
    plt.legend()
    plt.tight_layout()
    plt.savefig('chi2_fit_z.png')

# %%
if True:
    z = [ mini.post.like.theory.z for mini in minis]
    val = [ mini.get_best_fit_value('bias', return_hesse=True)[0] for mini in minis]
    err = [ mini.get_best_fit_value('bias', return_hesse=True)[1] for mini in minis]
    plt.errorbar(z, val, err, label='Px fits')
    #xi3d=theories[0].lya_model.default_lya_params['bias']
    #plt.plot(2.33, xi3d, 'ro', label='Xi3D fit')
    plt.xlabel('z')
    plt.ylabel('bias')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bias_fit_z.png')

# %%
if True:
    z = [ mini.post.like.theory.z for mini in minis]
    val = [ mini.get_best_fit_value('beta', return_hesse=True)[0] for mini in minis]
    err = [ mini.get_best_fit_value('beta', return_hesse=True)[1] for mini in minis]
    plt.errorbar(z, val, err, label='Px fits')
    #xi3d=theories[0].lya_model.default_lya_params['beta']
    #plt.plot(2.33, xi3d, 'ro', label='Xi3D fit')
    plt.xlabel('z')
    plt.ylabel('beta')
    plt.legend()
    plt.tight_layout()
    plt.savefig('beta_fit_z.png')

# %%
if True:
    z = [ mini.post.like.theory.z for mini in minis]
    val = [ mini.get_best_fit_value('b_X', return_hesse=True)[0] for mini in minis]
    err = [ mini.get_best_fit_value('b_X', return_hesse=True)[1] for mini in minis]
    plt.errorbar(z, val, err, label='DR2 Px fits')
    #xi3d=theories[0].lya_model.default_lya_params['beta']
    #plt.plot(2.33, xi3d, 'ro', label='Xi3D fit')
    plt.xlabel('z')
    plt.ylabel('b_X')
    plt.legend()
    plt.ylim(-0.02, 0.0)
    plt.tight_layout()
    plt.savefig('b_X_fit_z.png')

# %%

# %%

# %%
