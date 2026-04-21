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
# # Fit sky residuals from DESI DR2

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
from cupix.likelihood.minimize_posterior import Minimizer

# %% [markdown]
# ### Read the data from DESI DR2 (large angular separations only)

# %%
basedir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/"
fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5"
data = DESI_DR2(fname, kM_min_cut_AA=0.5, kM_max_cut_AA=1.0, km_max_cut_AA=1.2, theta_min_cut_arcmin=10.0)

# %%
# get the central value of each redshift bin, of length Nz
zs = data.z
# get a 1D array of central values of the measured k bins, of length Nk_M
k_M = data.k_M_centers_AA
# get two 1D arrays with the edges of each theta bin, of length Nt_A each
theta_A_min = data.theta_min_A_arcmin
theta_A_max = data.theta_max_A_arcmin

# %%
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = data.U_ZaMn.shape
print(f"native binning: Nz={Nz}, Nt_a={Nt_a}, Nk_m={Nk_m}")
# rebinned values
Nz, Nt_A, Nk_M = data.Px_ZAM.shape
print(f"rebinned values: Nz={Nz}, Nt_A={Nt_A}, Nk_M={Nk_M}")
# number of data points (per z bin)
Ndp = Nt_A * Nk_M


# %%
def plot_theta_bin(iz, it_M):
    label = r"${:.2f}' < \theta < {:.2f}'$".format(theta_A_min[it_M], theta_A_max[it_M])
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it_M]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_M]))
#    print(len(k_M), len(Px), len(sig_Px))
    plt.errorbar(k_M, Px, sig_Px, label=label)


# %%
def plot_z_bin(iz, its_M):
    for it_M in its_M:
        plot_theta_bin(iz=iz, it_M=it_M)
    plt.title('DESI DR2 at z={:.1f}'.format(zs[iz]))
    plt.legend()
    plt.xlabel(r'$k_\parallel$ [1/A]')
    plt.ylabel(r'$P_\times(\theta, k_\parallel)$ [A]')


# %%
for iz, z in enumerate(zs):
    plt.figure(figsize=[8,3])
    plot_z_bin(iz=iz, its_M=range(Nt_A))

# %% [markdown]
# ### Setup (contaminated) theory and likelihood objects

# %%
# define fiducial cosmo
cosmo = cosmology.Cosmology()

# %%
free_bias=False
free_params = []
if free_bias:
    free_params.append(FreeParameter(
        name='bias',
        min_value=-.5,
        max_value=-.02,
        ini_value=-0.15,
        delta=0.001
        ))
free_params.append(FreeParameter(
    name='b_noise_Mpc',
    min_value=1e-4,
    max_value=1e-1,
    ini_value=0.01,
    delta = 0.0001
    ))
for par in free_params:
    print(par.name)

# %%
config={'verbose': True, 'include_hcd': False, 'include_metal': False,
        'include_sky': True, 'include_continuum': True}
minis = []
for iz, z in enumerate(data.z): 
    theory = Theory(z=z, fid_cosmo=cosmo, config=config)
    like = Likelihood(data=data, theory=theory, iz=iz, config={'verbose':True})
    post = Posterior(like, free_params, config={'verbose': True})
    mini = Minimizer(post, config={'verbose':True}) 
    minis.append(mini)

# %%
for mini in minis:
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

# %%
for mini in minis:
    z = mini.post.like.theory.z
    label=""
    for key, par in mini.get_best_fit_params().items():
        label += "{} = {:.4f}   ".format(key, par)
    plot_fname='px_fit_sky_z_{}'.format(z)
    mini.plot_best_fit(multiply_by_k=False, theorylabel=label, datalabel='DESI DR2 (z={})'.format(z),
                      plot_fname=plot_fname)

# %%
z = [ mini.post.like.theory.z for mini in minis]
ini_chi2 = [ mini.post.like.get_chi2() for mini in minis]
best_fit_chi2 = [ mini.get_best_fit_chi2() for mini in minis]
plt.plot(z, best_fit_chi2, label=r'best-fit $\chi^2$')
#plt.plot(z, ini_chi2, label=r'initial $\chi^2$')
plt.plot(z, Ndp*np.ones_like(z), label='Number of data points')
plt.xlabel('z')
plt.legend()
plt.tight_layout()
plt.savefig('chi2_fit_sky_z.png')

# %%
z = [ mini.post.like.theory.z for mini in minis]
val = [ mini.get_best_fit_value('b_noise_Mpc', return_hesse=True)[0] for mini in minis]
err = [ mini.get_best_fit_value('b_noise_Mpc', return_hesse=True)[1] for mini in minis]
plt.errorbar(z, val, err, label='DESI DR2 Px')
plt.xlabel('z')
plt.ylabel('b_noise [Mpc]')
plt.legend()
plt.ylim([0.0,0.004])
#plt.axhline(y=0.00125)
plt.tight_layout()
plt.savefig('b_noise_z.png')

# %%
for mini in minis:
    z = mini.post.like.theory.z
    print(z, mini.get_best_fit_params())

# %%
