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
# # Fit HCD contamination from DESI DR2

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
#fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5"
fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_20_w_res.hdf5"
data = DESI_DR2(fname, kM_max_cut_AA=0.5, km_max_cut_AA=0.55, theta_min_cut_arcmin=10.0)

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
# set the likelihood parameters as the Arinyo params with some fiducial values
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
config={'verbose': False, 'include_hcd': True, 'include_metal': True,
        'include_sky': True, 'include_continuum': True}
# start with a single redshift, with different theories for fixed values of b_H
iz=1
z=data.z[iz]
minis={}
for b_H in [-0.01, -0.02, -0.04, -0.08]:
    config['b_H'] = b_H
    theory = Theory(z=z, fid_cosmo=cosmo, config=config)
    # chose more realistic initial values for bias/beta
    assert free_params[0].name == 'bias'
    free_params[0].ini_value = theory.lya_model.default_lya_params['bias']
    assert free_params[1].name == 'beta'
    free_params[1].ini_value = theory.lya_model.default_lya_params['beta']
    like = Likelihood(data=data, theory=theory, iz=iz, config={'verbose':False})
    post = Posterior(like, free_params, config={'verbose': False})
    mini = Minimizer(post, config={'verbose':False})
    minis[b_H] = mini

# %%
for b_H, mini in minis.items():
    mini.silence()
    mini.minimize(compute_hesse=True)
    chi2 = mini.get_best_fit_chi2()
    best_fit = mini.get_best_fit_params()
    print('best fit chi2 and params')
    print(b_H, Ndp, chi2, best_fit)
    print('-------')

# %%
for b_H, mini in minis.items():
    z=mini.post.like.theory.z
    mini.plot_best_fit(multiply_by_k=False, every_other_theta=False, xlim=[-.01, 0.4], 
                       datalabel="DR2 (z = {})".format(z), 
                       theorylabel="b_H = {:.3f}".format(b_H), show=True)

# %%
b_H = [ float(key) for key in minis.keys()]
chi2 = [ mini.get_best_fit_chi2() for mini in minis.values()]
plt.plot(b_H, chi2, label=r'$\chi^2$')
plt.plot(b_H, Ndp*np.ones_like(b_H), label='Number of data points')
plt.xlabel('b_H')
plt.legend()

# %% [markdown]
# ### get best-fit predictions from each likelihood

# %%
models = {}
for b_H, mini in minis.items():
    params = mini.get_best_fit_params()
    print(b_H, params)
    px = mini.post.like.get_convolved_px(params=params)
    models[b_H] = px


# %%
def plot_theta_bin(it_A):
    k_M = data.k_M_centers_AA
    theta = data.theta_centers_arcmin[it_A]
    Px = data.Px_ZAM[iz][it_A]
    sig_Px = np.sqrt(np.diagonal(data.cov_ZAM[iz][it_A]))
    plt.errorbar(k_M, Px, sig_Px)
    for b_H, model in models.items():
        plt.plot(k_M, model[it_A], label='b_H = {:.3f}'.format(b_H))
    plt.legend()
    plt.axhline(y=0, ls=':', color='gray')
    plt.title(r"$\theta = {:.2f}'$".format(theta))


# %%
plot_theta_bin(0)

# %%
plot_theta_bin(8)

# %%
