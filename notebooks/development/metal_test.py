# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: cupix
# ---

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
from cupix.inference.sampling_funcs import prepare_free_parameters

# %%
basedir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/DR2_Px/baseline/"
#fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_10_w_res.hdf5"
# fname = basedir + "bf3_binned_out_px-zbins_4-thetabins_20_w_res.hdf5"
# fname = basedir + "wp1d/drop_DLAs/GP_plus_snrcut/bf3_binned_out_px-zbins_4-thetabins_20_w_res_wp1d.hdf5"
fname = basedir + "wp1d/drop_BALs_and_DLAs/fs_cut/bf3_binned_out_px-zbins_4-thetabins_20_w_res_wp1d.hdf5"
kM_max_cut_AA = .7
km_max_cut_AA = 1.1 * kM_max_cut_AA
print(km_max_cut_AA)
data = DESI_DR2(config={'data_file':fname, 'kM_max_cut_AA':kM_max_cut_AA, 'km_max_cut_AA':km_max_cut_AA, 'theta_min_cut_arcmin':1})

# %%
zs = data.z
# define fiducial cosmo
cosmo = cosmology.Cosmology()
b_noise = [0.0040, 0.0017, 0.0017, 0.0016]
theories_lya = []
theories_cont = []
iz = 0
for z in zs:
    theories_lya.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False, 'default_lya_model':'best_fit_igm_from_p1d'}))
    theories_cont.append(Theory(z=z, fid_cosmo=cosmo, config={'verbose': False, 
                                                            'include_hcd': True, 'include_metal': True,
                                                            'include_sky': True, 'include_continuum': True, 'default_lya_model':'best_fit_igm_from_p1d',
                                                            'b_noise_Mpc': b_noise[iz]} ))
    iz += 1

# %%
kpar = np.linspace(0,1,100)
rt = np.linspace(0,60,20)
px_metal_auto = theories_lya[3].get_px_metal_auto_obs(rt, kpar, params={'b_X':0.03})
px_metal_cross = theories_lya[3].get_px_metal_cross_obs(rt, kpar, params={'b_X':0.03})
for theta in range(10):

    # plt.plot(kpar, px_metal_auto[theta,:], label=rf'$\theta={rt[theta]}$ Mpc')
    plt.plot(kpar, px_metal_cross[theta,:], label=rf'$\theta={rt[theta]}$ Mpc')    
    # plt.plot(kpar, px_metal_auto[theta,:]+px_metal_cross[theta,:], label=rf'$\theta={rt[theta]}$ Mpc')

# %%
k = np.linspace(0,1, 100)
mu = np.linspace(0,1,20)

px_metal_auto = theories_lya[3].get_p3d_metal_auto_Mpc(k[:,np.newaxis], mu[:,np.newaxis])

# %%

# %%
