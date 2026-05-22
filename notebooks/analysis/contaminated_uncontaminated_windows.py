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

# %%
# path to mocks
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
#fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
uncont_fname = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
cont_fname = mockdir + "contaminated/contaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"

iz = 1
data_uncont = DESI_DR2(uncont_fname, theta_min_cut_arcmin=20)
data_cont = DESI_DR2(cont_fname, theta_min_cut_arcmin=20)
z = data_cont.z[iz]
print('analyze zbin {}, z = {}'.format(iz, z))

# %%
# check out two of the window matrices
plt.imshow(data_uncont.U_ZaMn[0,0] / data_cont.U_ZaMn[0,0])
plt.colorbar()
plt.show()
plt.clf()

# %%
# setup cosmology defuault
cosmo = cosmology.Cosmology()
# starting point for Lya bias parameters in mocks
default_lya_model = 'best_fit_p1d_from_dr1'

theory_config = {'verbose': False, 'default_lya_model': default_lya_model, 'include_continuum': True, 'include_hcd':True}
theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
print(theory.lya_model.default_lya_params)
print(theory.cont_model.default_continuum_params)

# %%
like_cont = Likelihood(data=data_cont, theory=theory, iz=iz, config={'verbose':False})
like_uncont = Likelihood(data=data_uncont, theory=theory, iz=iz, config={'verbose':False})

# %%
like_cont.plot_px(every_other_theta=True, xlim=[0, 0.5], theorylabel='contaminated window matrix', datalabel='contaminated mock', ylim=[0,.0012])

# %%
like_uncont.plot_px(every_other_theta=True, xlim=[0, 0.5], theorylabel='uncontaminated window matrix', datalabel='Uncontaminated mock', ylim=[0,.0012])

# %%
model_px_cont = like_cont.get_convolved_px()
model_px_uncont = like_uncont.get_convolved_px()

# %%
k_AA = like_cont.data.k_M_centers_AA

# %%
# colors = ['C{}'.format(i) for i in range(len(model_px_cont))]
for theta_A in range(len(model_px_cont))[4:]:
    plt.plot(k_AA, model_px_uncont[theta_A], label='uncontaminated',   linestyle='dashed')
    plt.plot(k_AA, model_px_cont[theta_A], label='contaminated', color='k', linestyle='dotted')
    
plt.xlim([0, 0.3])


# %%
# plot residuals
colors = ['C{}'.format(i) for i in range(len(model_px_cont))]
for theta_A in range(len(model_px_cont)):
    plt.plot(k_AA, (model_px_cont[theta_A] - model_px_uncont[theta_A]) / model_px_uncont[theta_A], label=f'theta = {like_cont.data.theta_centers_arcmin[theta_A]:.1f}\'', color=colors[theta_A])

plt.xlim([0, 1])
plt.axhspan(-.03,.03, label='3%', color='grey', alpha=.5)
plt.ylabel('residual (cont - uncont) / uncont')
plt.xlabel('k [1/AA]')
plt.legend(fontsize=12, ncol=2)

# %%
