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

# %%
# ls /global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/contaminated/

# %%
# path to mocks
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
#fname = mockdir + "tru_cont/tru_cont_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
uncont_fname = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50.hdf5"
cont_fname = mockdir + "contaminated/contaminated_baseline_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg50_ncov.hdf5"

iz = 1
data_uncont = DESI_DR2(config = {'data_file':uncont_fname, 'theta_min_cut_arcmin':0})
data_cont = DESI_DR2(config = {'data_file':cont_fname, 'theta_min_cut_arcmin':0})
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
# default_lya_model = 'best_fit_p1d_from_dr1'
default_lya_model = 'best_fit_arinyo_from_colore'

theory_config = {'verbose': False, 'default_lya_model': default_lya_model, 'include_continuum': False, 'include_hcd':False}
theory = Theory(z=z, fid_cosmo=cosmo, config=theory_config)
print(theory.lya_model.default_lya_params)
print(theory.cont_model.default_continuum_params)

# %%

like_cont = Likelihood(data=data_cont, theory=theory, iz=iz, config={'verbose':False})
like_uncont = Likelihood(data=data_uncont, theory=theory, iz=iz, config={'verbose':False})

# %%
# like_cont.plot_px(every_other_theta=True, xlim=[0, 0.5], theorylabel='contaminated window matrix', datalabel='contaminated mock', ylim=[0,.0012])

# %%
# like_uncont.plot_px(every_other_theta=True, xlim=[0, 0.5], theorylabel='uncontaminated window matrix', datalabel='Uncontaminated mock', ylim=[0,.0012])

# %%
model_px_cont = like_cont.get_convolved_px()
model_px_uncont = like_uncont.get_convolved_px()

# %%
k_AA = like_cont.data.k_M_centers_AA

# %%
# colors = ['C{}'.format(i) for i in range(len(model_px_cont))]
has_label = False
for theta_A in range(len(model_px_cont))[4:]:
    if not has_label:
        labelun = 'uncontaminated'
        labelcont = 'contaminated'
        has_label = True
    else:
        labelun = None
        labelcont = None
    plt.plot(k_AA, k_AA**2*model_px_uncont[theta_A], label=labelun,   linestyle='solid')
    plt.plot(k_AA, k_AA**2*model_px_cont[theta_A], label=labelcont, color='k', linestyle='dotted')
    print(data_cont.theta_centers_arcmin[theta_A])
plt.xlim([0, 0.6])
plt.ylabel(r'$k^2 P_\times(k)$')
plt.legend()

# %%

# plot residuals
# get a continuous colormap
cmap = plt.get_cmap('viridis')
for theta_A in range(len(model_px_cont)):
    if theta_A%2==0:
        plt.plot(k_AA, (model_px_cont[theta_A]  / model_px_uncont[theta_A]), label=rf"$\theta = {like_cont.data.theta_centers_arcmin[theta_A]:.1f}^\prime$", color=cmap(theta_A/len(model_px_cont)))

plt.xlim([0, 0.77])
# plt.axhspan(-.03,.03, label='3%', color='grey', alpha=.5)
plt.ylabel(r'$P_\mathrm{\times}^\mathrm{mask}\;/\;P_\mathrm{\times}^\mathrm{nomask}$', fontsize=14)
plt.title("Windowed theory with vs. without masking", fontsize=16)
plt.axhline(1, color='k', linestyle='dashed', alpha=.5)
plt.xlabel(r'$k [\AA^{-1}$]')
plt.legend(fontsize=12, ncol=2)
plt.ylim([.5,3])

# %%
for a in range(10):
    plt.imshow(data_cont.U_ZaMn[0,a,:,:]/data_cont.U_ZaMn[0,a,:,:])
    plt.colorbar()
    break
# plt.imshow(data_cont.U_ZaMn[0,0,:,:])

# %%
