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
# # Compare the stack of uncontaminated and contaminated mocks

# %%
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
# %load_ext autoreload
# %autoreload 2

# %%
from cupix.px_data.data_DESI_DR2 import DESI_DR2

# %%
# path to mocks
mockdir = "/global/cfs/cdirs/desi/users/sindhu_s/Lya_Px_measurements/mocks/stacked_outputs/"
# will rescale covariance of the stacks with this number
rescale_cov = True
Nm = 50

# %%
unco_fname = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_10_w_res_avg{}.hdf5".format(Nm)
#unco_fname = mockdir + "uncontaminated/uncontaminated_binned_out_bf3_px-zbins_4-thetabins_20_w_res_avg{}.hdf5".format(Nm)
unco_data = DESI_DR2(unco_fname)
if rescale_cov:
    unco_data.cov_ZAM *= 1.0 / Nm

# %%
# path to mocks
cont_fname = mockdir + "contaminated/contaminated_binned_out_bf3_px-zbins_4-thetabins_10_w_res_avg{}.hdf5".format(Nm)
cont_data = DESI_DR2(cont_fname)
if rescale_cov:
    cont_data.cov_ZAM *= 1.0 / Nm

# %% [markdown]
# ### Compare measurement of stacks

# %%
iz = 1 
theta_centers = cont_data.theta_centers_arcmin
assert np.sum(theta_centers - unco_data.theta_centers_arcmin) == 0.0
print(theta_centers)
for it_A, theta in enumerate(theta_centers):
    plt.figure()
    for label, data in zip(['uncontaminated', 'contaminated'], [unco_data, cont_data]):
        k_M = data.k_M_centers_AA
        err = np.diag(np.squeeze(data.cov_ZAM[iz, it_A, :, :]))**0.5
        px = data.Px_ZAM[iz, it_A, :]
        plt.errorbar(k_M, px, err, label=label)
    plt.title(r'$\theta = {:.1f}$ arcmin'.format(theta))
    plt.xlabel(r'$k_\parallel$ [1/A]')
    plt.ylabel(r'$P_\times(k_\parallel, \theta)$ [A]')
    plt.legend()
    plt.savefig('px_mocks_contamination_{}.png'.format(it_A))
    plt.show()

# %%
cont_cov = cont_data.cov_ZAM
unco_cov = unco_data.cov_ZAM

# %%
cont_cov.shape


# %%
def compare_covs(iz, it_A):
    plt.figure()
    cont_err = np.sqrt(np.diagonal(cont_cov[iz, it_A]))
    unco_err = np.sqrt(np.diagonal(unco_cov[iz, it_A]))
    plt.plot(k_M, cont_err, label='contaminated')
    plt.plot(k_M, unco_err, label='uncontaminated')
    plt.xlabel(r'$k_\parallel$ [1/A]')
    plt.ylabel('mean error')
    plt.legend()
    plt.ylim(bottom=0)
    plt.title('z = {} , theta = {:.2f} arcmin'.format(cont_data.z[iz], theta_centers[it_A]))
    plt.tight_layout()
    plt.savefig('compare_errors_{}_{}.png'.format(iz, it_A))
    plt.close()


# %%
for iz in range(4):
    for it_A in range(len(theta_centers)):
        compare_covs(iz, it_A)

# %%
