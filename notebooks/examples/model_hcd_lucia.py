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
# # Compare P3D with and without HCD Contamination

# %% [markdown]
# Attempt to reproduce something similar to the bottom panel of Fig B1 in McQuinn & White (2011, https://arxiv.org/abs/1102.1752)

# %%
import numpy as np
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2


# %%
from lace.cosmo import cosmology
from cupix.likelihood.theory import Theory

# %%
# default cosmology for now
cosmo = cosmology.Cosmology()

# %%
# redshift and bias / beta from that paper
z = 2.5
bias = -0.18
beta = 1.0
# close to value used in McQuinn & White, although in there this is kpar smoothing only
kp_Mpc = 0.08 * cosmo.get_dkms_dMpc(z)
print('k_p = {:.3f} 1/Mpc'.format(kp_Mpc))
# HCD model (L_H
b_H = -0.036
beta_H = 0.5
L_H = 7.0 / cosmo.get_h()
print('L_H = {:.3f} Mpc'.format(L_H))
config = dict(bias=bias, beta=beta, q1=0, q2=0, kp_Mpc=kp_Mpc,
              include_hcd=True, b_H=b_H, beta_H=beta_H, L_H=L_H,
              verbose=True)

# %%
theory = Theory(z, fid_cosmo=cosmo, config=config)


# %%
def compare_p3d(mu, relative=False):
    k = np.logspace(-3, 1, 1000)
    p3d_lya = theory.get_p3d_lya_Mpc(k=k, mu=mu)
    p3d_lya_hcd = theory.get_p3d_lya_hcd_Mpc(k=k, mu=mu)
    if relative:
        plt.semilogx(k, p3d_lya_hcd / p3d_lya - 1.0)
        plt.ylabel('relative P3D contamination')
        plt.ylim(-0.5, 0.5)
    else:
        plt.loglog(k, p3d_lya, label='Lya only')
        plt.loglog(k, p3d_lya_hcd, label='Lya + HCD')
        p3d_max = np.max(p3d_lya_hcd)
        plt.ylim(0.001*p3d_max, 2*p3d_max)
        plt.ylabel('P3D [Mpc^3]')
        plt.legend()
    plt.xlabel('k [1/Mpc]')
    plt.title(r'$\mu =$ {:.2f}'.format(mu))


# %%
compare_p3d(mu=0.0, relative=True)

# %%
compare_p3d(mu=1.0, relative=True)
