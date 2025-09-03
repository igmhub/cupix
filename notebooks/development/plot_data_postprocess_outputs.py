# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: python3
# ---

# %%
# Read from postprocessing output and generate plots and useful info for cupix

# %%
import numpy as np
import h5py 
import matplotlib.pyplot as plt

# %%
root = '/Users/ssatyavolu/projects/DESI/Y3_Lya_Px/mocks/raw_mocks/'
data = root+'output_data_for_cupix_zbins_4_thetabins_40_nhp41.hdf5'

# %%
with h5py.File(data, 'r') as f:
    k_m = f['k_m'][:] # array of shape (Nk,)
    k_M = f['k_M'][:] # array of shape (NK,)
    theta_min_a = f['theta_min_a'][:] # array of shape (N_theta_a,)
    theta_max_a = f['theta_max_a'][:] 
    theta_min_A = f['theta_min_A'][:] # array of shape (N_theta_A,)
    theta_max_A = f['theta_max_A'][:]
    zbins = f['z_centers'][:] # array of shape (Nz,)
    N_fft = f.attrs['N_fft']
    L_fft  = f.attrs['L_fft']
    P_Z_AM = f['P_Z_AM'][:] # array of shape (Nz,N_theta_A, NK)
    C_Z_AM = f['C_Z_AMN'][:] # array of shape (Nz,N_theta_A, NK, NK)
    U_Z_aMn = f['U_Z_aMn'][:] # array of shape (Nz,N_theta_a, NK, Nk)
    B_A_a = f['B_A_a'][:]   # array of shape (N_theta_A, N_theta_a)
    V_Z_aM = f['V_Z_aM'][:] # array of shape (Nz, N_theta_a, NK)
    V_Z_am = f['V_Z_am'][:] # array of shape (Nz, N_theta_a, Nk)

  

# %%
# Plot the aveage binned Px in each z-bin for a few theta bins
nbins = len(zbins)
ntheta = len(theta_min_A)
print('Number of z-bins = ', nbins)
print('Number of theta bins = ', ntheta)
plt.figure(figsize=(12,8))
for i in range(nbins):
    plt.subplot(2,2,i+1)
    plt.subplots_adjust(hspace=0.3)
    for j in range(0,ntheta,3):
        print(i, j)
        plt.plot(k_M, P_Z_AM[i,j,:], label=r'$\theta$ = %.2f - %.2f'%(theta_min_A[j], theta_max_A[j]))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(1e1, 1e5)
    plt.xlabel(r'$k \,$[1/A]')
    plt.ylabel(r'$P_{\times}(k,\theta)$ [A]')
    plt.title(r'$z$ = %.2f'%(zbins[i]))
    plt.legend(fontsize=8)


# %%
# Plot the covarance matrix for a few z and a given theta bin
plt.figure(figsize=(12,8))
for i in range(nbins):
    plt.subplot(2,2,i+1)
    plt.subplots_adjust(hspace=0.3)
    j = ntheta//2
    plt.imshow(C_Z_AM[i,j,:,:], origin='lower', aspect='auto', extent=(k_M[0], k_M[-1], k_M[0], k_M[-1]))
    plt.colorbar()
    plt.xlabel(r'$k \,$[1/A]')
    plt.ylabel(r'$k \,$[1/A]')
    plt.title(r'$z$ = %.2f, $\theta$ = %.2f - %.2f'%(zbins[i], theta_min_A[j], theta_max_A[j]))

# %%
# read the window matrix
U_Z_aMn.shape

# %%
print(zbins)

# %%
print(theta_max_a)

# %%
