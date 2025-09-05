# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Likelihood example notebook

# %%
import sys
import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
# %load_ext autoreload
# %autoreload 2
from lace.cosmo import camb_cosmo, fit_linP
import matplotlib.pyplot as plt
import h5py
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta

# %% [markdown]
# Set the redshifts and fiducial cosmology

# %%

# Load emulator
z = np.array([2.2])
omnuh2 = 0.0006
mnu = omnuh2 * 93.14
H0 = 67.36
omch2 = 0.12
ombh2 = 0.02237
As = 2.1e-9
ns = 0.9649
nrun = 0.0
w = -1.0
omk = 0
fid_cosmo = {
    'H0': H0,
    'omch2': omch2,
    'ombh2': ombh2,
    'mnu': mnu,
    'omk': omk,
    'As': As,
    'ns': ns,
    'nrun': nrun,
    'w': w
}
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)

# %% [markdown]
# Set up the emulator

# %%
ffemu = FF_emulator(z, fid_cosmo, cc)
ffemu.kp_Mpc = 1 # set pivot point

# %% [markdown]
# Set up the theory with some default parameters (this part inherits some old behavior from cup1d that we may change later)

# %%
emu_params = Args()
emu_params.set_baseline()
print(emu_params)

# %%
theory_AA = set_theory(emu_params, ffemu, free_parameters=['ln_tau_0'], k_unit='iAA')
theory_AA.set_fid_cosmo(z)
theory_AA.emulator = ffemu


# %% [markdown]
# Set the data

# %%
# Sindhu's code to read the px measurements. This should be incorporated into the classes in px_data 
class Px_postproc:
    def __init__(self,filepath):
        self.filepath = filepath
        self.load_metadata()
        pass


    def load_metadata(self):


        with h5py.File(self.filepath,'r') as f:
        # Read metadata
            self.k_m = f['metadata'].attrs['k_m']
            self.k_M_edges = f['metadata'].attrs['k_M_edges']
            self.theta_min_a = f['metadata'].attrs['theta_min_a']
            self.theta_max_a = f['metadata'].attrs['theta_max_a']
            self.theta_min_A = f['metadata'].attrs['theta_min_A']
            self.theta_max_A = f['metadata'].attrs['theta_max_A']
            self.zbin_centers = f['metadata'].attrs['z_centers']
            self.N_fft = f['metadata'].attrs['N_fft']
            self.L_fft = f['metadata'].attrs['L_fft']
            self.B_A_a = f['metadata/B_A_a'][:]
        

    def get_Px_z_T(self,zbin_ind,theta_bin_ind):
        with h5py.File(self.filepath,'r') as f:
            print(f['P_Z_AM'].keys())
            P_Z_AM = f['P_Z_AM/z_{}/theta_rebin_{}/'.format(zbin_ind,theta_bin_ind)][:] # shape (NK,)
            zval = self.zbin_centers[zbin_ind]
            theta_val = 0.5*(self.theta_min_A[theta_bin_ind]+self.theta_max_A[theta_bin_ind])
            return P_Z_AM, zval, theta_val
  
    def get_cov_matrix_z_T(self,zbin_ind,theta_bin_ind):
        with h5py.File(self.filepath,'r') as f:
            cov_matrix = f['C_Z_AMN/z_{}/theta_rebin_{}/'.format(zbin_ind,theta_bin_ind)][:] # shape (NK,NK)
            zval = self.zbin_centers[zbin_ind]
            theta_val = 0.5*(self.theta_min_A[theta_bin_ind]+self.theta_max_A[theta_bin_ind])
            return cov_matrix, zval, theta_val
  
    def get_window_matrix_z_t(self,zbin_ind,theta_bin_ind):
        with h5py.File(self.filepath,'r') as f:
            window_matrix = f['U_Z_aMn/z_{}/theta_{}/'.format(zbin_ind,theta_bin_ind)][:] # shape (NK,Nk)
            zval = self.zbin_centers[zbin_ind]
            theta_val = 0.5*(self.theta_min_a[theta_bin_ind]+self.theta_max_a[theta_bin_ind])
            return window_matrix, zval, theta_val


    def get_V_Z_aM(self,zbin_ind,theta_bin_ind):
        with h5py.File(self.filepath,'r') as f:
            norm_V = f['V_Z_aM/z_{}/theta_{}/'.format(zbin_ind,theta_bin_ind)][:] # shape (NK,)
            zval = self.zbin_centers[zbin_ind]
            theta_val = 0.5*(self.theta_min_a[theta_bin_ind]+self.theta_max_a[theta_bin_ind])
        return norm_V, zval, theta_val

    def theta_a_in_A_ind(self,theta_bin_ind):
        ''' Returns a boolean array of shape N_k (# of small theta bins)
            where True indicates that the small theta bin is fully contained within the large theta bin '''
        return np.arange(len(self.theta_min_a))[(self.theta_min_a>self.theta_min_A[theta_bin_ind])&(self.theta_max_a<self.theta_max_A[theta_bin_ind])]

# %%
Px_data = Px_postproc("../../data/px_measurements/Lyacolore/output_data_for_cupix_zbins_4_thetabins_40_nhp41.hdf5")
# To read and plot rebinned Px at a given z and theta_A 
Px_z_t,zval,_ = Px_data.get_Px_z_T(3,9) # 3→ z index, 9 → theta index
plt.plot(Px_data.k_M_edges[:-1],Px_z_t,label='z={:.2f}'.format(zval))
plt.show()
plt.clf()
# To read and plot rebinned covariance at a given z and theta_A 
cov_matrix,_,_ = Px_data.get_cov_matrix_z_T(3,9) # 3→ z index, 9 → rebinned theta index
plt.imshow(cov_matrix)
plt.show()
plt.clf()
# To read window matrix at a given z and theta_a
U = Px_data.get_window_matrix_z_t(3,39)[0]# 3→ z index, 39 → native theta bin index
# To read normalisation V_Z_aM, rebinned in k, at a given z and theta_a
V = Px_data.get_V_Z_aM(3,39)[0]

# %% [markdown]
# Evaluate the likelihood for a given coarse-theta bin and redshift
#

# %%
z_ind = 0
theta_A_ind = 0
print("Evaluating likelihood for z center = {:.2f}, theta_A center = {:.2f} arcmin".format(Px_data.zbin_centers[z_ind],0.5*(Px_data.theta_min_A[theta_A_ind]+Px_data.theta_max_A[theta_A_ind])))
# get all small theta bins that correspond to this large theta bin
theta_a_inds = Px_data.theta_a_in_A_ind(theta_A_ind)
print(theta_a_inds)
print(f"The large theta bin spans (in arcmin): [{Px_data.theta_min_A[theta_A_ind]:.2f},{Px_data.theta_max_A[theta_A_ind]:.2f}]")
print("Number of small theta bins in this large theta bin: ",theta_a_inds.sum())
print(f"These are the small theta bins (in arcmin): {list(zip(Px_data.theta_min_a[theta_a_inds],Px_data.theta_max_a[theta_a_inds]))}")


# %% [markdown]
# Set some random Arinyo parameters (Again, this part will change)

# %%
likelihood_params = []
likelihood_params.append(LikelihoodParameter(
    name='Delta2_p',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='n_p',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='mF',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='gamma',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='kF_Mpc',
    min_value=-1.0,
    max_value=1.0,
    ))
likelihood_params.append(LikelihoodParameter(
    name='sigT_Mpc',
    min_value=-1.0,
    max_value=1.0,
    ))


# %%
z


# %%
theta_a_inds

# %%

# %%
# collect the Px for all small theta bins that correspond to this large theta bin
Px_aMZ_list = []
V_aMZ_list = []
for t_a in theta_a_inds:
    z    = Px_data.zbin_centers[z_ind]
    k_AA = Px_data.k_m[1:]
    theta_a_arcmin = [(Px_data.theta_min_a[t_a] + Px_data.theta_max_a[t_a])/2.]
    print(theta_a_arcmin)
    # first, get the theory prediction
    Px_amZ = theory_AA.get_px_AA(
        zs = [z],
        k_AA=[k_AA],
        theta_arcmin=[theta_a_arcmin],
        like_params=likelihood_params,
        return_blob=False
    )
    # next, apply the window matrix

    U_aMnZ = Px_data.get_window_matrix_z_t(z_ind,t_a)[0][:,1:] # remove first column corresponding to k=0
    Px_aMZ = convolve_window(U_aMnZ,Px_amZ[0].T)
    Px_aMZ_list.append(Px_aMZ.flatten())
    plt.plot(k_AA, Px_amZ.flatten(), label='theory theta_a={:.2f}'.format(theta_a_arcmin[0]))
    plt.plot(Px_data.k_M_edges[:-1], Px_aMZ.flatten(), label='convolved theta_a={:.2f}'.format(theta_a_arcmin[0]))
    plt.plot(Px_data.k_M_edges[:-1], Px_data.get_Px_z_T(z_ind,t_a)[0], label='data theta_a={:.2f}'.format(theta_a_arcmin[0]), linestyle='dashed')
    plt.xlabel('k [Mpc^-1]')
    plt.ylabel('P_x [Mpc]')
    plt.legend()
    plt.show()
    plt.clf()
    V_aMZ_list.append(Px_data.get_V_Z_aM(z_ind,t_a)[0])

Px_AMZ = rebin_theta(np.asarray(V_aMZ_list), np.array(Px_aMZ_list))
# plot the rebinned theory and data
plt.plot(Px_data.k_M_edges[:-1], Px_AMZ.flatten(), label='rebinned theory')
plt.plot(Px_data.k_M_edges[:-1], Px_data.get_Px_z_T(z_ind,theta_A_ind)[0], label='data', linestyle='dashed')
plt.xlabel('k [Mpc^-1]')
plt.ylabel('P_x [Mpc]')
plt.legend()
plt.show()



# %%
Px_AMZ.shape

# %%
Px_aMZ.shape

# %%
U_aMnZ.shape

# %%
np.array(Px_aMZ_list).shape

# %%
Px_amZ[0].shape

# %%
U_aMnZ.shape

# %%
Px_amZ.shape
